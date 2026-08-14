"""Retrieval evaluation, embedding benchmark, chunking sweep, confidence calibration.

Four modes, all writing machine-readable reports:

``(default)``            Recall@K / MRR / nDCG on the Khmer golden set
``--compare-embedders``  benchmark the candidates in configs/rag/embedding.yaml
``--sweep-chunking``     measure 400/600/800 token chunks and overlaps
``--calibrate-confidence`` derive the confidence thresholds from measured scores

    python -m evaluation.evaluate_retrieval --index-dir data/index/ACTIVE
    python -m evaluation.evaluate_retrieval --sweep-chunking 400,600,800 --overlap 0,50,100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from common.config import load_config
from common.io import write_json
from common.logging import get_logger
from common.paths import EVAL_REPORT_DIR, ensure_dir
from company_data.schema import CompanyDocument
from evaluation.metrics import latency_percentiles, mrr, ndcg_at_k, precision_at_k, recall_at_k
from evaluation.runner import load_golden, stamp_report, write_report
from evaluation.schemas import EvalReport, ItemResult
from rag.embeddings import build_embedder
from rag.hybrid_search import FusionConfig
from rag.ingestion import IngestionSettings, build_index, load_records
from rag.retrieval import RetrievalConfig, Retriever
from rag.schemas import RetrievalFilters
from rag.vector_store import LocalVectorStore

log = get_logger(__name__)

__all__ = ["evaluate_retrieval", "sweep_chunking", "compare_embedders", "calibrate_confidence", "main"]

_K_VALUES = (1, 3, 5, 10)


def _expected(item: Any) -> list[str]:
    if item.expected_document_ids:
        return list(item.expected_document_ids)
    return []


def evaluate_retrieval(
    index_dir: str | Path,
    golden_path: str | Path,
    *,
    strategy: str = "rrf",
    top_k: int = 10,
    thresholds: dict[str, float] | None = None,
) -> EvalReport:
    """Recall@K / MRR / nDCG over the Khmer golden set."""
    directory = Path(index_dir)
    store = LocalVectorStore.load(directory / "vectors")
    embedder = build_embedder()
    config = RetrievalConfig(
        top_k=top_k,
        candidate_k=max(top_k * 4, 24),
        fusion=FusionConfig(strategy=strategy, top_k=max(top_k * 4, 24)),
        # Confidence gating is a *serving* policy; measuring raw retrieval quality
        # means disabling it, otherwise recall is confounded with the threshold.
        min_score_to_answer=0.0,
        medium_confidence_score=0.0,
    )
    retriever = Retriever(store, embedder, config=config, index_version=directory.name)

    items = load_golden(golden_path)
    scored = [i for i in items if _expected(i) or i.expected_product_id]
    results: list[ItemResult] = []
    latencies: list[float] = []
    per_k: dict[int, list[float]] = {k: [] for k in _K_VALUES}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    empty = 0

    for item in scored:
        filters = RetrievalFilters()
        if item.product_id:
            filters.product_id = item.product_id
        started = time.perf_counter()
        result = retriever.retrieve(item.question, filters=filters, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        retrieved = [c.document_id for c in result.chunks]
        relevant = _expected(item)
        if not relevant and item.expected_product_id:
            relevant = [
                c.document_id
                for c in result.chunks
                if c.product_id.upper() == item.expected_product_id.upper()
            ]
        if not result.chunks:
            empty += 1

        item_result = ItemResult(item_id=item.id, category=item.category, passed=False)
        for k in _K_VALUES:
            value = recall_at_k(retrieved, relevant, k)
            per_k[k].append(value)
            item_result.scores[f"recall@{k}"] = value
        rr = mrr(retrieved, relevant)
        nd = ndcg_at_k(retrieved, relevant, 10)
        reciprocal_ranks.append(rr)
        ndcgs.append(nd)
        item_result.scores["mrr"] = rr
        item_result.scores["ndcg@10"] = nd
        item_result.scores["precision@5"] = precision_at_k(retrieved, relevant, 5)
        item_result.passed = item_result.scores["recall@5"] > 0
        if not item_result.passed:
            item_result.failures.append("not_retrieved_in_top_5")
        results.append(item_result)

    report = EvalReport(
        name="retrieval",
        model=f"{embedder.name}:{embedder.model}",
        items=len(results),
        passed=sum(1 for r in results if r.passed),
        results=results,
    )
    report.aggregate = {
        **{
            f"recall@{k}": round(sum(v) / len(v), 4) if v else 0.0
            for k, v in per_k.items()
        },
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        "ndcg@10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "empty_result_rate": round(empty / len(scored), 4) if scored else 0.0,
        "chunks_indexed": float(store.size),
    }
    report.aggregate.update({f"latency_ms_{k}": v for k, v in latency_percentiles(latencies).items()})

    gates = thresholds or {}
    report.add_gate(
        "recall@5", report.aggregate["recall@5"], gates.get("retrieval_recall_at_5_min", 0.85)
    )
    if len(scored) < len(items):
        report.notes = (
            f"{len(items) - len(scored)} golden item(s) declare no expected document and were "
            "excluded from the retrieval metrics."
        )
    return stamp_report(report)


def sweep_chunking(
    records_path: str | Path,
    golden_path: str | Path,
    *,
    chunk_sizes: list[int],
    overlaps: list[int],
    output_root: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build one index per (chunk_size, overlap) and measure retrieval on each.

    This is what §Phase 10 means by "do not choose a chunk size only by
    intuition" - the winner is whichever configuration measures best here.
    """
    documents: list[CompanyDocument] = load_records(records_path)
    if not documents:
        raise ValueError(f"no company records in {records_path}")

    rows: list[dict[str, Any]] = []
    for size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= size:
                continue
            version = f"sweep-{size}-{overlap}"
            settings = IngestionSettings(
                chunk_size=size,
                chunk_overlap=overlap,
                min_chunk_size=min(80, size // 4),
                index_root=output_root,
                allow_non_semantic_embedder=True,
            )
            store, manifest = build_index(
                documents, settings=settings, index_version=version, source_file=str(records_path)
            )
            if isinstance(store, LocalVectorStore):
                store.persist()
            report = evaluate_retrieval(output_root / version, golden_path, top_k=top_k)
            rows.append(
                {
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                    "chunks": manifest.chunks,
                    "recall@1": report.aggregate.get("recall@1", 0.0),
                    "recall@3": report.aggregate.get("recall@3", 0.0),
                    "recall@5": report.aggregate.get("recall@5", 0.0),
                    "mrr": report.aggregate.get("mrr", 0.0),
                    "ndcg@10": report.aggregate.get("ndcg@10", 0.0),
                    "latency_ms_p95": report.aggregate.get("latency_ms_p95", 0.0),
                }
            )
            log.info("evaluation.chunking_sweep.point", extra=rows[-1])

    best = max(rows, key=lambda r: (r["recall@5"], r["mrr"])) if rows else {}
    return {
        "grid": rows,
        "best": best,
        "recommendation": (
            f"chunk_size={best.get('chunk_size')} chunk_overlap={best.get('chunk_overlap')}"
            if best
            else "no result"
        ),
    }


def compare_embedders(
    records_path: str | Path,
    golden_path: str | Path,
    *,
    candidates: list[dict[str, Any]],
    output_root: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    """Benchmark each configured embedding candidate on the same golden set."""
    documents = load_records(records_path)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        identifier = str(candidate.get("id", "unknown"))
        try:
            embedder = build_embedder(
                str(candidate.get("backend", "ollama")),
                str(candidate.get("model", "")),
                int(candidate.get("dim", 1024)),
            )
            settings = IngestionSettings(
                index_root=output_root, allow_non_semantic_embedder=not embedder.is_semantic
            )
            version = f"embed-{identifier}"
            started = time.perf_counter()
            store, manifest = build_index(
                documents, settings=settings, index_version=version, embedder=embedder
            )
            if isinstance(store, LocalVectorStore):
                store.persist()
            build_seconds = time.perf_counter() - started
            report = evaluate_retrieval(output_root / version, golden_path, top_k=top_k)
            rows.append(
                {
                    "id": identifier,
                    "backend": embedder.name,
                    "model": embedder.model,
                    "dim": embedder.dim,
                    "chunks": manifest.chunks,
                    "index_build_seconds": round(build_seconds, 2),
                    "recall@1": report.aggregate.get("recall@1", 0.0),
                    "recall@5": report.aggregate.get("recall@5", 0.0),
                    "mrr": report.aggregate.get("mrr", 0.0),
                    "ndcg@10": report.aggregate.get("ndcg@10", 0.0),
                }
            )
        except Exception as exc:  # noqa: BLE001 - a missing model must not stop the sweep
            rows.append({"id": identifier, "error": f"{type(exc).__name__}: {exc}"})
            log.error("evaluation.embedder.failed", extra={"id": identifier, "error": str(exc)})

    usable = [r for r in rows if "error" not in r]
    best = max(usable, key=lambda r: (r["recall@5"], r["mrr"])) if usable else {}
    return {"candidates": rows, "best": best, "recommendation": best.get("id", "no result")}


def calibrate_confidence(
    index_dir: str | Path, golden_path: str | Path, *, percentile: float = 0.10
) -> dict[str, Any]:
    """Derive confidence thresholds from measured dense scores.

    Answerable golden questions produce a distribution of top-1 similarities;
    the ``min_score_to_answer`` floor is set below its lower tail so that genuine
    questions are not gated out, while unanswerable probes sit below it.
    """
    directory = Path(index_dir)
    store = LocalVectorStore.load(directory / "vectors")
    embedder = build_embedder()
    retriever = Retriever(
        store,
        embedder,
        config=RetrievalConfig(top_k=5, min_score_to_answer=0.0, medium_confidence_score=0.0),
        index_version=directory.name,
    )

    answerable: list[float] = []
    unanswerable: list[float] = []
    for item in load_golden(golden_path):
        result = retriever.retrieve(item.question)
        score = max((c.dense_score for c in result.chunks), default=0.0)
        (unanswerable if item.is_unanswerable else answerable).append(score)

    answerable.sort()
    unanswerable.sort()
    index = max(0, int(len(answerable) * percentile) - 1)
    floor = answerable[index] if answerable else 0.35
    return {
        "answerable": latency_percentiles(answerable),
        "unanswerable": latency_percentiles(unanswerable),
        "suggested": {
            "min_score_to_answer": round(max(0.0, floor * 0.95), 4),
            "medium": round(answerable[len(answerable) // 2], 4) if answerable else 0.45,
            "high": round(answerable[int(len(answerable) * 0.75)], 4) if answerable else 0.62,
        },
        "note": (
            "Apply these to configs/rag/retrieval.yaml -> retrieval.confidence. "
            "Re-calibrate whenever the embedding model or the corpus changes."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.evaluate_retrieval")
    parser.add_argument("--index-dir", default="data/index/ACTIVE")
    parser.add_argument("--golden", default="evaluation/golden/customer_support.jsonl")
    parser.add_argument("--records", default="data/interim/company_records.jsonl")
    parser.add_argument("--config", default="configs/rag/retrieval.yaml")
    parser.add_argument("--strategy", default="rrf", choices=("rrf", "weighted", "dense_only", "lexical_only"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sweep-chunking", default=None, help="comma-separated chunk sizes")
    parser.add_argument("--overlap", default="0,50,100", help="comma-separated overlaps")
    parser.add_argument("--compare-embedders", action="store_true")
    parser.add_argument("--calibrate-confidence", action="store_true")
    parser.add_argument("--report-name", default="retrieval")
    args = parser.parse_args(argv)

    output_root = ensure_dir(EVAL_REPORT_DIR / "sweeps")

    if args.sweep_chunking:
        sizes = [int(s) for s in args.sweep_chunking.split(",") if s.strip()]
        overlaps = [int(s) for s in args.overlap.split(",") if s.strip()]
        payload = sweep_chunking(
            args.records, args.golden, chunk_sizes=sizes, overlaps=overlaps, output_root=output_root
        )
        write_json(EVAL_REPORT_DIR / "chunking_sweep.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.compare_embedders:
        candidates = load_config("configs/rag/embedding.yaml").get("candidates", [])
        payload = compare_embedders(
            args.records, args.golden, candidates=candidates, output_root=output_root
        )
        write_json(EVAL_REPORT_DIR / "embedding_benchmark.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.calibrate_confidence:
        payload = calibrate_confidence(args.index_dir, args.golden)
        write_json(EVAL_REPORT_DIR / "confidence_calibration.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    gates = (load_config("configs/base.yaml").get("slo", {}) or {}).get("quality_gates", {}) or {}
    report = evaluate_retrieval(
        args.index_dir,
        args.golden,
        strategy=args.strategy,
        top_k=args.top_k,
        thresholds={
            "retrieval_recall_at_5_min": float(gates.get("retrieval_recall_at_5_min", 0.85))
        },
    )
    write_report(report, args.report_name)
    print(report.to_markdown())
    return 0 if report.gates_passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
