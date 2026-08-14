"""Grounding evaluation: is every business claim supported by retrieved context?

Runs the answer and the context it was given through
``rag.citations.verify_grounding`` and reports grounding precision, the rate of
unsupported claims, and citation hygiene.

Unlike the hallucination evaluator (which asks "did it refuse when it should?"),
this asks "when it *did* answer, was every claim traceable?".

    python -m evaluation.evaluate_grounding --backend api
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import load_config
from common.logging import get_logger
from evaluation.metrics import aggregate
from evaluation.runner import build_runner, load_golden, stamp_report, write_report
from evaluation.schemas import EvalReport, GoldenItem, ItemResult, ModelAnswer
from preprocessing.language_mixing import verify_protected_spans
from rag.citations import verify_grounding
from rag.embeddings import build_embedder
from rag.retrieval import Retriever
from rag.schemas import RetrievalFilters, RetrievedChunk
from rag.vector_store import LocalVectorStore

log = get_logger(__name__)

__all__ = ["evaluate_grounding", "main", "score_item"]


def _load_retriever(index_dir: str | Path) -> Retriever:
    directory = Path(index_dir)
    store = LocalVectorStore.load(directory / "vectors")
    embedder = build_embedder()
    return Retriever(store, embedder, index_version=directory.name)


def score_item(item: GoldenItem, answer: ModelAnswer, chunks: list[RetrievedChunk]) -> ItemResult:
    result = ItemResult(
        item_id=item.id,
        category=item.category,
        passed=False,
        answer=answer.answer,
        latency_ms=answer.latency_ms,
    )
    if answer.error:
        result.failures.append(f"error:{answer.error}")
        return result
    if not answer.answer.strip():
        result.failures.append("empty_answer")
        return result

    report = verify_grounding(answer.answer, chunks)
    result.scores["grounding_precision"] = report.grounding_precision
    result.scores["claims"] = float(report.total_claims)
    result.scores["grounded"] = 1.0 if report.is_grounded else 0.0

    if report.unsupported_sentences:
        result.failures.append(f"unsupported_claims:{len(report.unsupported_sentences)}")
    if report.invalid_markers:
        result.failures.append("invalid_citations:" + ",".join(report.invalid_markers[:3]))

    if chunks:
        context = "\n".join(c.text for c in chunks)
        ok, missing = verify_protected_spans(context, answer.answer)
        result.scores["identifiers_preserved"] = 1.0 if ok else 0.0
        if not ok:
            result.failures.append("corrupted_identifiers:" + ",".join(s.text for s in missing[:3]))

    if item.expected_document_ids and answer.sources:
        expected = set(item.expected_document_ids)
        result.scores["cited_expected_source"] = 1.0 if expected & set(answer.sources) else 0.0

    # Citing at all when context was supplied is part of the contract.
    if chunks and report.total_claims > 0:
        result.scores["cited_anything"] = 1.0 if report.citation_markers else 0.0
        if not report.citation_markers:
            result.failures.append("no_citation_markers")

    result.passed = not result.failures
    return result


def evaluate_grounding(
    golden_path: str | Path,
    backend: str = "static",
    *,
    index_dir: str | Path | None = None,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    answers: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> EvalReport:
    items = load_golden(golden_path)
    retriever = _load_retriever(index_dir) if index_dir else None
    runner = build_runner(backend, model=model, base_url=base_url, api_key=api_key, answers=answers)

    results: list[ItemResult] = []
    try:
        for item in items:
            answer = runner.answer(item)
            chunks: list[RetrievedChunk] = []
            if retriever is not None:
                filters = RetrievalFilters()
                if item.product_id:
                    filters.product_id = item.product_id
                chunks = retriever.retrieve(item.question, filters=filters).chunks
            results.append(score_item(item, answer, chunks))
    finally:
        runner.close()

    report = EvalReport(
        name="grounding",
        model=model or runner.name,
        items=len(results),
        passed=sum(1 for r in results if r.passed),
        results=results,
        aggregate=aggregate([r.scores for r in results]),
    )
    unsupported = sum(
        1 for r in results if any(f.startswith("unsupported_claims") for f in r.failures)
    )
    report.aggregate["unsupported_claim_rate"] = round(
        unsupported / len(results) if results else 0.0, 4
    )

    gates = thresholds or {}
    report.add_gate(
        "grounding_precision",
        report.aggregate.get("grounding_precision", 0.0),
        gates.get("grounding_precision_min", 0.90),
    )
    report.add_gate(
        "unsupported_claim_rate",
        report.aggregate["unsupported_claim_rate"],
        gates.get("unsupported_claim_rate_max", 0.02),
        higher_is_better=False,
    )
    if retriever is None:
        report.notes = (
            "No --index-dir was supplied, so grounding was scored against an empty "
            "context: every claim counts as unsupported. Pass --index-dir "
            "data/index/ACTIVE for a meaningful result."
        )
    return stamp_report(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.evaluate_grounding")
    parser.add_argument("--golden", default="evaluation/golden/customer_support.jsonl")
    parser.add_argument("--index-dir", default="data/index/ACTIVE")
    parser.add_argument("--backend", choices=("api", "ollama", "static"), default="static")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--answers", default=None)
    parser.add_argument("--report-name", default="grounding")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)

    index_dir = args.index_dir if args.index_dir and Path(args.index_dir).exists() else None
    gates = (load_config(args.config).get("slo", {}) or {}).get("quality_gates", {}) or {}
    report = evaluate_grounding(
        args.golden,
        args.backend,
        index_dir=index_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        answers=args.answers,
        thresholds={
            "grounding_precision_min": float(gates.get("grounding_precision_min", 0.90)),
            "unsupported_claim_rate_max": float(gates.get("unsupported_claim_rate_max", 0.02)),
        },
    )
    write_report(report, args.report_name)
    print(report.to_markdown())
    return 0 if report.gates_passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
