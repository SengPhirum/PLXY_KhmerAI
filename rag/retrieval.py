"""The retriever: filter -> dense + lexical -> fuse -> rerank -> confidence.

This module owns the two behaviours that keep the assistant honest:

**No-result / low-confidence logic.**  When nothing clears the confidence floor,
the retriever returns an *empty* result with ``confidence=none|low`` rather than
handing the model whatever came back.  ``server/chat_service.py`` turns that into
an "I don't have that information" answer.  Stuffing weakly-related context is
the single largest cause of confident hallucination in a support RAG system, so
it is prevented here rather than being left to the prompt.

**Conflict exposure.**  When two *active* documents about the same product and
category state different facts, both are kept and a
:class:`~rag.schemas.ConflictGroup` is attached.  The answer prompt is then
required to acknowledge the discrepancy instead of silently picking one.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from common.logging import get_logger
from preprocessing.language_mixing import SpanKind, extract_protected_spans
from preprocessing.unicode_normalization import normalize_text
from rag.bm25 import BM25Index
from rag.embeddings import EmbeddingBackend
from rag.hybrid_search import FusionConfig, fuse
from rag.reranker import HeuristicReranker, Reranker
from rag.schemas import (
    Chunk,
    ConflictGroup,
    RetrievalConfidence,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)
from rag.vector_store import VectorStore
from security.prompt_injection import scan_for_injection

log = get_logger(__name__)

__all__ = ["RetrievalConfig", "Retriever"]

_FACT_KINDS = frozenset(
    {SpanKind.CURRENCY, SpanKind.MEASUREMENT, SpanKind.NUMBER, SpanKind.MODEL_NUMBER}
)


@dataclass(slots=True)
class RetrievalConfig:
    """Runtime retrieval policy.  Loaded from ``configs/rag/retrieval.yaml``."""

    top_k: int = 6
    candidate_k: int = 24
    use_lexical: bool = True
    use_reranker: bool = False
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Confidence thresholds, calibrated in evaluation/evaluate_retrieval.py.
    min_score_to_answer: float = 0.35
    high_confidence_score: float = 0.62
    medium_confidence_score: float = 0.45
    min_chunks_for_high_confidence: int = 2

    # Security
    drop_injected_chunks: bool = True
    injection_block_threshold: float = 0.6

    max_context_tokens: int = 3200
    detect_conflicts: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalConfig:
        retrieval = data.get("retrieval", data) or {}
        fusion_raw = retrieval.get("fusion", {}) or {}
        fusion = FusionConfig(
            strategy=fusion_raw.get("strategy", "rrf"),
            dense_weight=float(fusion_raw.get("dense_weight", 0.65)),
            lexical_weight=float(fusion_raw.get("lexical_weight", 0.35)),
            rrf_k=int(fusion_raw.get("rrf_k", 60)),
            top_k=int(retrieval.get("candidate_k", 24)),
        )
        confidence = retrieval.get("confidence", {}) or {}
        security = retrieval.get("security", {}) or {}
        return cls(
            top_k=int(retrieval.get("top_k", 6)),
            candidate_k=int(retrieval.get("candidate_k", 24)),
            use_lexical=bool(retrieval.get("use_lexical", True)),
            use_reranker=bool((retrieval.get("reranker", {}) or {}).get("enabled", False)),
            fusion=fusion,
            min_score_to_answer=float(confidence.get("min_score_to_answer", 0.35)),
            high_confidence_score=float(confidence.get("high", 0.62)),
            medium_confidence_score=float(confidence.get("medium", 0.45)),
            min_chunks_for_high_confidence=int(confidence.get("min_chunks_for_high", 2)),
            drop_injected_chunks=bool(security.get("drop_injected_chunks", True)),
            injection_block_threshold=float(security.get("injection_block_threshold", 0.6)),
            max_context_tokens=int(retrieval.get("max_context_tokens", 3200)),
            detect_conflicts=bool(retrieval.get("detect_conflicts", True)),
        )


class Retriever:
    """Hybrid retriever over one vector store plus its BM25 sidecar."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingBackend,
        *,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
        index_version: str = "unknown",
        build_lexical_index: bool = True,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.reranker = reranker or HeuristicReranker()
        self.index_version = index_version
        self._bm25: BM25Index | None = None
        self._chunks_by_id: dict[str, Chunk] = {}
        if build_lexical_index:
            self.rebuild_lexical_index()

    # -- lexical sidecar ----------------------------------------------------
    def rebuild_lexical_index(self) -> None:
        """(Re)build the BM25 index and the id->chunk map from the vector store."""
        started = time.perf_counter()
        chunks = self.store.all_chunks()
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        if not self.config.use_lexical:
            self._bm25 = None
            return
        index = BM25Index()
        index.add_many((c.chunk_id, c.contextual_text()) for c in chunks)
        index.finalise()
        self._bm25 = index
        log.info(
            "rag.lexical_index.built",
            extra={
                "chunks": len(chunks),
                "vocabulary": index.vocabulary_size,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

    # -- query --------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
        as_of: date | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        cfg = self.config
        active_filters = filters or RetrievalFilters()
        limit = top_k or cfg.top_k
        normalised_query = normalize_text(query)

        result = RetrievalResult(
            query=query,
            filters_applied=active_filters.model_dump(mode="json"),
            strategy=cfg.fusion.strategy if cfg.use_lexical else "dense_only",
            index_version=self.index_version,
        )
        if not normalised_query.strip() or self.store.size == 0:
            result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return result

        def predicate(metadata: dict[str, Any]) -> bool:
            return active_filters.matches(metadata, as_of=as_of)

        # 1. Dense retrieval over the filtered candidate set.
        query_vector = self.embedder.embed_query(normalised_query)
        dense_hits = self.store.search(
            query_vector, top_k=cfg.candidate_k, predicate=predicate
        )
        dense_scores = {chunk.chunk_id: score for chunk, score in dense_hits}
        for chunk, _ in dense_hits:
            self._chunks_by_id.setdefault(chunk.chunk_id, chunk)

        # 2. Lexical retrieval, restricted to the same eligible set.
        lexical_scores: dict[str, float] = {}
        if cfg.use_lexical and self._bm25 is not None:
            allowed = {
                chunk_id
                for chunk_id, chunk in self._chunks_by_id.items()
                if predicate(chunk.metadata)
            }
            lexical_scores = dict(
                self._bm25.search(normalised_query, top_k=cfg.candidate_k, allowed=allowed)
            )

        # 3. Fusion.
        fused = fuse(
            sorted(dense_scores.items(), key=lambda kv: -kv[1]),
            sorted(lexical_scores.items(), key=lambda kv: -kv[1]),
            cfg.fusion,
        )
        if not fused:
            result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return result

        # 4. Materialise, dropping any chunk that carries an injection payload.
        candidates: list[RetrievedChunk] = []
        for chunk_id, score in fused:
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            if cfg.drop_injected_chunks:
                scan = scan_for_injection(
                    chunk.text, block_threshold=cfg.injection_block_threshold
                )
                if scan.blocked:
                    result.dropped_for_injection += 1
                    log.warning(
                        "rag.retrieval.injected_chunk_dropped",
                        extra={
                            "chunk_id": chunk_id,
                            "document_id": chunk.document_id,
                            "matches": [m.name for m in scan.matches],
                        },
                    )
                    continue
            candidates.append(
                RetrievedChunk.from_chunk(
                    chunk,
                    score=score,
                    dense_score=dense_scores.get(chunk_id, 0.0),
                    lexical_score=lexical_scores.get(chunk_id, 0.0),
                )
            )

        # 5. Optional reranking, then truncate to the context budget.
        if cfg.use_reranker and candidates:
            candidates = self.reranker.rerank(normalised_query, candidates, top_k=limit * 2)

        selected = self._fit_context(candidates, limit)

        # 6. Confidence and conflicts.
        confidence, confidence_score = self._score_confidence(selected, dense_scores)
        result.confidence = confidence
        result.confidence_score = round(confidence_score, 4)

        if confidence is RetrievalConfidence.LOW or confidence is RetrievalConfidence.NONE:
            # Deliberately withhold weak context instead of letting the model
            # rationalise over it.
            log.info(
                "rag.retrieval.low_confidence",
                extra={
                    "score": round(confidence_score, 4),
                    "candidates": len(selected),
                    "threshold": cfg.min_score_to_answer,
                },
            )
            result.chunks = []
        else:
            result.chunks = selected
            if cfg.detect_conflicts:
                result.conflicts = detect_conflicts(selected)

        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    # -- helpers ------------------------------------------------------------
    def _fit_context(self, candidates: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
        """Take the best chunks that fit the context budget."""
        budget = self.config.max_context_tokens
        chosen: list[RetrievedChunk] = []
        used = 0
        for chunk in candidates[: max(limit * 3, limit)]:
            cost = int(chunk.metadata.get("token_estimate", 0)) or max(1, len(chunk.text) // 3)
            if chosen and used + cost > budget:
                continue
            chosen.append(chunk)
            used += cost
            if len(chosen) >= limit:
                break
        return chosen

    def _score_confidence(
        self, chunks: list[RetrievedChunk], dense_scores: dict[str, float]
    ) -> tuple[RetrievalConfidence, float]:
        """Confidence from the *dense* similarity of the best hits.

        The fused score is rank-based and therefore not comparable across
        queries, so it cannot be thresholded.  Cosine similarity from the
        embedder is comparable, which is what makes a fixed floor meaningful.
        """
        if not chunks:
            return RetrievalConfidence.NONE, 0.0

        cfg = self.config
        top_dense = [dense_scores.get(c.chunk_id, 0.0) for c in chunks]
        best = max(top_dense) if top_dense else 0.0
        supporting = sum(1 for s in top_dense if s >= cfg.medium_confidence_score)

        if best < cfg.min_score_to_answer:
            return RetrievalConfidence.LOW, best
        if best >= cfg.high_confidence_score and supporting >= cfg.min_chunks_for_high_confidence:
            return RetrievalConfidence.HIGH, best
        if best >= cfg.medium_confidence_score:
            return RetrievalConfidence.MEDIUM, best
        return RetrievalConfidence.MEDIUM, best


def _fact_values(text: str) -> set[str]:
    return {
        span.normalised()
        for span in extract_protected_spans(text)
        if span.kind in _FACT_KINDS
    }


def _identity(chunk: RetrievedChunk) -> str:
    return "|".join(
        [
            (chunk.product_id or chunk.metadata.get("service_id", "") or "general").lower(),
            (chunk.category or "").lower(),
        ]
    )


def _version_key(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in str(version).replace("-", ".").split("."):
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def detect_conflicts(chunks: list[RetrievedChunk]) -> list[ConflictGroup]:
    """Find retrieved chunks that state different facts about the same thing.

    Only chunks from *different documents* with ``status == active`` are
    compared: two chunks of one document naturally state different facts, and a
    superseded document is not a conflict.
    """
    grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.status == "active":
            grouped[_identity(chunk)].append(chunk)

    conflicts: list[ConflictGroup] = []
    for identity, group in grouped.items():
        documents = {c.document_id for c in group}
        if len(documents) < 2:
            continue
        by_document: dict[str, set[str]] = defaultdict(set)
        for chunk in group:
            by_document[chunk.document_id] |= _fact_values(chunk.text)
        populated = {d: f for d, f in by_document.items() if f}
        if len(populated) < 2:
            continue
        distinct = {frozenset(f) for f in populated.values()}
        if len(distinct) < 2:
            continue

        newest = max(group, key=lambda c: (_version_key(c.version), c.effective_date))
        shared = set.intersection(*populated.values())
        conflicts.append(
            ConflictGroup(
                identity=identity,
                chunk_ids=sorted(c.chunk_id for c in group),
                document_ids=sorted(documents),
                versions=sorted({c.version for c in group if c.version}),
                differing_values=sorted(
                    {v for values in populated.values() for v in values} - shared
                )[:20],
                newest_document_id=newest.document_id,
            )
        )
    return conflicts
