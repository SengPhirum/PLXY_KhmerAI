"""Optional reranking stage.

Reranking is the highest-leverage retrieval improvement per unit of latency, but
a cross-encoder costs 50-150 ms for 20 candidates on the Mac Studio and competes
for the same unified memory as the generation model.  So it is **off by default**
and enabled only when ``evaluation/evaluate_retrieval.py`` shows it earns its
latency on the Khmer golden set.

Two implementations:

``HeuristicReranker`` (default, ~0 ms)
    Signal-based rescoring using facts the embedder cannot see: exact model
    number / SKU overlap, Khmer syllable overlap, document recency, status,
    heading match and category agreement with the detected intent.  On Khmer
    support queries the exact-identifier signal alone recovers most of what a
    cross-encoder provides, because the failure mode is "retrieved the right
    product family, wrong model".

``CrossEncoderReranker``
    A real cross-encoder (``BAAI/bge-reranker-v2-m3`` handles Khmer) loaded
    through sentence-transformers.  Enable with ``reranker.enabled: true`` and
    ``reranker.backend: cross_encoder`` in ``configs/rag/retrieval.yaml``.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from common.logging import get_logger
from preprocessing.khmer_script import is_khmer_char, iter_clusters, tokenize_for_search
from preprocessing.language_mixing import SpanKind, extract_protected_spans
from rag.schemas import RetrievedChunk

log = get_logger(__name__)

__all__ = ["CrossEncoderReranker", "HeuristicReranker", "Reranker", "build_reranker"]

_IDENTIFIER_KINDS = frozenset({SpanKind.MODEL_NUMBER, SpanKind.SKU})


@dataclass(slots=True)
class RerankWeights:
    base: float = 1.0
    exact_identifier: float = 0.55
    syllable_overlap: float = 0.25
    heading_match: float = 0.10
    recency: float = 0.08
    status_active: float = 0.05


class Reranker(ABC):
    name = "abstract"

    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int | None = None
    ) -> list[RetrievedChunk]: ...


class HeuristicReranker(Reranker):
    """Zero-dependency signal reranker.  Default."""

    name = "heuristic"

    def __init__(self, weights: RerankWeights | None = None, *, as_of: date | None = None) -> None:
        self.weights = weights or RerankWeights()
        self.as_of = as_of

    def _identifiers(self, text: str) -> set[str]:
        return {
            span.normalised()
            for span in extract_protected_spans(text)
            if span.kind in _IDENTIFIER_KINDS
        }

    @staticmethod
    def _khmer_syllables(text: str) -> set[str]:
        return {c for c in iter_clusters(text) if is_khmer_char(c[0])}

    def _recency(self, effective_date: str) -> float:
        if not effective_date:
            return 0.0
        try:
            effective = date.fromisoformat(effective_date)
        except ValueError:
            return 0.0
        today = self.as_of or datetime.now(UTC).date()
        age_days = max(0, (today - effective).days)
        # Half-life of two years: a document from 2 years ago scores 0.5.
        return math.exp(-age_days / 1055.0)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        query_identifiers = self._identifiers(query)
        query_syllables = self._khmer_syllables(query)
        query_tokens = set(tokenize_for_search(query))

        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            weight = self.weights
            score = weight.base * chunk.score

            if query_identifiers:
                haystack = self._identifiers(chunk.text) | {chunk.product_id.lower()}
                overlap = len(query_identifiers & haystack) / len(query_identifiers)
                score += weight.exact_identifier * overlap

            if query_syllables:
                chunk_syllables = self._khmer_syllables(chunk.text)
                overlap = len(query_syllables & chunk_syllables) / len(query_syllables)
                score += weight.syllable_overlap * overlap

            if chunk.heading_path:
                heading_tokens = set(tokenize_for_search(" ".join(chunk.heading_path)))
                if heading_tokens & query_tokens:
                    score += weight.heading_match

            score += weight.recency * self._recency(chunk.effective_date)
            if chunk.status == "active":
                score += weight.status_active

            scored.append(chunk.model_copy(update={"rerank_score": round(score, 6)}))

        scored.sort(key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
        return scored[: top_k or len(scored)]


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker (sentence-transformers)."""

    name = "cross_encoder"

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        batch_size: int = 16,
        fallback: Reranker | None = None,
    ) -> None:
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self.fallback = fallback or HeuristicReranker()
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        try:
            model = self._load()
            scores = model.predict(
                [(query, chunk.text) for chunk in chunks],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            log.warning(
                "rag.reranker.failed_falling_back",
                extra={"model": self.model_name, "error": str(exc)},
            )
            return self.fallback.rerank(query, chunks, top_k=top_k)

        scored = [
            chunk.model_copy(update={"rerank_score": float(score)})
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        scored.sort(key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
        return scored[: top_k or len(scored)]


def build_reranker(backend: str | None = None, **kwargs: Any) -> Reranker:
    name = (backend or os.environ.get("KHMERAI_RERANKER_BACKEND", "heuristic")).lower()
    if name in ("heuristic", "none", ""):
        return HeuristicReranker()
    if name == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    raise ValueError(f"unknown reranker backend {name!r}")
