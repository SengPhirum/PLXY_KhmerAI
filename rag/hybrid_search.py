"""Fusion of dense and lexical retrieval.

Two fusion strategies, both implemented and both benchmarkable through
``evaluation/evaluate_retrieval.py``:

``rrf`` (Reciprocal Rank Fusion, default)
    ``score = Σ weight / (k + rank)``.  Rank-based, so it needs no score
    calibration between a cosine similarity in [-1, 1] and an unbounded BM25
    score.  That property is why it is the default: the two scales genuinely are
    incomparable, and any weighted-sum scheme silently changes meaning when the
    embedding model is swapped.

``weighted``
    Min-max normalises each score list to [0, 1] within the candidate set and
    takes a weighted sum.  Retains score magnitude, which the confidence
    estimator can use, but is sensitive to outliers in the candidate set.

Both fuse over the *union* of candidates, so a chunk found by only one retriever
still competes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = ["FusionConfig", "reciprocal_rank_fusion", "weighted_fusion", "fuse"]

FusionStrategy = Literal["rrf", "weighted", "dense_only", "lexical_only"]


@dataclass(slots=True)
class FusionConfig:
    strategy: FusionStrategy = "rrf"
    dense_weight: float = 0.65
    lexical_weight: float = 0.35
    rrf_k: int = 60
    top_k: int = 10

    def __post_init__(self) -> None:
        if self.dense_weight < 0 or self.lexical_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.dense_weight + self.lexical_weight == 0:
            raise ValueError("at least one fusion weight must be positive")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return dict.fromkeys(scores, 1.0)
    return {k: (v - low) / (high - low) for k, v in scores.items()}


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[Sequence[str], float]], *, k: int = 60
) -> dict[str, float]:
    """Fuse ranked ID lists.  Each entry is ``(ids_in_rank_order, weight)``."""
    fused: dict[str, float] = {}
    for ids, weight in ranked_lists:
        if weight <= 0:
            continue
        for rank, identifier in enumerate(ids, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + weight / (k + rank)
    return fused


def weighted_fusion(
    dense: dict[str, float], lexical: dict[str, float], *, dense_weight: float, lexical_weight: float
) -> dict[str, float]:
    """Min-max normalise each list, then take the weighted sum over their union."""
    dense_norm = _min_max(dense)
    lexical_norm = _min_max(lexical)
    total_weight = dense_weight + lexical_weight
    fused: dict[str, float] = {}
    for identifier in set(dense_norm) | set(lexical_norm):
        fused[identifier] = (
            dense_weight * dense_norm.get(identifier, 0.0)
            + lexical_weight * lexical_norm.get(identifier, 0.0)
        ) / total_weight
    return fused


def fuse(
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    config: FusionConfig | None = None,
) -> list[tuple[str, float]]:
    """Fuse two scored lists into one ranking.

    Returns ``(chunk_id, fused_score)`` sorted by descending score, capped at
    ``config.top_k``.  Ties break on chunk_id so the ordering is deterministic,
    which matters for reproducible evaluation runs.
    """
    cfg = config or FusionConfig()

    if cfg.strategy == "dense_only":
        fused = dict(dense)
    elif cfg.strategy == "lexical_only":
        fused = dict(lexical)
    elif cfg.strategy == "weighted":
        fused = weighted_fusion(
            dict(dense),
            dict(lexical),
            dense_weight=cfg.dense_weight,
            lexical_weight=cfg.lexical_weight,
        )
    else:
        fused = reciprocal_rank_fusion(
            [
                ([i for i, _ in dense], cfg.dense_weight),
                ([i for i, _ in lexical], cfg.lexical_weight),
            ],
            k=cfg.rrf_k,
        )

    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[: cfg.top_k]
