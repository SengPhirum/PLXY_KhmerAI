"""Evaluation framework: Khmer quality, support accuracy, grounding, retrieval, regression."""

from evaluation.metrics import (
    chrf,
    khmer_fluency,
    latency_percentiles,
    mrr,
    ndcg_at_k,
    recall_at_k,
    token_f1,
)
from evaluation.runner import Runner, StaticRunner, build_runner, load_golden, write_report
from evaluation.schemas import (
    RUBRIC_DIMENSIONS,
    ComparisonReport,
    EvalCategory,
    EvalReport,
    GoldenItem,
    HumanRubric,
    ItemResult,
    ModelAnswer,
)

__all__ = [
    "RUBRIC_DIMENSIONS",
    "ComparisonReport",
    "EvalCategory",
    "EvalReport",
    "GoldenItem",
    "HumanRubric",
    "ItemResult",
    "ModelAnswer",
    "Runner",
    "StaticRunner",
    "build_runner",
    "chrf",
    "khmer_fluency",
    "latency_percentiles",
    "load_golden",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "token_f1",
    "write_report",
]
