"""Evaluation data types and the human review rubric (Phase 17)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EvalCategory",
    "GoldenItem",
    "ModelAnswer",
    "ItemResult",
    "EvalReport",
    "HumanRubric",
    "RUBRIC_DIMENSIONS",
    "ComparisonReport",
]


class EvalCategory(StrEnum):
    KHMER_GENERAL = "khmer_general"
    CUSTOMER_SUPPORT = "customer_support"
    HALLUCINATION = "hallucination"
    CODE_SWITCH = "code_switch"
    MULTITURN = "multiturn"
    ADVERSARIAL = "adversarial"
    NOISE = "noise"
    RETRIEVAL = "retrieval"


class GoldenItem(BaseModel):
    """One sealed evaluation case.

    ``expected_behaviour`` is what makes an unanswerable case checkable: for a
    fake product the correct answer is not a string, it is *refusing to invent
    one*, so the check is behavioural rather than a text comparison.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    category: EvalCategory = EvalCategory.CUSTOMER_SUPPORT
    question: str
    turns: list[dict[str, str]] = Field(default_factory=list)
    reference_answer: str = ""
    expected_behaviour: str = "answer"
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_product_id: str = ""
    product_id: str | None = None
    intent: str = "general_inquiry"
    language: str = "km"
    notes: str = ""

    @property
    def is_unanswerable(self) -> bool:
        return self.expected_behaviour in ("refuse", "uncertainty", "escalate")


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")

    item_id: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    escalation_required: bool = False
    error: str = ""

    @property
    def tokens_per_second(self) -> float:
        seconds = self.latency_ms / 1000.0
        return round(self.completion_tokens / seconds, 2) if seconds > 0 else 0.0


class ItemResult(BaseModel):
    """Per-item verdict with every measured sub-score."""

    model_config = ConfigDict(extra="allow")

    item_id: str
    category: EvalCategory
    passed: bool
    scores: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    answer: str = ""
    latency_ms: float = 0.0
    needs_human_review: bool = False


class EvalReport(BaseModel):
    """Machine-readable report; ``to_markdown`` renders the human summary."""

    model_config = ConfigDict(extra="allow")

    name: str
    model: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    code_commit: str = "unknown"
    prompt_version: str = ""
    index_version: str = ""
    dataset_version: str = ""
    items: int = 0
    passed: int = 0
    results: list[ItemResult] = Field(default_factory=list)
    aggregate: dict[str, float] = Field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    gates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: str = ""

    @property
    def pass_rate(self) -> float:
        return self.passed / self.items if self.items else 0.0

    @property
    def gates_passed(self) -> bool:
        return all(g.get("passed", False) for g in self.gates.values())

    def add_gate(self, name: str, value: float, threshold: float, *, higher_is_better: bool = True) -> None:
        passed = value >= threshold if higher_is_better else value <= threshold
        self.gates[name] = {
            "value": round(value, 4),
            "threshold": threshold,
            "direction": "min" if higher_is_better else "max",
            "passed": passed,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation report - {self.name}",
            "",
            f"- **Model**: `{self.model or 'n/a'}`",
            f"- **Generated**: {self.created_at.isoformat()}",
            f"- **Code commit**: `{self.code_commit}`",
            f"- **Prompt version**: {self.prompt_version or 'n/a'}",
            f"- **Index version**: {self.index_version or 'n/a'}",
            f"- **Dataset version**: {self.dataset_version or 'n/a'}",
            "",
            f"**Result: {self.passed}/{self.items} passed ({self.pass_rate:.1%})**",
            "",
        ]
        if self.aggregate:
            lines += ["## Aggregate metrics", "", "| Metric | Value |", "|---|---:|"]
            lines += [f"| {k} | {v:.4f} |" for k, v in sorted(self.aggregate.items())]
            lines.append("")
        if self.by_category:
            lines += ["## By category", "", "| Category | Items | Pass rate |", "|---|---:|---:|"]
            for category, stats in sorted(self.by_category.items()):
                lines.append(
                    f"| {category} | {int(stats.get('items', 0))} | {stats.get('pass_rate', 0):.1%} |"
                )
            lines.append("")
        if self.gates:
            lines += [
                "## Release gates",
                "",
                "| Gate | Value | Threshold | Result |",
                "|---|---:|---:|---|",
            ]
            for name, gate in sorted(self.gates.items()):
                symbol = "PASS" if gate["passed"] else "**FAIL**"
                direction = ">=" if gate["direction"] == "min" else "<="
                lines.append(
                    f"| {name} | {gate['value']} | {direction} {gate['threshold']} | {symbol} |"
                )
            lines.append("")
            lines.append(
                f"**Overall gate status: {'PASS' if self.gates_passed else 'FAIL'}**"
            )
            lines.append("")
        failures = [r for r in self.results if not r.passed]
        if failures:
            lines += ["## Failures", "", "| Item | Category | Reasons |", "|---|---|---|"]
            for result in failures[:50]:
                lines.append(
                    f"| `{result.item_id}` | {result.category} | {', '.join(result.failures)} |"
                )
            if len(failures) > 50:
                lines.append(f"\n_{len(failures) - 50} further failures omitted._")
            lines.append("")
        review = [r for r in self.results if r.needs_human_review]
        if review:
            lines += [
                "## Needs native-speaker review",
                "",
                f"{len(review)} item(s) could not be judged automatically. "
                "Score them with the rubric in `docs/evaluation.md`.",
                "",
            ]
        if self.notes:
            lines += ["## Notes", "", self.notes, ""]
        return "\n".join(lines)


# --- human evaluation -------------------------------------------------------
RUBRIC_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("naturalness", "Does it read like a Khmer speaker wrote it, not a translation?"),
    ("correctness", "Are the facts right, given the source documents?"),
    ("helpfulness", "Does it actually resolve the customer's need?"),
    ("professional_tone", "Polite, calm and appropriate for a support channel?"),
    ("faithfulness", "Every claim traceable to a source; nothing invented?"),
    ("clarity", "Unambiguous, well structured and appropriately concise?"),
)


class HumanRubric(BaseModel):
    """Native Khmer reviewer scores, 1-5 per dimension (§Phase 17)."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    reviewer: str
    naturalness: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    professional_tone: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    comment: str = ""
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mean(self) -> float:
        return round(
            sum(
                getattr(self, dimension)
                for dimension, _ in RUBRIC_DIMENSIONS
            )
            / len(RUBRIC_DIMENSIONS),
            3,
        )

    @property
    def blocking(self) -> bool:
        """Faithfulness or correctness below 3 blocks a release regardless of the mean."""
        return self.faithfulness < 3 or self.correctness < 3


class ComparisonReport(BaseModel):
    """Candidate vs current production (§Phase 17 release comparison)."""

    model_config = ConfigDict(extra="allow")

    candidate: str
    baseline: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    verdict: str = "undecided"

    def to_markdown(self) -> str:
        lines = [
            f"# Regression comparison: `{self.candidate}` vs `{self.baseline}`",
            "",
            f"Generated: {self.created_at.isoformat()}",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "|---|---:|---:|---:|",
        ]
        for metric, values in sorted(self.metrics.items()):
            baseline = values.get("baseline", 0.0)
            candidate = values.get("candidate", 0.0)
            delta = candidate - baseline
            lines.append(f"| {metric} | {baseline:.4f} | {candidate:.4f} | {delta:+.4f} |")
        lines.append("")
        if self.regressions:
            lines += ["## Regressions", "", *[f"- {r}" for r in self.regressions], ""]
        if self.improvements:
            lines += ["## Improvements", "", *[f"- {i}" for i in self.improvements], ""]
        lines += [f"**Verdict: {self.verdict.upper()}**", ""]
        return "\n".join(lines)
