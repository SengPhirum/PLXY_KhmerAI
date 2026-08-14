"""Review workflow for synthetic and human-written SFT samples (§33).

States: ``unreviewed`` -> ``auto_checked`` -> ``human_approved`` / ``human_rejected``.

Only ``auto_checked`` and ``human_approved`` samples enter the training split.
``human_rejected`` samples are kept - they are the natural source of DPO
*rejected* candidates, so throwing them away discards the most valuable signal
in the whole pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ReviewState", "ReviewPriority", "ReviewRecord", "review_priority", "TRAINABLE_STATES"]


class ReviewState(StrEnum):
    UNREVIEWED = "unreviewed"
    AUTO_CHECKED = "auto_checked"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"


TRAINABLE_STATES = frozenset({ReviewState.AUTO_CHECKED, ReviewState.HUMAN_APPROVED})


class ReviewPriority(StrEnum):
    CRITICAL = "critical"   # money, warranty, policy - a wrong answer is a liability
    HIGH = "high"           # technical troubleshooting - a wrong answer damages hardware
    MEDIUM = "medium"       # synthetic examples generally
    LOW = "low"             # sampled spot checks


# Intent -> review priority (§33 "Prioritize human review for").
_PRIORITY_BY_INTENT: dict[str, ReviewPriority] = {
    "warranty": ReviewPriority.CRITICAL,
    "refund": ReviewPriority.CRITICAL,
    "returns": ReviewPriority.CRITICAL,
    "pricing": ReviewPriority.CRITICAL,
    "policy": ReviewPriority.CRITICAL,
    "safety": ReviewPriority.CRITICAL,
    "troubleshooting": ReviewPriority.HIGH,
    "installation": ReviewPriority.HIGH,
    "escalation": ReviewPriority.HIGH,
    "account_related": ReviewPriority.HIGH,
}


def review_priority(intent: str, *, synthetic: bool, index: int = 0) -> ReviewPriority:
    """Decide how urgently a sample needs a native-speaker review."""
    if intent in _PRIORITY_BY_INTENT:
        return _PRIORITY_BY_INTENT[intent]
    if synthetic and index < 1000:
        # The first 1000 synthetic samples calibrate the generator.
        return ReviewPriority.MEDIUM
    return ReviewPriority.LOW


class ReviewRecord(BaseModel):
    """One review decision, attached to an SFT sample id."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    state: ReviewState = ReviewState.UNREVIEWED
    priority: ReviewPriority = ReviewPriority.MEDIUM
    reviewer: str = ""
    khmer_natural: bool | None = None
    facts_supported: bool | None = None
    intent_correct: bool | None = None
    uncertainty_handled: bool | None = None
    harmful_hallucination: bool | None = None
    duplicate: bool | None = None
    formatting_ok: bool | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    comment: str = ""
    reviewed_at: datetime | None = None

    @property
    def trainable(self) -> bool:
        return self.state in TRAINABLE_STATES

    def approve(self, reviewer: str, comment: str = "") -> ReviewRecord:
        return self.model_copy(
            update={
                "state": ReviewState.HUMAN_APPROVED,
                "reviewer": reviewer,
                "comment": comment,
                "reviewed_at": datetime.now(timezone.utc),
            }
        )

    def reject(self, reviewer: str, comment: str) -> ReviewRecord:
        return self.model_copy(
            update={
                "state": ReviewState.HUMAN_REJECTED,
                "reviewer": reviewer,
                "comment": comment,
                "reviewed_at": datetime.now(timezone.utc),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
