"""Record schemas produced by the preprocessing pipeline."""

from __future__ import annotations

import itertools
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["INTENTS", "CleanRecord", "PipelineStats", "SFTMessage", "SFTMetadata", "SFTRecord"]

# §Phase 7 - the closed intent vocabulary the SFT dataset must cover.
INTENTS: tuple[str, ...] = (
    "greeting",
    "general_inquiry",
    "product_info",
    "service_info",
    "pricing",
    "specification",
    "availability",
    "warranty",
    "returns",
    "refund",
    "installation",
    "troubleshooting",
    "comparison",
    "recommendation",
    "complaint",
    "escalation",
    "unknown",
    "ambiguous",
    "follow_up",
    "multi_turn",
    "policy",
    "how_to",
    "account_related",
    "unsupported",
    "safety",
)


class CleanRecord(BaseModel):
    """One cleaned corpus record (Phase 3 output schema)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    text: str
    language: str = "km"
    khmer_ratio: float = 0.0
    quality_score: float = 0.0
    hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("khmer_ratio", "quality_score")
    @classmethod
    def _bounded(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class PipelineStats(BaseModel):
    """Audit statistics for one pipeline run."""

    model_config = ConfigDict(extra="allow")

    input_records: int = 0
    output_records: int = 0
    rejected_quality: int = 0
    rejected_language: int = 0
    rejected_pii: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    normalisation_changes: dict[str, int] = Field(default_factory=dict)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    bytes_in: int = 0
    bytes_out: int = 0
    seconds: float = 0.0
    config_fingerprint: str = ""

    @property
    def retention_rate(self) -> float:
        return self.output_records / self.input_records if self.input_records else 0.0


class SFTMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class SFTMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    language: str = "km"
    intent: str = "general_inquiry"
    source_type: str = ""
    source_id: str = ""
    quality_score: float = 1.0
    review_status: Literal["unreviewed", "auto_checked", "human_approved", "human_rejected"] = (
        "unreviewed"
    )
    synthetic: bool = False

    @field_validator("intent")
    @classmethod
    def _known_intent(cls, value: str) -> str:
        if value not in INTENTS:
            raise ValueError(f"unknown intent {value!r}; expected one of {', '.join(INTENTS)}")
        return value


class SFTRecord(BaseModel):
    """One supervised fine-tuning example (Phase 7 schema)."""

    model_config = ConfigDict(extra="forbid")

    messages: list[SFTMessage]
    metadata: SFTMetadata

    @field_validator("messages")
    @classmethod
    def _well_formed(cls, messages: list[SFTMessage]) -> list[SFTMessage]:
        if len(messages) < 2:
            raise ValueError("an SFT record needs at least a user and an assistant message")
        if messages[-1].role != "assistant":
            raise ValueError("the final message must come from the assistant")
        roles = [m.role for m in messages]
        if roles.count("system") > 1:
            raise ValueError("at most one system message is allowed")
        if "system" in roles and roles[0] != "system":
            raise ValueError("the system message must come first")
        body = [r for r in roles if r != "system"]
        for previous, current in itertools.pairwise(body):
            if previous == current:
                raise ValueError(f"consecutive {current!r} messages are not allowed")
        if any(not m.content.strip() for m in messages):
            raise ValueError("messages must not be empty")
        return messages

    @property
    def turns(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    @property
    def answer(self) -> str:
        return self.messages[-1].content
