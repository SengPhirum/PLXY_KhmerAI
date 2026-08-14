"""Request and response models for the public API."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "EscalationReason",
    "HealthResponse",
    "ModelsResponse",
    "ReadyResponse",
    "ReindexRequest",
    "ReindexResponse",
    "SearchRequest",
    "SearchResponse",
    "SourceRef",
    "StreamEvent",
]

_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_\-:.]{1,64}$")


class EscalationReason(StrEnum):
    NONE = "none"
    NO_INFORMATION = "no_information"
    CONFLICTING_SOURCES = "conflicting_sources"
    CUSTOMER_REQUESTED = "customer_requested"
    MONEY_OR_CLAIM = "money_or_claim"
    SAFETY = "safety"
    REPEATED_FAILURE = "repeated_failure"
    UNSUPPORTED = "unsupported"
    ACCOUNT_OR_PRIVACY = "account_or_privacy"


class SourceRef(BaseModel):
    """A company document that supported the answer."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str = ""
    version: str = ""
    effective_date: str = ""
    marker: str = ""
    source: str = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, description="the customer's message")
    conversation_id: str | None = Field(
        default=None, description="omit to start a new conversation"
    )
    language: Literal["km", "en", "auto"] = "km"
    product_id: str | None = Field(default=None, description="narrows retrieval to one product")
    category: str | None = None
    stream: bool = True
    model: str | None = Field(default=None, description="override the served model (admin only)")
    max_output_tokens: int | None = Field(default=None, ge=16, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    @field_validator("message")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @field_validator("conversation_id")
    @classmethod
    def _safe_conversation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _CONVERSATION_ID.match(value):
            raise ValueError(
                "conversation_id may contain only letters, digits, '_', '-', ':' and '.' "
                "(max 64 characters)"
            )
        return value

    @field_validator("product_id")
    @classmethod
    def _safe_product_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if len(cleaned) > 64 or not re.match(r"^[A-Za-z0-9_\-./ ]*$", cleaned):
            raise ValueError("product_id contains unsupported characters")
        return cleaned or None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    answer: str
    language: str = "km"
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: float = 0.0
    escalation_required: bool = False
    escalation_reason: EscalationReason = EscalationReason.NONE
    grounded: bool = True
    conflict_detected: bool = False
    intent: str = "general_inquiry"
    model: str = ""
    prompt_version: str = ""
    index_version: str = ""
    request_id: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StreamEvent(BaseModel):
    """One server-sent event on ``POST /v1/chat/stream``.

    Event order is guaranteed: ``start`` -> ``token``* -> ``sources``? -> ``done``,
    or ``start`` -> ``error``.  A client that only handles ``token`` and ``done``
    is still correct.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["start", "token", "sources", "done", "error"]
    conversation_id: str = ""
    request_id: str = ""
    content: str = ""
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: float = 0.0
    escalation_required: bool = False
    escalation_reason: EscalationReason = EscalationReason.NONE
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=50)
    product_id: str | None = None
    category: str | None = None
    include_expired: bool = False
    status: list[str] | None = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    confidence: str = "none"
    confidence_score: float = 0.0
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    index_version: str = ""
    dropped_for_injection: int = 0


class ReindexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: str | None = Field(
        default=None, description="canonical records JSONL; defaults to the configured path"
    )
    activate: bool = Field(default=False, description="activate if the regression gate passes")
    index_version: str | None = None
    dry_run: bool = False


class ReindexResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_version: str
    activated: bool
    regression_passed: bool
    documents: int = 0
    chunks: int = 0
    regression: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class ComponentHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    healthy: bool
    detail: str = ""
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "error"]
    uptime_seconds: float
    versions: dict[str, str] = Field(default_factory=dict)
    active_generations: int = 0
    queued_requests: int = 0


class ReadyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    components: list[ComponentHealth] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str = ""
    available: bool = False
    parameter_size: str = ""
    quantization: str = ""
    size_bytes: int = 0


class ModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str
    fallback: str = ""
    models: list[ModelInfo] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    generation_defaults: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str = ""
    request_id: str = ""
    retry_after_seconds: float | None = None
