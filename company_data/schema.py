"""Canonical company-document schema (Phase 4).

Everything the assistant is allowed to state as a company fact must exist as a
:class:`CompanyDocument`.  The schema carries the governance metadata required by
§34: an owner, an effective date, a version and a status, so that "where did
this claim come from?" is always answerable from the retrieval result alone.

Status semantics (used by retrieval filters):

``active``    current and answerable - the default retrieval filter
``draft``     not yet approved; never served to customers
``expired``   past ``expiration_date``; excluded unless explicitly requested
``archived``  superseded by a newer version; excluded by default
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.hashing import sha256_text, stable_id

__all__ = [
    "DocumentStatus",
    "Confidentiality",
    "AccessLevel",
    "ValidationStatus",
    "CompanyDocument",
    "IngestionIssue",
    "IngestionReport",
]


class DocumentStatus(StrEnum):
    ACTIVE = "active"
    DRAFT = "draft"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class Confidentiality(StrEnum):
    PUBLIC = "public"                 # safe to quote to a customer
    CUSTOMER_SHAREABLE = "customer_shareable"
    INTERNAL = "internal"             # may inform an answer but must not be quoted
    RESTRICTED = "restricted"         # never enters the retrieval index


class AccessLevel(StrEnum):
    ANONYMOUS = "anonymous"
    CUSTOMER = "customer"
    AGENT = "agent"
    ADMIN = "admin"


class ValidationStatus(StrEnum):
    VALID = "valid"
    NEEDS_REVIEW = "needs_review"
    QUARANTINED = "quarantined"       # injection or secret detected - not indexed
    INVALID = "invalid"


_RETRIEVABLE_CONFIDENTIALITY = frozenset(
    {Confidentiality.PUBLIC, Confidentiality.CUSTOMER_SHAREABLE}
)


class CompanyDocument(BaseModel):
    """One canonical company knowledge record."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    document_id: str = ""
    document_title: str
    product_id: str | None = None
    product_name: str | None = None
    service_id: str | None = None
    service_name: str | None = None
    category: str = "general"
    subcategory: str | None = None
    version: str = "1.0"
    effective_date: date | None = None
    expiration_date: date | None = None
    language: str = "km"
    source_path: str = ""
    source_url: str | None = None
    confidentiality: Confidentiality = Confidentiality.INTERNAL
    access_level: AccessLevel = AccessLevel.AGENT
    validation_status: ValidationStatus = ValidationStatus.NEEDS_REVIEW
    status: DocumentStatus = DocumentStatus.DRAFT
    owner: str = "unassigned"
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- validators ---------------------------------------------------------
    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("document text must not be empty")
        return value

    @field_validator("language")
    @classmethod
    def _known_language(cls, value: str) -> str:
        if value not in ("km", "en", "mixed"):
            raise ValueError(f"unsupported language {value!r}; expected km, en or mixed")
        return value

    @model_validator(mode="after")
    def _derive_fields(self) -> CompanyDocument:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", sha256_text(self.text))
        if not self.document_id:
            object.__setattr__(
                self,
                "document_id",
                stable_id(self.source_path or self.document_title, self.version, self.content_hash),
            )
        if self.expiration_date and self.effective_date and self.expiration_date < self.effective_date:
            raise ValueError(
                f"expiration_date {self.expiration_date} precedes effective_date {self.effective_date}"
            )
        return self

    # -- behaviour ----------------------------------------------------------
    def is_expired(self, as_of: date | None = None) -> bool:
        if self.expiration_date is None:
            return False
        return self.expiration_date < (as_of or datetime.now(timezone.utc).date())

    def is_effective(self, as_of: date | None = None) -> bool:
        """Currently in force: effective, not expired, and marked active."""
        today = as_of or datetime.now(timezone.utc).date()
        if self.status is not DocumentStatus.ACTIVE:
            return False
        if self.effective_date and self.effective_date > today:
            return False
        return not self.is_expired(today)

    def is_retrievable(self, as_of: date | None = None) -> bool:
        """Eligible for the customer-facing retrieval index."""
        return (
            self.is_effective(as_of)
            and self.validation_status is ValidationStatus.VALID
            and self.confidentiality in _RETRIEVABLE_CONFIDENTIALITY
        )

    def resolved_status(self, as_of: date | None = None) -> DocumentStatus:
        """Status with the expiry date applied - a stale ``active`` becomes ``expired``."""
        if self.status is DocumentStatus.ACTIVE and self.is_expired(as_of):
            return DocumentStatus.EXPIRED
        return self.status

    def identity_key(self) -> str:
        """Groups every version of the same logical document.

        Two records share an identity when they describe the same product/service
        and category - that is how ``validate.py`` finds conflicting versions.
        """
        return "|".join(
            [
                (self.product_id or self.service_id or "general").lower(),
                self.category.lower(),
                (self.subcategory or "").lower(),
                self.document_title.strip().lower(),
            ]
        )

    def to_index_metadata(self) -> dict[str, Any]:
        """Flat metadata attached to every chunk in the vector store."""
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "product_id": self.product_id or "",
            "product_name": self.product_name or "",
            "service_id": self.service_id or "",
            "service_name": self.service_name or "",
            "category": self.category,
            "subcategory": self.subcategory or "",
            "version": self.version,
            "effective_date": self.effective_date.isoformat() if self.effective_date else "",
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else "",
            "status": str(self.resolved_status()),
            "language": self.language,
            "confidentiality": str(self.confidentiality),
            "access_level": str(self.access_level),
            "owner": self.owner,
            "source": self.source_url or self.source_path,
            "content_hash": self.content_hash,
        }


class IngestionIssue(BaseModel):
    """One problem found while ingesting or validating a batch."""

    model_config = ConfigDict(extra="forbid")

    severity: str = "warning"        # info | warning | error
    code: str
    message: str
    source_path: str = ""
    document_id: str = ""

    def __str__(self) -> str:  # pragma: no cover - human output
        where = self.document_id or self.source_path or "-"
        return f"[{self.severity}] {self.code}: {self.message} ({where})"


class IngestionReport(BaseModel):
    """Required validation report for every ingestion batch (Phase 4)."""

    model_config = ConfigDict(extra="allow")

    files_processed: int = 0
    files_failed: int = 0
    documents_produced: int = 0
    duplicate_documents: int = 0
    missing_metadata: int = 0
    expired_documents: int = 0
    draft_documents: int = 0
    quarantined_documents: int = 0
    conflicting_versions: list[dict[str, Any]] = Field(default_factory=list)
    redactions: dict[str, int] = Field(default_factory=dict)
    issues: list[IngestionIssue] = Field(default_factory=list)
    by_category: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """A batch passes when nothing failed hard and no conflict is unresolved."""
        return (
            self.files_failed == 0
            and not self.conflicting_versions
            and not any(i.severity == "error" for i in self.issues)
        )

    def add(self, issue: IngestionIssue) -> None:
        self.issues.append(issue)

    def summary(self) -> str:
        return (
            f"files={self.files_processed} failed={self.files_failed} "
            f"docs={self.documents_produced} duplicates={self.duplicate_documents} "
            f"expired={self.expired_documents} quarantined={self.quarantined_documents} "
            f"conflicts={len(self.conflicting_versions)} ok={self.ok}"
        )
