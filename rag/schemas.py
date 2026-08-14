"""RAG data types: chunks, retrieval results, filters and index manifests."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Chunk",
    "RetrievedChunk",
    "RetrievalFilters",
    "RetrievalResult",
    "ConflictGroup",
    "IndexManifest",
    "RetrievalConfidence",
]


class RetrievalConfidence(StrEnum):
    HIGH = "high"        # answer from the context
    MEDIUM = "medium"    # answer, but hedge and cite
    LOW = "low"          # do not answer from context; express uncertainty
    NONE = "none"        # nothing retrieved


class Chunk(BaseModel):
    """An indexable unit of a company document."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    text: str
    ordinal: int = 0
    token_estimate: int = 0
    heading_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def product_id(self) -> str:
        return str(self.metadata.get("product_id", ""))

    @property
    def status(self) -> str:
        return str(self.metadata.get("status", ""))

    def contextual_text(self) -> str:
        """Text with its heading trail, so an isolated chunk still reads in context."""
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n" + self.text


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with the scores that put it there.

    The flat fields at the top mirror the retrieval-output contract in the
    specification (§Phase 10) so that an API consumer never has to reach into
    ``metadata``.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    text: str
    score: float
    product_id: str = ""
    version: str = ""
    effective_date: str = ""
    source: str = ""

    document_title: str = ""
    category: str = ""
    status: str = "active"
    language: str = "km"
    confidentiality: str = "public"
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float | None = None
    heading_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        *,
        score: float,
        dense_score: float = 0.0,
        lexical_score: float = 0.0,
        rerank_score: float | None = None,
    ) -> RetrievedChunk:
        meta = chunk.metadata
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=score,
            product_id=str(meta.get("product_id", "")),
            version=str(meta.get("version", "")),
            effective_date=str(meta.get("effective_date", "")),
            source=str(meta.get("source", "")),
            document_title=str(meta.get("document_title", "")),
            category=str(meta.get("category", "")),
            status=str(meta.get("status", "active")),
            language=str(meta.get("language", "km")),
            confidentiality=str(meta.get("confidentiality", "public")),
            dense_score=dense_score,
            lexical_score=lexical_score,
            rerank_score=rerank_score,
            heading_path=chunk.heading_path,
            metadata=meta,
        )


class RetrievalFilters(BaseModel):
    """Metadata filters applied before scoring (Phase 10 requirement)."""

    model_config = ConfigDict(extra="forbid")

    product_id: str | None = None
    service_id: str | None = None
    category: str | None = None
    version: str | None = None
    language: str | None = None
    status: list[str] = Field(default_factory=lambda: ["active"])
    max_confidentiality: str = "customer_shareable"
    effective_on: date | None = None
    include_expired: bool = False

    def matches(self, metadata: dict[str, Any], *, as_of: date | None = None) -> bool:
        """Evaluate the filter against a chunk's metadata."""
        if self.product_id and str(metadata.get("product_id", "")).upper() != self.product_id.upper():
            return False
        if self.service_id and str(metadata.get("service_id", "")).upper() != self.service_id.upper():
            return False
        if self.category and str(metadata.get("category", "")) != self.category:
            return False
        if self.version and str(metadata.get("version", "")) != self.version:
            return False
        if self.language and str(metadata.get("language", "")) not in (self.language, "mixed"):
            return False
        if self.status and str(metadata.get("status", "active")) not in self.status:
            return False

        allowed = _CONFIDENTIALITY_ORDER.get(self.max_confidentiality, 1)
        level = _CONFIDENTIALITY_ORDER.get(str(metadata.get("confidentiality", "public")), 3)
        if level > allowed:
            return False

        if not self.include_expired:
            expiry = str(metadata.get("expiration_date", "") or "")
            if expiry:
                today = self.effective_on or as_of or datetime.now(timezone.utc).date()
                try:
                    if date.fromisoformat(expiry) < today:
                        return False
                except ValueError:
                    pass
        return True


_CONFIDENTIALITY_ORDER = {
    "public": 0,
    "customer_shareable": 1,
    "internal": 2,
    "restricted": 3,
}


class ConflictGroup(BaseModel):
    """Two or more retrieved chunks that state different facts about one thing."""

    model_config = ConfigDict(extra="forbid")

    identity: str
    chunk_ids: list[str]
    document_ids: list[str]
    versions: list[str]
    differing_values: list[str] = Field(default_factory=list)
    newest_document_id: str = ""

    def describe(self) -> str:
        return (
            f"conflicting information for {self.identity!r} across versions "
            f"{', '.join(self.versions)}"
        )


class RetrievalResult(BaseModel):
    """Everything retrieval hands to answer generation."""

    model_config = ConfigDict(extra="forbid")

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    confidence: RetrievalConfidence = RetrievalConfidence.NONE
    confidence_score: float = 0.0
    conflicts: list[ConflictGroup] = Field(default_factory=list)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    strategy: str = "hybrid"
    index_version: str = "unknown"
    dropped_for_injection: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    def top_documents(self) -> list[dict[str, str]]:
        """De-duplicated citation list, preserving retrieval order."""
        seen: set[str] = set()
        out: list[dict[str, str]] = []
        for chunk in self.chunks:
            if chunk.document_id in seen:
                continue
            seen.add(chunk.document_id)
            out.append(
                {
                    "document_id": chunk.document_id,
                    "title": chunk.document_title,
                    "version": chunk.version,
                    "effective_date": chunk.effective_date,
                    "source": chunk.source,
                }
            )
        return out


class IndexManifest(BaseModel):
    """Written next to every built index; makes a build reproducible and rollback-able."""

    model_config = ConfigDict(extra="allow")

    index_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding_backend: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    chunk_size_tokens: int = 0
    chunk_overlap_tokens: int = 0
    documents: int = 0
    chunks: int = 0
    source_file: str = ""
    source_sha256: str = ""
    code_commit: str = "unknown"
    config_fingerprint: str = ""
    build_seconds: float = 0.0
    notes: str = ""
