"""Company knowledge ingestion: loaders, canonical schema, normalisation, validation."""

from company_data.normalize import NormalizeOptions, normalize_document, normalize_documents
from company_data.schema import (
    AccessLevel,
    CompanyDocument,
    Confidentiality,
    DocumentStatus,
    IngestionIssue,
    IngestionReport,
    ValidationStatus,
)
from company_data.validate import find_conflicts, ingest_directory, validate_documents

__all__ = [
    "AccessLevel",
    "CompanyDocument",
    "Confidentiality",
    "DocumentStatus",
    "IngestionIssue",
    "IngestionReport",
    "NormalizeOptions",
    "ValidationStatus",
    "find_conflicts",
    "ingest_directory",
    "normalize_document",
    "normalize_documents",
    "validate_documents",
]
