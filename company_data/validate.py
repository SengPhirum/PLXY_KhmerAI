"""Batch validation and the required ingestion report (Phase 4 completion gate).

Run it::

    python -m company_data.validate \
        --input data/raw/company \
        --output data/interim/company_records.jsonl \
        --report data/manifests/company_validation.json

Checks performed
----------------
* files processed / failed, with the failure reason per file
* duplicate documents (identical content hash)
* missing governance metadata (owner, effective date, category, product)
* expired and draft documents
* quarantined documents (prompt injection or credentials found)
* **conflicting versions** - two documents that are simultaneously ``active``,
  describe the same product+category, and state different facts.  These are the
  dangerous ones: silently picking either is exactly what §2.3 forbids, so they
  are reported and, unless ``--allow-conflicts`` is passed, fail the batch.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from common.io import write_json, write_jsonl
from common.logging import get_logger
from company_data.loaders import SUPPORTED_EXTENSIONS, iter_documents, load_any
from company_data.normalize import NormalizeOptions, normalize_document
from company_data.schema import (
    CompanyDocument,
    DocumentStatus,
    IngestionIssue,
    IngestionReport,
    ValidationStatus,
)
from preprocessing.language_mixing import extract_protected_spans

log = get_logger(__name__)

__all__ = ["find_conflicts", "ingest_directory", "main", "validate_documents"]

REQUIRED_METADATA = ("owner", "category", "effective_date")


def _version_key(version: str) -> tuple[int, ...]:
    """Sortable version tuple; non-numeric parts sort as 0."""
    parts: list[int] = []
    for chunk in str(version).replace("-", ".").split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def _fact_fingerprint(document: CompanyDocument) -> frozenset[str]:
    """The verifiable facts a document asserts - prices, models, measurements.

    Two active documents about the same thing that assert *different* numbers
    are in conflict; two that assert the same numbers are merely duplicated.
    """
    return frozenset(
        span.normalised()
        for span in extract_protected_spans(document.text)
        if span.kind.value in ("currency", "measurement", "number", "model_number")
    )


def find_conflicts(
    documents: list[CompanyDocument], *, as_of: date | None = None
) -> list[dict[str, Any]]:
    """Detect simultaneously-active documents that assert different facts."""
    grouped: dict[str, list[CompanyDocument]] = defaultdict(list)
    for document in documents:
        if document.resolved_status(as_of) is DocumentStatus.ACTIVE:
            grouped[document.identity_key()].append(document)

    conflicts: list[dict[str, Any]] = []
    for key, group in grouped.items():
        if len(group) < 2:
            continue
        fingerprints = {d.document_id: _fact_fingerprint(d) for d in group}
        distinct = {frozenset(f) for f in fingerprints.values()}
        if len(distinct) <= 1:
            continue  # same facts, just duplicated - handled as a duplicate
        newest = max(group, key=lambda d: (_version_key(d.version), d.effective_date or date.min))
        conflicts.append(
            {
                "identity": key,
                "document_ids": sorted(d.document_id for d in group),
                "versions": sorted({d.version for d in group}),
                "sources": sorted(d.source_path for d in group),
                "suggested_winner": newest.document_id,
                "suggested_reason": (
                    f"highest version {newest.version}"
                    + (f" effective {newest.effective_date}" if newest.effective_date else "")
                ),
                "differing_facts": sorted(
                    {fact for facts in fingerprints.values() for fact in facts}
                )[:20],
            }
        )
    return conflicts


def validate_documents(
    documents: list[CompanyDocument],
    *,
    as_of: date | None = None,
    report: IngestionReport | None = None,
    promote_valid: bool = True,
) -> tuple[list[CompanyDocument], IngestionReport]:
    """Validate a normalised batch and produce the ingestion report.

    ``promote_valid=True`` flips ``needs_review`` to ``valid`` for documents that
    pass every automated check, which is what makes an unattended reindex
    possible.  Quarantined documents are never promoted.
    """
    result = report or IngestionReport()
    seen_hashes: dict[str, str] = {}
    accepted: list[CompanyDocument] = []

    for document in documents:
        result.documents_produced += 1
        status = document.resolved_status(as_of)
        result.by_status[str(status)] = result.by_status.get(str(status), 0) + 1
        result.by_category[document.category] = result.by_category.get(document.category, 0) + 1

        for kind, count in (document.metadata.get("pii_redactions") or {}).items():
            result.redactions[kind] = result.redactions.get(kind, 0) + int(count)

        if document.validation_status is ValidationStatus.QUARANTINED:
            result.quarantined_documents += 1
            result.add(
                IngestionIssue(
                    severity="error",
                    code="quarantined",
                    message=(
                        "prompt injection or credentials detected; document excluded from the index "
                        f"(matches: {', '.join(document.metadata.get('injection_matches', [])) or 'secret'})"
                    ),
                    source_path=document.source_path,
                    document_id=document.document_id,
                )
            )
            continue

        previous = seen_hashes.get(document.content_hash)
        if previous is not None:
            result.duplicate_documents += 1
            result.add(
                IngestionIssue(
                    severity="warning",
                    code="duplicate_document",
                    message=f"identical content to {previous}",
                    source_path=document.source_path,
                    document_id=document.document_id,
                )
            )
            continue
        seen_hashes[document.content_hash] = document.document_id

        missing = [
            field
            for field in REQUIRED_METADATA
            if not getattr(document, field, None)
            or getattr(document, field) in ("unassigned", "general")
        ]
        if not document.product_id and not document.service_id and document.category != "policy":
            missing.append("product_id_or_service_id")
        if missing:
            result.missing_metadata += 1
            result.add(
                IngestionIssue(
                    severity="warning",
                    code="missing_metadata",
                    message=f"missing or default: {', '.join(missing)}",
                    source_path=document.source_path,
                    document_id=document.document_id,
                )
            )

        if status is DocumentStatus.EXPIRED:
            result.expired_documents += 1
            result.add(
                IngestionIssue(
                    severity="info",
                    code="expired",
                    message=f"expired on {document.expiration_date}; excluded from default retrieval",
                    source_path=document.source_path,
                    document_id=document.document_id,
                )
            )
        elif status is DocumentStatus.DRAFT:
            result.draft_documents += 1
            result.add(
                IngestionIssue(
                    severity="info",
                    code="draft",
                    message="draft status; excluded from customer-facing retrieval",
                    source_path=document.source_path,
                    document_id=document.document_id,
                )
            )

        if (
            promote_valid
            and document.validation_status is ValidationStatus.NEEDS_REVIEW
            and not missing
        ):
            document = document.model_copy(update={"validation_status": ValidationStatus.VALID})

        accepted.append(document)

    # A silent "nothing is retrievable" is the most confusing possible outcome:
    # ingestion reports success and then the assistant answers "I don't know" to
    # everything.  Say so explicitly, with the reason.
    not_retrievable = [d for d in accepted if not d.is_retrievable(as_of)]
    if accepted and len(not_retrievable) == len(accepted):
        reasons = sorted({str(d.confidentiality) for d in not_retrievable})
        result.add(
            IngestionIssue(
                severity="error",
                code="nothing_retrievable",
                message=(
                    f"all {len(accepted)} document(s) are excluded from customer-facing "
                    f"retrieval (confidentiality: {', '.join(reasons)}). Documents default to "
                    "'internal' when they do not declare a confidentiality; add a "
                    "`confidentiality: public` field (front matter, CSV/XLSX column or HTML "
                    "meta tag) to the ones customers may see."
                ),
            )
        )
    elif not_retrievable:
        result.add(
            IngestionIssue(
                severity="info",
                code="not_retrievable",
                message=(
                    f"{len(not_retrievable)} of {len(accepted)} document(s) are excluded from "
                    "customer-facing retrieval (draft, expired, or not public)"
                ),
            )
        )

    result.conflicting_versions = find_conflicts(accepted, as_of=as_of)
    for conflict in result.conflicting_versions:
        result.add(
            IngestionIssue(
                severity="error",
                code="conflicting_versions",
                message=(
                    f"{len(conflict['document_ids'])} active documents for {conflict['identity']!r} "
                    f"assert different facts; suggested winner {conflict['suggested_winner']} "
                    f"({conflict['suggested_reason']})"
                ),
                document_id=conflict["suggested_winner"],
            )
        )
    return accepted, result


def ingest_directory(
    root: str | Path,
    *,
    options: NormalizeOptions | None = None,
    as_of: date | None = None,
    allowed_extensions: frozenset[str] | None = None,
) -> tuple[list[CompanyDocument], IngestionReport]:
    """Load, normalise and validate every supported file under ``root``."""
    report = IngestionReport()
    normalised: list[CompanyDocument] = []

    for path in iter_documents(root, allowed_extensions=allowed_extensions or SUPPORTED_EXTENSIONS):
        report.files_processed += 1
        try:
            for loaded in load_any(path):
                if loaded.is_empty:
                    report.add(
                        IngestionIssue(
                            severity="warning",
                            code="empty_document",
                            message="loader produced no text (scanned PDF? empty sheet?)",
                            source_path=str(path),
                        )
                    )
                    continue
                normalised.append(normalize_document(loaded, options))
        except Exception as exc:
            report.files_failed += 1
            report.add(
                IngestionIssue(
                    severity="error",
                    code="load_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    source_path=str(path),
                )
            )
            log.error("company_data.load_failed", extra={"path": str(path), "error": str(exc)})

    return validate_documents(normalised, as_of=as_of, report=report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m company_data.validate",
        description="Ingest, normalise and validate company documents",
    )
    parser.add_argument("--input", required=True, help="directory or file of company documents")
    parser.add_argument("--output", required=True, help="canonical records JSONL")
    parser.add_argument("--report", required=True, help="validation report JSON")
    parser.add_argument(
        "--allow-conflicts",
        action="store_true",
        help="exit 0 even when conflicting active versions are found (not for production)",
    )
    parser.add_argument(
        "--include-non-retrievable",
        action="store_true",
        help="write drafts/expired/internal records too (they are still excluded at query time)",
    )
    args = parser.parse_args(argv)

    documents, report = ingest_directory(args.input)

    to_write = (
        documents if args.include_non_retrievable else [d for d in documents if d.is_retrievable()]
    )
    write_jsonl(args.output, [d.model_dump(mode="json") for d in to_write])
    write_json(args.report, report.model_dump(mode="json"))

    print(report.summary())
    print(f"written: {len(to_write)} record(s) -> {args.output}")
    for issue in report.issues:
        if issue.severity == "error":
            print(f"  {issue}")

    if report.files_failed or report.quarantined_documents:
        return 1
    if report.conflicting_versions and not args.allow_conflicts:
        print(
            f"\nBLOCKED: {len(report.conflicting_versions)} conflicting active version(s). "
            "Resolve them (archive the superseded document) or rerun with --allow-conflicts."
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
