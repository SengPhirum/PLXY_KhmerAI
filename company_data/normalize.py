"""Turn :class:`LoadedDocument` values into canonical :class:`CompanyDocument`s.

Responsibilities:

1. Khmer Unicode normalisation of the body text (reusing the corpus rules, with
   ZWSP kept - company text is short and the word-break hints help chunking).
2. Metadata resolution: front matter > loader metadata > path conventions >
   defaults.  Path conventions matter in practice, because most teams organise
   documents as ``company/<category>/<product_id>/<file>``.
3. Injection neutralisation and PII redaction *before* the text can ever be
   embedded, with the outcome recorded on the document.
4. Status derivation from the effective/expiration dates.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from common.logging import get_logger
from company_data.loaders.base import LoadedDocument
from company_data.schema import (
    AccessLevel,
    CompanyDocument,
    Confidentiality,
    DocumentStatus,
    ValidationStatus,
)
from preprocessing.khmer_detection import TextLanguage, detect_language
from preprocessing.pii_filter import PiiFilter, PiiPolicy
from preprocessing.unicode_normalization import NormalizationConfig, normalize_khmer
from security.prompt_injection import sanitise_document, scan_for_injection

log = get_logger(__name__)

__all__ = ["NormalizeOptions", "normalize_document", "normalize_documents", "parse_date"]

# Company text keeps ZWSP: it is authored, not crawled, and the word breaks help
# the chunker split on real boundaries.
_COMPANY_NORMALIZATION = NormalizationConfig(zwsp_policy="collapse")

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y", "%Y%m%d")
_CATEGORY_ALIASES = {
    "warranty": {"warranty", "guarantee", "ធានា", "ការធានា"},
    "pricing": {"pricing", "price", "prices", "តម្លៃ"},
    "policy": {"policy", "policies", "terms", "គោលការណ៍"},
    "product": {"product", "products", "catalogue", "catalog", "ផលិតផល"},
    "service": {"service", "services", "សេវាកម្ម"},
    "troubleshooting": {"troubleshooting", "support", "faq", "ដោះស្រាយបញ្ហា"},
    "installation": {"installation", "setup", "install", "ការតំឡើង"},
    "returns": {"returns", "refund", "refunds", "ការប្តូរទំនិញ", "សំណង"},
}
_STATUS_ALIASES = {
    "active": DocumentStatus.ACTIVE,
    "current": DocumentStatus.ACTIVE,
    "published": DocumentStatus.ACTIVE,
    "live": DocumentStatus.ACTIVE,
    "draft": DocumentStatus.DRAFT,
    "pending": DocumentStatus.DRAFT,
    "expired": DocumentStatus.EXPIRED,
    "archived": DocumentStatus.ARCHIVED,
    "superseded": DocumentStatus.ARCHIVED,
}
_PRODUCT_ID_RE = re.compile(r"\b[A-Z]{1,6}-?\d{2,}[A-Z0-9-]*\b")


class NormalizeOptions:
    """Ingestion-time policy knobs."""

    def __init__(
        self,
        *,
        default_confidentiality: Confidentiality = Confidentiality.INTERNAL,
        default_access_level: AccessLevel = AccessLevel.AGENT,
        default_status: DocumentStatus = DocumentStatus.DRAFT,
        default_owner: str = "unassigned",
        redact_pii: bool = True,
        pii_policy: PiiPolicy | None = None,
        injection_block_threshold: float = 0.6,
        infer_from_path: bool = True,
        as_of: date | None = None,
    ) -> None:
        self.default_confidentiality = default_confidentiality
        self.default_access_level = default_access_level
        self.default_status = default_status
        self.default_owner = default_owner
        self.redact_pii = redact_pii
        self.pii_policy = pii_policy or PiiPolicy.for_company_public_docs()
        self.injection_block_threshold = injection_block_threshold
        self.infer_from_path = infer_from_path
        self.as_of = as_of or datetime.now(timezone.utc).date()


def parse_date(value: Any) -> date | None:
    """Parse the date spellings that actually appear in company files."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()  # noqa: DTZ007 - a date has no timezone
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        log.warning("company_data.unparsable_date", extra={"value": text[:40]})
        return None


def _canonical_category(raw: str | None) -> str:
    if not raw:
        return "general"
    lowered = str(raw).strip().lower()
    for canonical, aliases in _CATEGORY_ALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    return lowered.replace(" ", "_")


def _from_path(path: Path) -> dict[str, str]:
    """Infer category/product from ``.../<category>/<product_id>/<file>``."""
    inferred: dict[str, str] = {}
    parts = [p for p in path.parts if p not in ("/", ".", "..")]
    for part in parts[:-1]:
        category = _canonical_category(part)
        if category in _CATEGORY_ALIASES:
            inferred["category"] = category
        match = _PRODUCT_ID_RE.fullmatch(part.upper())
        if match:
            inferred["product_id"] = match.group(0)
    stem_match = _PRODUCT_ID_RE.search(path.stem.upper())
    if stem_match and "product_id" not in inferred:
        inferred["product_id"] = stem_match.group(0)
    return inferred


def _pick(meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = meta.get(key)
        if value not in (None, "", []):
            return value
    return None


def _derive_status(
    declared: Any, effective: date | None, expires: date | None, options: NormalizeOptions
) -> DocumentStatus:
    if declared:
        status = _STATUS_ALIASES.get(str(declared).strip().lower())
        if status is not None:
            if status is DocumentStatus.ACTIVE and expires and expires < options.as_of:
                return DocumentStatus.EXPIRED
            if status is DocumentStatus.ACTIVE and effective and effective > options.as_of:
                return DocumentStatus.DRAFT
            return status
    if expires and expires < options.as_of:
        return DocumentStatus.EXPIRED
    if effective and effective <= options.as_of:
        return DocumentStatus.ACTIVE
    return options.default_status


def _detect_language(text: str) -> str:
    language, _ = detect_language(text, khmer_present=0.10)
    if language is TextLanguage.KHMER:
        return "km"
    if language is TextLanguage.KHMER_ENGLISH:
        return "mixed"
    if language is TextLanguage.ENGLISH:
        return "en"
    return "km"


def normalize_document(
    loaded: LoadedDocument, options: NormalizeOptions | None = None
) -> CompanyDocument:
    """Convert one loaded file into a validated canonical record."""
    opts = options or NormalizeOptions()
    meta: dict[str, Any] = dict(loaded.metadata)

    # 1. Neutralise injection carriers before anything else touches the text.
    sanitised, transformations = sanitise_document(loaded.text)

    # 2. Khmer Unicode normalisation.
    normalisation = normalize_khmer(sanitised, _COMPANY_NORMALIZATION)
    text = normalisation.text

    # 3. PII.
    redactions: dict[str, int] = {}
    if opts.redact_pii:
        pii = PiiFilter(opts.pii_policy)
        redacted, redactions = pii.process_text(text)
        text = redacted if redacted is not None else text
        if redacted is None:
            redactions = redactions or {"blocked": 1}

    # 4. Injection score of the final text decides indexability.
    injection = scan_for_injection(text, block_threshold=opts.injection_block_threshold)

    if opts.infer_from_path:
        for key, value in _from_path(Path(loaded.source_path)).items():
            meta.setdefault(key, value)

    effective = parse_date(_pick(meta, "effective_date", "valid_from", "start_date"))
    expires = parse_date(_pick(meta, "expiration_date", "valid_to", "end_date", "expiry"))
    status = _derive_status(_pick(meta, "status", "state"), effective, expires, opts)

    confidentiality_raw = str(_pick(meta, "confidentiality") or opts.default_confidentiality)
    try:
        confidentiality = Confidentiality(confidentiality_raw)
    except ValueError:
        confidentiality = opts.default_confidentiality

    access_raw = str(_pick(meta, "access_level") or opts.default_access_level)
    try:
        access_level = AccessLevel(access_raw)
    except ValueError:
        access_level = opts.default_access_level

    validation_status = ValidationStatus.NEEDS_REVIEW
    if injection.blocked or redactions.get("blocked"):
        validation_status = ValidationStatus.QUARANTINED
    elif meta.get("likely_scanned") or meta.get("legacy_khmer_font_suspected"):
        validation_status = ValidationStatus.NEEDS_REVIEW

    title = (
        str(_pick(meta, "document_title", "title") or "")
        or loaded.title
        or Path(loaded.source_path).stem
    )

    document_metadata = {
        "loader": loaded.loader,
        "part": loaded.part,
        "normalisation": normalisation.changes,
        "pii_redactions": redactions,
        "sanitisation": transformations,
        "injection_score": round(injection.score, 3),
        "injection_matches": [m.name for m in injection.matches],
        "source_metadata": {
            k: v
            for k, v in meta.items()
            if k not in {"document_title", "title", "status", "state"}
        },
    }

    return CompanyDocument(
        document_title=title.strip(),
        product_id=(str(_pick(meta, "product_id", "sku", "model")) or None),
        product_name=(str(_pick(meta, "product_name")) if _pick(meta, "product_name") else None),
        service_id=(str(_pick(meta, "service_id")) if _pick(meta, "service_id") else None),
        service_name=(str(_pick(meta, "service_name")) if _pick(meta, "service_name") else None),
        category=_canonical_category(_pick(meta, "category")),
        subcategory=(str(_pick(meta, "subcategory")) if _pick(meta, "subcategory") else None),
        version=str(_pick(meta, "version", "rev", "revision") or "1.0"),
        effective_date=effective,
        expiration_date=expires,
        language=str(_pick(meta, "language") or _detect_language(text)),
        source_path=loaded.source_path,
        source_url=(str(_pick(meta, "source_url", "url")) if _pick(meta, "source_url", "url") else None),
        confidentiality=confidentiality,
        access_level=access_level,
        validation_status=validation_status,
        status=status,
        owner=str(_pick(meta, "owner", "department", "responsible") or opts.default_owner),
        text=text,
        metadata=document_metadata,
    )


def normalize_documents(
    loaded: list[LoadedDocument], options: NormalizeOptions | None = None
) -> list[CompanyDocument]:
    """Normalise a batch, skipping (and logging) any record that fails validation."""
    out: list[CompanyDocument] = []
    for item in loaded:
        if item.is_empty:
            log.warning("company_data.empty_document", extra={"source": item.source_path})
            continue
        try:
            out.append(normalize_document(item, options))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            log.error(
                "company_data.normalise_failed",
                extra={"source": item.source_path, "error": str(exc)},
            )
    return out
