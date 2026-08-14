"""Company-document ingestion, governance and validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from company_data.loaders import (
    SUPPORTED_EXTENSIONS,
    LoaderError,
    UnsupportedFormatError,
    load_any,
    load_csv,
    load_html,
    load_text,
)
from company_data.loaders.base import FileTooLargeError, check_file
from company_data.loaders.csv_loader import canonical_column
from company_data.loaders.registry import iter_documents
from company_data.normalize import NormalizeOptions, normalize_document, parse_date
from company_data.schema import (
    CompanyDocument,
    Confidentiality,
    DocumentStatus,
    ValidationStatus,
)
from company_data.validate import find_conflicts, ingest_directory, validate_documents


# --- schema -----------------------------------------------------------------
def _doc(**overrides: object) -> CompanyDocument:
    base: dict[str, object] = {
        "document_title": "គោលការណ៍ធានា",
        "text": "ការធានារយៈពេល ២៤ ខែ។",
        "category": "warranty",
        "product_id": "QN-4500A",
        "owner": "after-sales",
        "status": DocumentStatus.ACTIVE,
        "confidentiality": Confidentiality.PUBLIC,
        "validation_status": ValidationStatus.VALID,
        "effective_date": date(2026, 1, 1),
    }
    base.update(overrides)
    return CompanyDocument(**base)  # type: ignore[arg-type]


def test_document_id_and_hash_are_derived_and_stable() -> None:
    a, b = _doc(), _doc()
    assert a.document_id and a.document_id == b.document_id
    assert a.content_hash == b.content_hash
    assert _doc(text="ការធានារយៈពេល ១២ ខែ។").document_id != a.document_id


def test_expiry_logic() -> None:
    doc = _doc(effective_date=date(2025, 4, 1), expiration_date=date(2025, 4, 30))
    assert doc.is_expired(date(2026, 8, 14))
    assert not doc.is_expired(date(2025, 4, 1))
    assert doc.resolved_status(date(2026, 8, 14)) is DocumentStatus.EXPIRED
    assert not doc.is_retrievable(date(2026, 8, 14))


def test_future_effective_date_is_not_effective_yet() -> None:
    doc = _doc(effective_date=date(2027, 1, 1))
    assert not doc.is_effective(date(2026, 8, 14))


def test_internal_documents_are_not_retrievable() -> None:
    assert not _doc(confidentiality=Confidentiality.INTERNAL).is_retrievable()
    assert not _doc(confidentiality=Confidentiality.RESTRICTED).is_retrievable()
    assert _doc(confidentiality=Confidentiality.PUBLIC).is_retrievable()


def test_quarantined_documents_are_not_retrievable() -> None:
    assert not _doc(validation_status=ValidationStatus.QUARANTINED).is_retrievable()


def test_expiration_before_effective_is_rejected() -> None:
    with pytest.raises(ValueError, match="precedes"):
        _doc(effective_date=date(2026, 5, 1), expiration_date=date(2026, 1, 1))


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _doc(text="   ")


def test_index_metadata_is_flat_and_complete() -> None:
    meta = _doc().to_index_metadata()
    assert meta["product_id"] == "QN-4500A"
    assert meta["status"] == "active"
    assert all(isinstance(v, str) for v in meta.values())


# --- loaders ----------------------------------------------------------------
def test_text_loader_reads_front_matter(company_dir: Path) -> None:
    docs = load_text(company_dir / "warranty" / "QN-4500A" / "warranty_v2.md")
    assert len(docs) == 1
    assert docs[0].metadata["product_id"] == "QN-4500A"
    assert docs[0].metadata["version"] == "2.0"
    assert "២៤ ខែ" in docs[0].text
    assert "---" not in docs[0].text


def test_csv_loader_emits_one_document_per_row(company_dir: Path) -> None:
    docs = load_csv(company_dir / "pricing" / "price_list.csv")
    assert len(docs) == 3
    assert {d.metadata["product_id"] for d in docs} == {"QN-4500A", "RF-22B", "WM-900"}
    assert "price: 520" in docs[0].text
    assert docs[0].part == "row1"


def test_csv_column_mapping_handles_aliases() -> None:
    assert canonical_column("SKU") == "product_id"
    assert canonical_column("Unit Price") == "price"
    assert canonical_column("តម្លៃ") == "price"
    assert canonical_column("random_column") is None


def test_html_loader_extracts_meta_and_drops_chrome(company_dir: Path) -> None:
    docs = load_html(company_dir / "policy" / "repair_service.html")
    assert docs[0].metadata["html_meta_service_id"] == "REPAIR"
    assert docs[0].title == "សេវាកម្មជួសជុល"
    assert "ទំព័រដើម | សេវាកម្ម" not in docs[0].text
    assert "© 2026" not in docs[0].text
    assert "សេវាកម្មជួសជុលមានផ្តល់ជូន" in docs[0].text


def test_unsupported_extension_is_refused(company_dir: Path) -> None:
    with pytest.raises(UnsupportedFormatError):
        load_any(company_dir / "notes.exe")


def test_directory_walk_only_yields_allowed_types(company_dir: Path) -> None:
    found = list(iter_documents(company_dir))
    assert found
    assert all(p.suffix.lower() in SUPPORTED_EXTENSIONS for p in found)
    assert not any(p.name == "notes.exe" for p in found)


def test_check_file_rejects_symlinks_and_missing(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("ការធានា", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    with pytest.raises(LoaderError, match="symlink"):
        check_file(link)
    with pytest.raises(LoaderError, match="not found"):
        check_file(tmp_path / "missing.txt")
    with pytest.raises(FileTooLargeError):
        check_file(real, max_bytes=1)


def test_path_traversal_is_blocked() -> None:
    from common.paths import resolve_under_root

    root = Path("/tmp/khmerai-root")
    with pytest.raises(ValueError, match="escapes"):
        resolve_under_root("../../etc/passwd", root)


# --- normalisation ----------------------------------------------------------
def test_parse_date_accepts_common_spellings() -> None:
    assert parse_date("2026-01-01") == date(2026, 1, 1)
    assert parse_date("01/02/2026") == date(2026, 2, 1)
    assert parse_date(date(2026, 3, 4)) == date(2026, 3, 4)
    assert parse_date("") is None
    assert parse_date("not a date") is None


def test_normalise_derives_status_from_dates(company_dir: Path) -> None:
    loaded = load_text(company_dir / "pricing" / "promo_new_year.md")[0]
    document = normalize_document(loaded, NormalizeOptions(as_of=date(2026, 8, 14)))
    assert document.status is DocumentStatus.EXPIRED
    assert not document.is_retrievable(date(2026, 8, 14))


def test_normalise_infers_metadata_from_path(tmp_path: Path) -> None:
    path = tmp_path / "warranty" / "QN-4500A" / "note.md"
    path.parent.mkdir(parents=True)
    path.write_text("ការធានារយៈពេល ២៤ ខែ សម្រាប់ផលិតផលនេះ។", encoding="utf-8")
    document = normalize_document(load_text(path)[0])
    assert document.category == "warranty"
    assert document.product_id == "QN-4500A"


def test_injected_document_is_quarantined(company_dir: Path) -> None:
    loaded = load_text(company_dir / "product" / "rf22b_injected.md")[0]
    document = normalize_document(loaded)
    assert document.validation_status is ValidationStatus.QUARANTINED
    assert document.metadata["injection_matches"]
    assert not document.is_retrievable()


def test_khmer_text_is_normalised_not_mangled(company_dir: Path) -> None:
    loaded = load_text(company_dir / "warranty" / "QN-4500A" / "warranty_v2.md")[0]
    document = normalize_document(loaded)
    assert "ការធានារយៈពេល ២៤ ខែ" in document.text
    assert document.language in ("km", "mixed")


# --- validation -------------------------------------------------------------
def test_ingest_directory_produces_a_full_report(company_dir: Path) -> None:
    documents, report = ingest_directory(company_dir, as_of=date(2026, 8, 14))
    assert report.files_processed == 6  # .exe excluded
    assert report.files_failed == 0
    assert report.quarantined_documents == 1
    assert report.expired_documents == 1
    assert report.draft_documents == 1
    assert report.by_category
    assert not any(d.validation_status is ValidationStatus.QUARANTINED for d in documents)


def test_conflicting_active_versions_are_detected(conflicting_dir: Path) -> None:
    _documents, report = ingest_directory(conflicting_dir, as_of=date(2026, 8, 14))
    assert report.conflicting_versions, "two active warranty versions were not flagged"
    conflict = report.conflicting_versions[0]
    assert len(conflict["document_ids"]) == 2
    assert conflict["suggested_winner"]
    assert "2.0" in conflict["versions"]
    assert not report.ok


def test_identical_facts_are_a_duplicate_not_a_conflict() -> None:
    a = _doc(version="1.0", source_path="a.md")
    b = _doc(version="2.0", source_path="b.md")
    assert find_conflicts([a, b]) == []


def test_duplicate_content_is_reported_once() -> None:
    documents, report = validate_documents([_doc(source_path="a.md"), _doc(source_path="a.md")])
    assert len(documents) == 1
    assert report.duplicate_documents == 1


def test_missing_metadata_is_reported() -> None:
    _, report = validate_documents([_doc(owner="unassigned", effective_date=None)])
    assert report.missing_metadata == 1
    assert any(i.code == "missing_metadata" for i in report.issues)


def test_valid_documents_are_promoted() -> None:
    documents, _ = validate_documents([_doc(validation_status=ValidationStatus.NEEDS_REVIEW)])
    assert documents[0].validation_status is ValidationStatus.VALID


def test_retrievable_subset_excludes_draft_expired_and_internal(company_dir: Path) -> None:
    documents, _ = ingest_directory(company_dir, as_of=date(2026, 8, 14))
    retrievable = [d for d in documents if d.is_retrievable(date(2026, 8, 14))]
    assert retrievable, "nothing survived the retrievability filter"
    for document in retrievable:
        assert document.status is DocumentStatus.ACTIVE
        assert document.confidentiality in (
            Confidentiality.PUBLIC,
            Confidentiality.CUSTOMER_SHAREABLE,
        )
