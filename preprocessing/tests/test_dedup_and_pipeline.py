"""Deduplication, leakage prevention and the full pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocessing.exact_dedup import ExactDeduplicator, dedupe_exact
from preprocessing.near_dedup import (
    LeakageChecker,
    NearDedupConfig,
    NearDeduplicator,
    khmer_shingle_tokens,
)
from preprocessing.pii_filter import PiiFilter, PiiPolicy, build_pre_upload_report
from preprocessing.pipeline import KhmerPipeline, PipelineConfig, run_pipeline
from preprocessing.tests.fixtures import SAMPLE_DOCUMENT_KM

BASE = "ការធានាលើផលិតផលអេឡិចត្រូនិកមានរយៈពេលពីរឆ្នាំ ចាប់ពីថ្ងៃទិញផលិតផល។"
NEAR = "ការធានាលើផលិតផលអេឡិចត្រូនិកមានរយៈពេលពីរឆ្នាំ ចាប់ពីថ្ងៃទិញផលិតផលនោះ។"
DIFFERENT = "សេវាកម្មដឹកជញ្ជូនទៅបណ្តាខេត្តត្រូវចំណាយពេលពីរទៅបីថ្ងៃធ្វើការ។"


# --- exact ------------------------------------------------------------------
def test_exact_duplicate_detected() -> None:
    d = ExactDeduplicator()
    assert d.is_new(BASE)
    assert not d.is_new(BASE)
    assert d.stats.exact_duplicates == 1


def test_normalised_duplicate_detected() -> None:
    d = ExactDeduplicator()
    assert d.is_new(BASE)
    assert not d.is_new(f"  {BASE}  ")
    assert d.stats.normalised_duplicates == 1


def test_distinct_text_survives() -> None:
    d = ExactDeduplicator()
    assert d.is_new(BASE)
    assert d.is_new(DIFFERENT)
    assert d.stats.kept == 2


def test_contains_does_not_mutate_stats() -> None:
    d = ExactDeduplicator()
    d.is_new(BASE)
    before = d.stats.seen
    assert d.contains(BASE)
    assert d.stats.seen == before


def test_dedupe_exact_keep_best_prefers_higher_score() -> None:
    records = [
        {"text": BASE, "source": "noisy", "quality_score": 0.4},
        {"text": f"{BASE} ", "source": "clean", "quality_score": 0.9},
    ]
    kept, stats = dedupe_exact(records, keep="best")
    assert len(kept) == 1
    assert kept[0]["source"] == "clean"
    assert stats.normalised_duplicates == 1


# --- near -------------------------------------------------------------------
def test_near_duplicate_detected() -> None:
    d = NearDeduplicator(NearDedupConfig(threshold=0.7))
    assert not d.add("a", BASE).is_duplicate
    assert d.add("b", NEAR).is_duplicate
    assert not d.add("c", DIFFERENT).is_duplicate
    assert d.stats.near_duplicates == 1


def test_check_does_not_insert() -> None:
    d = NearDeduplicator(NearDedupConfig(threshold=0.7))
    d.add("a", BASE)
    before = len(d)
    d.check(NEAR)
    assert len(d) == before


def test_shingle_tokens_are_khmer_aware() -> None:
    tokens = khmer_shingle_tokens("តម្លៃ QN-4500A")
    assert "qn4500a" in tokens
    assert any(t not in ("qn4500a",) for t in tokens)


def test_short_texts_fall_back_to_exact_matching() -> None:
    d = NearDeduplicator(NearDedupConfig(min_units=50))
    assert not d.add("a", "តម្លៃ").is_duplicate
    assert d.add("b", "តម្លៃ").is_duplicate
    assert d.stats.too_short_for_minhash == 2


# --- leakage ----------------------------------------------------------------
def test_leakage_checker_blocks_paraphrases_of_the_test_set() -> None:
    checker = LeakageChecker.from_texts([BASE], NearDedupConfig(threshold=0.7))
    assert checker.leaks(NEAR)
    assert not checker.leaks(DIFFERENT)
    report = checker.report()
    assert report["leaks_found"] == 1
    assert report["protected_records"] == 1


def test_leakage_checker_blocks_exact_copies() -> None:
    checker = LeakageChecker.from_texts([BASE])
    assert checker.leaks(BASE)


# --- PII --------------------------------------------------------------------
def test_pii_redaction_keeps_the_record() -> None:
    f = PiiFilter(PiiPolicy.for_public_corpus())
    text, counts = f.process_text("ទាក់ទង someone@example.com សម្រាប់ព័ត៌មាន")
    assert text is not None
    assert "someone@example.com" not in text
    assert counts["email"] == 1


def test_credentials_cause_the_record_to_be_dropped() -> None:
    f = PiiFilter(PiiPolicy.for_public_corpus())
    text, counts = f.process_text(
        'config: api_key="sk-abcdefghijklmnop1234567890"'  # pragma: allowlist secret
    )
    assert text is None
    assert f.report.records_dropped == 1
    assert counts


def test_khmer_digit_phone_numbers_are_caught() -> None:
    f = PiiFilter(PiiPolicy.for_public_corpus())
    text, counts = f.process_text("ទូរស័ព្ទ ០១២ ៣៤៥ ៦៧៨ សូមទាក់ទង")
    assert text is not None
    assert counts, "Khmer-digit phone number was not detected"


def test_company_policy_keeps_the_official_contact_details() -> None:
    f = PiiFilter(PiiPolicy.for_company_public_docs())
    text, counts = f.process_text("ទាក់ទងផ្នែកលក់ support@company.com")
    assert text is not None
    assert "support@company.com" in text
    assert "email" not in counts


def test_pre_upload_report_blocks_a_dirty_dataset(tmp_path: Path) -> None:
    dirty = tmp_path / "train.jsonl"
    dirty.write_text(
        json.dumps({"text": 'token = "ghp_" + "A"*40'}, ensure_ascii=False)
        + "\n"
        + json.dumps({"text": SAMPLE_DOCUMENT_KM}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    report = build_pre_upload_report([dirty])
    assert report["records_scanned"] == 2
    assert "files" in report and report["files"][0]["sha256"]

    clean = tmp_path / "clean.jsonl"
    clean.write_text(
        json.dumps({"text": SAMPLE_DOCUMENT_KM}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert build_pre_upload_report([clean])["approved"] is True


# --- pipeline ---------------------------------------------------------------
def test_pipeline_end_to_end(tmp_path: Path) -> None:
    rows = [
        {"id": "1", "source": "wiki", "text": SAMPLE_DOCUMENT_KM},
        {"id": "2", "source": "wiki", "text": SAMPLE_DOCUMENT_KM},  # exact dup
        {
            "id": "3",
            "source": "web",
            "text": "<p>" + SAMPLE_DOCUMENT_KM + "</p><footer>© 2026</footer>",
        },
        {
            "id": "4",
            "source": "web",
            "text": (
                "This page is written entirely in English and describes warranty terms, "
                "delivery options and refund policy without a single Khmer character."
            ),
        },
        {"id": "5", "source": "web", "text": DIFFERENT + " " + DIFFERENT + " " + DIFFERENT},
    ]
    src = tmp_path / "raw.jsonl"
    src.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    out = tmp_path / "clean.jsonl"
    report = tmp_path / "report.json"
    audit = run_pipeline(src, out, report_path=report, config=PipelineConfig(keep_rejected=True))

    assert audit["input_records"] == 5
    assert audit["output_records"] >= 1
    assert audit["exact_duplicates"] + audit["near_duplicates"] >= 1
    assert audit["rejected_language"] == 1
    assert report.exists()

    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written, "pipeline produced no records"
    for record in written:
        assert record["language"] == "km"
        assert 0.0 <= record["quality_score"] <= 1.0
        assert record["hash"]
        assert "©" not in record["text"]


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    rows = [
        {"id": str(i), "source": "s", "text": SAMPLE_DOCUMENT_KM + f" ចំណាំទី {i}។"} for i in range(5)
    ]
    src = tmp_path / "raw.jsonl"
    src.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    first = run_pipeline(src, tmp_path / "a.jsonl")
    second = run_pipeline(src, tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert first["config_fingerprint"] == second["config_fingerprint"]


def test_pipeline_records_normalisation_changes() -> None:
    pipeline = KhmerPipeline(PipelineConfig())
    pipeline.process_record({"id": "x", "source": "s", "text": "ឤ " + SAMPLE_DOCUMENT_KM})
    assert pipeline.stats.normalisation_changes.get("deprecated_replaced", 0) >= 1


def test_pipeline_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "nothing", tmp_path / "out.jsonl")
