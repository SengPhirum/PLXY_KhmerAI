"""Language detection, code-switch analysis and quality scoring."""

from __future__ import annotations

import pytest

from preprocessing.html_cleanup import clean_web_text, remove_boilerplate, strip_html
from preprocessing.khmer_detection import TextLanguage, detect_language, profile_text
from preprocessing.language_mixing import (
    SpanKind,
    analyse_code_switching,
    extract_protected_spans,
    verify_protected_spans,
)
from preprocessing.quality_filter import (
    QualityThresholds,
    assess_quality,
    repetition_ratio,
    shannon_entropy,
    summarise_rejections,
)
from preprocessing.tests.fixtures import (
    CODE_SWITCHED,
    HTML_SAMPLE,
    LOW_QUALITY,
    SAMPLE_DOCUMENT_KM,
    VALID_KHMER,
)


# --- detection --------------------------------------------------------------
@pytest.mark.parametrize(("label", "text"), VALID_KHMER, ids=[label for label, _ in VALID_KHMER])
def test_valid_khmer_detected_as_khmer(label: str, text: str) -> None:
    language, profile = detect_language(text)
    assert language in (TextLanguage.KHMER, TextLanguage.KHMER_ENGLISH), label
    assert profile.khmer_ratio > 0.5


@pytest.mark.parametrize("text", CODE_SWITCHED)
def test_code_switched_detected(text: str) -> None:
    language, _ = detect_language(text)
    assert language is TextLanguage.KHMER_ENGLISH


def test_english_detected() -> None:
    language, _ = detect_language("This is a plain English sentence about warranty terms.")
    assert language is TextLanguage.ENGLISH


def test_empty_is_invalid() -> None:
    assert detect_language("")[0] is TextLanguage.INVALID
    assert detect_language("   ")[0] is TextLanguage.INVALID


def test_short_khmer_is_still_khmer() -> None:
    """A two-syllable customer message must not be rejected as too short."""
    language, _ = detect_language("តម្លៃ?")
    assert language is TextLanguage.KHMER


def test_profile_ratios_sum_sensibly() -> None:
    profile = profile_text("តម្លៃ QN-4500A ១២៣ 456")
    assert profile.khmer_chars > 0
    assert profile.latin_chars > 0
    assert profile.khmer_digit_chars == 3  # ១២៣
    assert profile.digit_chars == 7  # 4500 + 456
    assert 0.0 <= profile.khmer_ratio <= 1.0


# --- code switching ---------------------------------------------------------
@pytest.mark.parametrize("text", CODE_SWITCHED)
def test_analysis_flags_code_switching(text: str) -> None:
    analysis = analyse_code_switching(text)
    assert analysis.is_code_switched
    assert analysis.switch_points >= 1


def test_pure_khmer_is_not_code_switched() -> None:
    assert not analyse_code_switching("សូមអរគុណច្រើន។").is_code_switched


# --- protected spans --------------------------------------------------------
def test_extract_model_number_and_url() -> None:
    spans = extract_protected_spans("សូមមើល QN-4500A នៅ https://example.com/a?b=1")
    kinds = {s.kind for s in spans}
    assert SpanKind.MODEL_NUMBER in kinds
    assert SpanKind.URL in kinds


def test_url_is_not_fragmented() -> None:
    spans = extract_protected_spans("https://example.com/products/qn4500a")
    urls = [s for s in spans if s.kind is SpanKind.URL]
    assert len(urls) == 1
    assert urls[0].text == "https://example.com/products/qn4500a"


def test_currency_span() -> None:
    spans = extract_protected_spans("តម្លៃ $120 ឬ 480000៛")
    assert any(s.kind is SpanKind.CURRENCY for s in spans)


def test_verify_protected_spans_accepts_faithful_answer() -> None:
    source = "ម៉ូដែល QN-4500A មានតម្លៃ $120"
    ok, missing = verify_protected_spans(source, "ម៉ូដែល QN-4500A មានតម្លៃ $120 ។")
    assert ok and not missing


def test_verify_protected_spans_rejects_transliterated_model_number() -> None:
    source = "ម៉ូដែល QN-4500A"
    ok, missing = verify_protected_spans(source, "ម៉ូដែល QN ៤៥០០ អា", require_all=True)
    assert not ok
    assert any(s.text == "QN-4500A" for s in missing)


def test_verify_allows_answer_that_omits_a_fact() -> None:
    source = "ម៉ូដែល QN-4500A តម្លៃ $120"
    ok, _ = verify_protected_spans(source, "សូមទាក់ទងផ្នែកលក់សម្រាប់ព័ត៌មានបន្ថែម។")
    assert ok


# --- html cleanup -----------------------------------------------------------
def test_strip_html_drops_script_and_style() -> None:
    text = strip_html(HTML_SAMPLE)
    assert "var a=1" not in text
    assert "display:none" not in text
    assert "ការធានារយៈពេល" in text


def test_clean_web_text_removes_nav_cookie_and_footer() -> None:
    text = clean_web_text(HTML_SAMPLE)
    assert "cookies" not in text.lower()
    assert "All rights reserved" not in text
    assert "ទំព័រដើម | ផលិតផល" not in text
    assert "ការធានារយៈពេល ២ ឆ្នាំ" in text


def test_remove_boilerplate_drops_repeated_short_lines() -> None:
    text = "\n".join(["អានបន្ថែម"] * 5 + ["ខ្លឹមសារពិតប្រាកដដែលមានប្រយោជន៍សម្រាប់អតិថិជន"])
    assert "ខ្លឹមសារពិតប្រាកដ" in remove_boilerplate(text)


def test_strip_html_passthrough_for_plain_text() -> None:
    assert strip_html(SAMPLE_DOCUMENT_KM) == SAMPLE_DOCUMENT_KM


# --- quality ----------------------------------------------------------------
def test_good_document_accepted() -> None:
    assessment = assess_quality(SAMPLE_DOCUMENT_KM)
    assert assessment.accepted
    assert assessment.score > 0.6
    assert assessment.rejection_reason is None


@pytest.mark.parametrize(("label", "text"), LOW_QUALITY, ids=[label for label, _ in LOW_QUALITY])
def test_low_quality_rejected_with_a_reason(label: str, text: str) -> None:
    assessment = assess_quality(text)
    assert not assessment.accepted, label
    assert assessment.rejection_reason, label


def test_specific_rejection_reasons() -> None:
    assert assess_quality("").rejection_reason == "empty"
    assert assess_quality("ា" * 120).rejection_reason in ("character_run", "low_entropy")
    assert assess_quality("ា" * 20).rejection_reason == "too_short"
    reason = assess_quality(
        "This page is entirely in English and contains no Khmer text whatsoever."
    ).rejection_reason
    assert reason == "language_english"


def test_sft_profile_accepts_short_support_turns() -> None:
    text = "តើម៉ូដែលនេះមាន warranty ប៉ុន្មានឆ្នាំ?"
    assert not assess_quality(text).accepted  # too short for a corpus
    assert assess_quality(text, QualityThresholds.for_sft()).accepted


def test_company_profile_tolerates_digit_heavy_tables() -> None:
    text = "QN-4500A | 120 USD | 24 months | 350 L | 2026-01-01"
    assert assess_quality(text, QualityThresholds.for_company_documents()).accepted


def test_entropy_and_repetition_helpers() -> None:
    assert shannon_entropy("aaaa") == pytest.approx(0.0)
    assert shannon_entropy("abcd") == pytest.approx(2.0)
    assert repetition_ratio("ក" * 40) > 0.5
    assert repetition_ratio(SAMPLE_DOCUMENT_KM) < 0.3


def test_summarise_rejections() -> None:
    assessments = [assess_quality(text) for _, text in LOW_QUALITY]
    assessments.append(assess_quality(SAMPLE_DOCUMENT_KM))
    summary = summarise_rejections(assessments)
    assert summary["total"] == len(assessments)
    assert summary["accepted"] == 1
    assert summary["rejection_reasons"]
