"""The central preprocessing invariant: valid Khmer is never damaged."""

from __future__ import annotations

import pytest

from preprocessing.khmer_script import ZWSP, count_syllables
from preprocessing.tests.fixtures import (
    CODE_SWITCHED,
    NOISY_KHMER,
    REPAIRABLE,
    VALID_KHMER,
)
from preprocessing.unicode_normalization import (
    NormalizationConfig,
    normalize_for_hashing,
    normalize_khmer,
    normalize_text,
)


@pytest.mark.parametrize(("label", "text"), VALID_KHMER, ids=[label for label, _ in VALID_KHMER])
def test_valid_khmer_is_unchanged(label: str, text: str) -> None:
    report = normalize_khmer(text)
    assert report.text == text, f"{label}: normalisation altered valid Khmer\n{report.changes}"
    assert not report.changes, f"{label}: unexpected rules fired: {report.changes}"


@pytest.mark.parametrize("text", CODE_SWITCHED)
def test_code_switched_text_is_unchanged(text: str) -> None:
    assert normalize_text(text) == text


@pytest.mark.parametrize("text", NOISY_KHMER)
def test_noisy_input_keeps_its_khmer_content(text: str) -> None:
    """Noisy input may be repaired, but Khmer syllables must not be lost."""
    out = normalize_text(text)
    assert count_syllables(out) >= count_syllables(text) - 1
    assert out.strip()


@pytest.mark.parametrize(
    ("label", "source", "expected"), REPAIRABLE, ids=[r[0] for r in REPAIRABLE]
)
def test_repairable_sequences(label: str, source: str, expected: str) -> None:
    assert normalize_text(source) == expected, label


def test_report_counts_each_rule() -> None:
    report = normalize_khmer("ឤ កេា ស្្ថាន")
    assert report.changes["deprecated_replaced"] == 1
    assert report.changes["vowels_composed"] == 1
    assert report.changes["coeng_fixed"] == 1
    assert report.changed


def test_empty_and_whitespace() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   \n\n   ") == ""


def test_control_characters_removed_but_newlines_kept() -> None:
    out = normalize_text("ក\x00ខ\r\nគ")
    assert "\x00" not in out
    assert "\r" not in out
    assert "\n" in out
    assert "ក" in out and "ខ" in out and "គ" in out


def test_zwsp_policies() -> None:
    text = f"ខ្ញុំ{ZWSP}ចង់{ZWSP}{ZWSP}ដឹង"
    assert ZWSP not in normalize_text(text, NormalizationConfig(zwsp_policy="strip"))
    assert normalize_text(text, NormalizationConfig(zwsp_policy="keep")) == text
    collapsed = normalize_text(text, NormalizationConfig(zwsp_policy="collapse"))
    assert collapsed.count(ZWSP) == 2  # runs collapsed, boundaries preserved


def test_zwsp_adjacent_to_space_is_dropped() -> None:
    assert normalize_text(f"ខ្ញុំ {ZWSP}ចង់ដឹង") == "ខ្ញុំ ចង់ដឹង"


def test_khmer_digits_are_not_converted() -> None:
    """Converting ០-៩ to 0-9 would destroy information; only detection reports it."""
    assert normalize_text("តម្លៃ ១២៣") == "តម្លៃ ១២៣"


def test_space_before_khmer_terminator_is_removed() -> None:
    assert normalize_text("សូមអរគុណ ។") == "សូមអរគុណ។"


def test_space_after_khmer_terminator_is_added() -> None:
    assert normalize_text("សូមអរគុណ។យើងនឹងទាក់ទង") == "សូមអរគុណ។ យើងនឹងទាក់ទង"


def test_minimal_config_only_applies_nfc() -> None:
    text = "ខ្ញំុ"  # mis-ordered, but minimal config must not repair it
    assert normalize_text(text, NormalizationConfig.minimal()) == text


def test_normalize_for_hashing_is_form_insensitive() -> None:
    a = "តម្លៃ​ថ្មី។"
    b = "  តម្លៃ ថ្មី ។ "
    assert normalize_for_hashing(a) == normalize_for_hashing(b)


def test_normalize_for_hashing_is_case_insensitive_for_latin() -> None:
    assert normalize_for_hashing("Model QN-4500A") == normalize_for_hashing("model qn4500a")


def test_normalisation_is_idempotent() -> None:
    for _, text in VALID_KHMER:
        once = normalize_text(text)
        assert normalize_text(once) == once
    for _, source, _ in REPAIRABLE:
        once = normalize_text(source)
        assert normalize_text(once) == once
