"""Orthographic clustering, syllable counting and search tokenisation."""

from __future__ import annotations

import pytest

from preprocessing.khmer_script import (
    CharClass,
    classify_char,
    count_syllables,
    is_khmer_char,
    iter_clusters,
    segment_syllables,
    split_script_runs,
    strip_marks,
    tokenize_for_search,
)
from preprocessing.tests.fixtures import CODE_SWITCHED, VALID_KHMER


@pytest.mark.parametrize(("label", "text"), VALID_KHMER, ids=[label for label, _ in VALID_KHMER])
def test_clustering_is_lossless(label: str, text: str) -> None:
    """Concatenating the clusters must rebuild the original string exactly."""
    assert "".join(iter_clusters(text)) == text, label


@pytest.mark.parametrize("text", CODE_SWITCHED)
def test_segmentation_is_lossless_for_mixed_text(text: str) -> None:
    assert "".join(segment_syllables(text)) == text


def test_coeng_binds_the_following_consonant() -> None:
    assert list(iter_clusters("ខ្ញុំ")) == ["ខ្ញុំ"]
    assert list(iter_clusters("ស្ថាន")) == ["ស្ថា", "ន"]


def test_double_subscript_stays_in_one_cluster() -> None:
    # ស + ្ត + ្រ  (two stacked subscripts)
    assert list(iter_clusters("ស្ត្រី")) == ["ស្ត្រី"]


def test_syllable_count_ignores_latin_and_spaces() -> None:
    assert count_syllables("QN-4500A") == 0
    assert count_syllables("តម្លៃ") == 2  # ត + ម្លៃ


def test_classify_char() -> None:
    assert classify_char("ក") is CharClass.BASE
    assert classify_char("្") is CharClass.COENG
    assert classify_char("៉") is CharClass.SHIFTER
    assert classify_char("ា") is CharClass.VOWEL
    assert classify_char("ំ") is CharClass.SIGN
    assert classify_char("៥") is CharClass.KHMER_DIGIT
    assert classify_char("។") is CharClass.KHMER_PUNCT
    assert classify_char("A") is CharClass.LATIN
    assert classify_char("5") is CharClass.DIGIT
    assert classify_char(" ") is CharClass.WHITESPACE


def test_is_khmer_char() -> None:
    assert is_khmer_char("ក")
    assert is_khmer_char("៛")
    assert not is_khmer_char("A")
    assert not is_khmer_char("5")


def test_split_script_runs_round_trips() -> None:
    text = "តើ model QN-4500A តម្លៃប៉ុន្មាន?"
    runs = split_script_runs(text)
    assert "".join(run for _, run in runs) == text
    assert any(script == "khmer" for script, _ in runs)
    assert any(script == "other" for script, _ in runs)


def test_tokenizer_keeps_model_numbers_intact() -> None:
    tokens = tokenize_for_search("តើ QN-4500A តម្លៃប៉ុន្មាន?")
    assert "qn-4500a" in tokens


def test_tokenizer_emits_khmer_unigrams_and_bigrams() -> None:
    unigrams = tokenize_for_search("តម្លៃ", khmer_ngrams=(1,))
    both = tokenize_for_search("តម្លៃ", khmer_ngrams=(1, 2))
    assert unigrams == ["ត", "ម្លៃ"]
    assert both == ["ត", "ម្លៃ", "តម្លៃ"]


def test_tokenizer_lowercases_latin_only() -> None:
    tokens = tokenize_for_search("Warranty ការធានា")
    assert "warranty" in tokens
    assert "ការធានា" not in tokens  # Khmer is n-grammed, never case-folded


def test_strip_marks_removes_vowels_and_signs() -> None:
    assert strip_marks("ត្រជាក់") == strip_marks("ត្រជាក")


def test_empty_input() -> None:
    assert list(iter_clusters("")) == []
    assert segment_syllables("") == []
    assert tokenize_for_search("") == []
    assert count_syllables("") == 0
