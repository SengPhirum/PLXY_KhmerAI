"""A model of the Khmer script: character classes, orthographic syllables, tokens.

Khmer is written without spaces between words, so every English-centric
assumption ("split on whitespace", "a character is a token", "length in
characters ~ length in words") is wrong.  This module gives the rest of the
platform three primitives that *are* correct for Khmer:

``iter_clusters``
    Split text into orthographic syllable clusters.  A cluster is
    ``base (COENG base)* shifter? vowel* sign*`` - the unit a Khmer reader
    perceives as one written syllable, and the unit that must never be split by
    normalisation, truncation or chunking.

``segment_syllables``
    The cluster list for Khmer runs, keeping non-Khmer runs (Latin product
    names, model numbers, URLs) intact.

``tokenize_for_search``
    Search tokens for BM25.  Khmer has no reliable free word segmenter that is
    worth a runtime dependency here, so Khmer runs are indexed as syllable
    unigrams *and* bigrams - the standard n-gram fallback for space-less
    scripts, which recovers most word-level matching without a lexicon - while
    Latin/numeric runs are indexed as whole words so that ``QN-4500A`` stays one
    token.

References for the code-point ranges: The Unicode Standard, Khmer block
(U+1780..U+17FF) and Khmer Symbols block (U+19E0..U+19FF).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from enum import IntEnum
from typing import Final

__all__ = [
    "CharClass",
    "classify_char",
    "is_khmer_char",
    "iter_clusters",
    "segment_syllables",
    "count_syllables",
    "tokenize_for_search",
    "split_script_runs",
    "KHMER_CONSONANTS",
    "KHMER_PUNCTUATION",
    "COENG",
    "ZERO_WIDTH",
]

# --- code points ------------------------------------------------------------
COENG: Final = "្"
ZWSP: Final = "​"
ZWNJ: Final = "‌"
ZWJ: Final = "‍"
ZERO_WIDTH: Final = frozenset({ZWSP, ZWNJ, ZWJ, "﻿", "⁠"})

KHMER_CONSONANTS: Final = frozenset(chr(c) for c in range(0x1780, 0x17A3))  # ក..អ
KHMER_INDEPENDENT_VOWELS: Final = frozenset(chr(c) for c in range(0x17A3, 0x17B4))
KHMER_DEPENDENT_VOWELS: Final = frozenset(chr(c) for c in range(0x17B6, 0x17C6))
KHMER_REGISTER_SHIFTERS: Final = frozenset({"៉", "៊"})  # MUUSIKATOAN, TRIISAP
KHMER_SIGNS: Final = frozenset(
    chr(c) for c in list(range(0x17C6, 0x17D2)) + [0x17D3, 0x17DD]
) - KHMER_REGISTER_SHIFTERS
KHMER_DIGITS: Final = frozenset(chr(c) for c in range(0x17E0, 0x17EA))
KHMER_LEK_ATTAK: Final = frozenset(chr(c) for c in range(0x17F0, 0x17FA))
# U+17D4..U+17DC: khan, bariyoosan, camnuc pii kuuh, lek too, beyyal, phnaek
# muan, koomuut, riel sign, avakrahasanya.  Plus the Khmer Symbols block
# (lunar-date signs) at U+19E0..U+19FF.
KHMER_PUNCTUATION: Final = frozenset(chr(c) for c in range(0x17D4, 0x17DD)) | frozenset(
    chr(c) for c in range(0x19E0, 0x1A00)
)
# Deprecated / discouraged code points handled by unicode_normalization.py.
KHMER_DEPRECATED: Final = frozenset({"ឣ", "ឤ", "឴", "឵", "៓"})


class CharClass(IntEnum):
    """Rank inside an orthographic cluster.  Also the canonical ordering key."""

    BASE = 0            # consonant or independent vowel
    COENG = 1           # U+17D2 and the subscript consonant that follows it
    SHIFTER = 2         # U+17C9 / U+17CA
    VOWEL = 3           # dependent vowel U+17B6..U+17C5
    SIGN = 4            # nikahit, reahmuk, robat, toandakhiat, ...
    KHMER_DIGIT = 5
    KHMER_PUNCT = 6
    LATIN = 7
    DIGIT = 8
    WHITESPACE = 9
    ZERO_WIDTH = 10
    OTHER = 11


def classify_char(ch: str) -> CharClass:
    """Classify a single character for clustering and canonical ordering."""
    if ch in KHMER_CONSONANTS or ch in KHMER_INDEPENDENT_VOWELS:
        return CharClass.BASE
    if ch == COENG:
        return CharClass.COENG
    if ch in KHMER_REGISTER_SHIFTERS:
        return CharClass.SHIFTER
    if ch in KHMER_DEPENDENT_VOWELS:
        return CharClass.VOWEL
    if ch in KHMER_SIGNS:
        return CharClass.SIGN
    if ch in KHMER_DIGITS or ch in KHMER_LEK_ATTAK:
        return CharClass.KHMER_DIGIT
    if ch in KHMER_PUNCTUATION:
        return CharClass.KHMER_PUNCT
    if ch in ZERO_WIDTH:
        return CharClass.ZERO_WIDTH
    if ch.isspace():
        return CharClass.WHITESPACE
    if ch.isdigit():
        return CharClass.DIGIT
    if ch.isalpha():
        return CharClass.LATIN
    return CharClass.OTHER


def is_khmer_char(ch: str) -> bool:
    """True for any code point in the Khmer or Khmer Symbols blocks."""
    cp = ord(ch)
    return 0x1780 <= cp <= 0x17FF or 0x19E0 <= cp <= 0x19FF


# A cluster: an optional leading base, then any number of attached parts.
# Written as an explicit scanner rather than one regex because the COENG rule
# ("U+17D2 binds the *next* character into this cluster") is not expressible as
# a plain character class.
def iter_clusters(text: str) -> Iterator[str]:
    """Yield orthographic clusters for Khmer runs and single chars elsewhere.

    >>> list(iter_clusters("ខ្ញុំ"))
    ['ខ្ញុំ']
    >>> list(iter_clusters("ស្ថាន"))
    ['ស្ថា', 'ន']
    """
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        cls = classify_char(ch)
        if cls is not CharClass.BASE:
            # Orphan mark or non-Khmer character: emit alone so nothing is lost.
            yield ch
            i += 1
            continue

        start = i
        i += 1
        while i < n:
            ch = text[i]
            cls = classify_char(ch)
            if cls is CharClass.COENG:
                # COENG plus the following character (usually a consonant) are
                # part of this cluster.  A trailing COENG at end of input is
                # kept so normalisation, not the scanner, decides to drop it.
                i += 2 if i + 1 < n else 1
                continue
            if cls in (CharClass.SHIFTER, CharClass.VOWEL, CharClass.SIGN):
                i += 1
                continue
            break
        yield text[start:i]


def segment_syllables(text: str, *, keep_non_khmer: bool = True) -> list[str]:
    """Cluster list for a whole string.

    Runs of non-Khmer characters are coalesced into single tokens (so
    ``"QN-4500A"`` survives as one item) when ``keep_non_khmer`` is set.
    """
    out: list[str] = []
    buffer: list[str] = []
    for cluster in iter_clusters(text):
        if is_khmer_char(cluster[0]):
            if buffer:
                if keep_non_khmer:
                    out.append("".join(buffer))
                buffer = []
            out.append(cluster)
        else:
            buffer.append(cluster)
    if buffer and keep_non_khmer:
        out.append("".join(buffer))
    return out


def count_syllables(text: str) -> int:
    """Number of Khmer orthographic clusters - the right proxy for Khmer length.

    Only clusters that open with a base character count.  An orphan vowel or
    diacritic is emitted by :func:`iter_clusters` as its own item so nothing is
    lost, but it is not a syllable - counting it would let a run of stray marks
    masquerade as well-formed Khmer.
    """
    return sum(
        1 for cluster in iter_clusters(text) if classify_char(cluster[0]) is CharClass.BASE
    )


_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_./][A-Za-z0-9]+)*|\d+(?:[.,]\d+)*")
_KHMER_RUN = re.compile(r"[ក-៿᧠-᧿]+")


def split_script_runs(text: str) -> list[tuple[str, str]]:
    """Split into ``(script, run)`` pairs where script is ``"khmer"`` or ``"other"``."""
    runs: list[tuple[str, str]] = []
    pos = 0
    for match in _KHMER_RUN.finditer(text):
        if match.start() > pos:
            runs.append(("other", text[pos : match.start()]))
        runs.append(("khmer", match.group(0)))
        pos = match.end()
    if pos < len(text):
        runs.append(("other", text[pos:]))
    return runs


def tokenize_for_search(
    text: str, *, khmer_ngrams: tuple[int, ...] = (1, 2), lowercase: bool = True
) -> list[str]:
    """Tokens for BM25 / lexical matching.

    Khmer runs contribute syllable n-grams; Latin and numeric runs contribute
    whole words, which keeps SKUs and model numbers matchable exactly.  For
    example ``"តម្លៃ QN-4500A"`` yields the Khmer syllables of ``តម្លៃ``, their
    bigram, and the single token ``qn-4500a``.
    """
    tokens: list[str] = []
    for script, run in split_script_runs(text):
        if script == "khmer":
            syllables = [c for c in iter_clusters(run) if is_khmer_char(c[0])]
            for width in khmer_ngrams:
                if width == 1:
                    tokens.extend(syllables)
                else:
                    tokens.extend(
                        "".join(syllables[i : i + width])
                        for i in range(len(syllables) - width + 1)
                    )
        else:
            for match in _LATIN_TOKEN.finditer(run):
                token = match.group(0)
                tokens.append(token.lower() if lowercase else token)
    return tokens


def strip_marks(text: str) -> str:
    """Base characters only - used by the fuzzy spelling-variant matcher.

    Khmer readers routinely omit or mistype diacritics.  Comparing mark-stripped
    forms lets ``ត្រជាក់`` and ``ត្រជាក`` collapse to the same key without a
    lexicon.
    """
    return "".join(
        ch
        for ch in unicodedata.normalize("NFC", text)
        if classify_char(ch) not in (CharClass.SIGN, CharClass.VOWEL, CharClass.SHIFTER)
    )
