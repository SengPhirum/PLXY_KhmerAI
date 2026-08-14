"""Language detection and character-composition analysis for Khmer text.

No third-party language-ID model is used.  For the two-language problem this
platform actually has (Khmer vs English, plus their mixture), script ratios are
both more accurate and far cheaper than a general-purpose classifier - fastText
and CLD3 routinely mislabel a two-word Khmer fragment, while the script ratio
cannot.

The classifier returns one of:

``khmer``        predominantly Khmer script
``khmer_english`` genuine code-switching - Khmer sentence structure with Latin
                  product names, model numbers or technical terms
``english``      predominantly Latin script
``other``        another script dominates
``invalid``      empty, control-character soup, or below the minimum length
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from preprocessing.khmer_script import (
    KHMER_DIGITS,
    CharClass,
    classify_char,
    count_syllables,
    is_khmer_char,
)

__all__ = [
    "TextLanguage",
    "ScriptProfile",
    "profile_text",
    "detect_language",
    "khmer_ratio",
]


class TextLanguage(StrEnum):
    KHMER = "khmer"
    KHMER_ENGLISH = "khmer_english"
    ENGLISH = "english"
    OTHER = "other"
    INVALID = "invalid"


@dataclass(slots=True, frozen=True)
class ScriptProfile:
    """Character-composition statistics for one text."""

    length: int
    khmer_chars: int
    latin_chars: int
    digit_chars: int
    khmer_digit_chars: int
    symbol_chars: int
    control_chars: int
    whitespace_chars: int
    other_script_chars: int
    khmer_syllables: int

    @property
    def scriptful(self) -> int:
        """Characters that carry script identity (letters only)."""
        return self.khmer_chars + self.latin_chars + self.other_script_chars

    @property
    def khmer_ratio(self) -> float:
        return self.khmer_chars / self.scriptful if self.scriptful else 0.0

    @property
    def latin_ratio(self) -> float:
        return self.latin_chars / self.scriptful if self.scriptful else 0.0

    @property
    def other_script_ratio(self) -> float:
        return self.other_script_chars / self.scriptful if self.scriptful else 0.0

    @property
    def digit_ratio(self) -> float:
        total = self.length or 1
        return (self.digit_chars + self.khmer_digit_chars) / total

    @property
    def symbol_ratio(self) -> float:
        return self.symbol_chars / (self.length or 1)

    @property
    def control_ratio(self) -> float:
        return self.control_chars / (self.length or 1)

    @property
    def whitespace_ratio(self) -> float:
        return self.whitespace_chars / (self.length or 1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            khmer_ratio=round(self.khmer_ratio, 4),
            latin_ratio=round(self.latin_ratio, 4),
            digit_ratio=round(self.digit_ratio, 4),
            symbol_ratio=round(self.symbol_ratio, 4),
            control_ratio=round(self.control_ratio, 4),
        )
        return data


def profile_text(text: str) -> ScriptProfile:
    """Count characters by class.  O(n), no allocations per character."""
    khmer = latin = digits = khmer_digits = symbols = controls = spaces = other = 0
    for ch in text:
        cls = classify_char(ch)
        if cls is CharClass.KHMER_DIGIT or ch in KHMER_DIGITS:
            khmer_digits += 1
        elif is_khmer_char(ch):
            if cls is CharClass.KHMER_PUNCT:
                symbols += 1
            else:
                khmer += 1
        elif cls is CharClass.LATIN:
            # `str.isalpha()` is true for every script; separate Latin from the rest.
            if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("À" <= ch <= "ɏ"):
                latin += 1
            else:
                other += 1
        elif cls is CharClass.DIGIT:
            digits += 1
        elif cls is CharClass.WHITESPACE:
            spaces += 1
        elif cls is CharClass.ZERO_WIDTH:
            controls += 1
        elif cls is CharClass.OTHER:
            symbols += 1
        else:
            symbols += 1
    return ScriptProfile(
        length=len(text),
        khmer_chars=khmer,
        latin_chars=latin,
        digit_chars=digits,
        khmer_digit_chars=khmer_digits,
        symbol_chars=symbols,
        control_chars=controls,
        whitespace_chars=spaces,
        other_script_chars=other,
        khmer_syllables=count_syllables(text),
    )


def khmer_ratio(text: str) -> float:
    """Fraction of script-bearing characters that are Khmer."""
    return profile_text(text).khmer_ratio


def detect_language(
    text: str,
    *,
    min_length: int = 3,
    khmer_dominant: float = 0.70,
    khmer_present: float = 0.15,
    latin_dominant: float = 0.85,
    max_control_ratio: float = 0.10,
) -> tuple[TextLanguage, ScriptProfile]:
    """Classify ``text``.  Returns the label and the profile it was based on.

    Thresholds are deliberately explicit parameters: the ingestion pipeline uses
    stricter values for training corpora than the API uses for a live customer
    message, where a two-word reply like ``"បាទ ok"`` must still be Khmer.
    """
    profile = profile_text(text)
    stripped = text.strip()

    if not stripped or profile.scriptful == 0:
        return TextLanguage.INVALID, profile
    if len(stripped) < min_length and profile.khmer_syllables == 0:
        return TextLanguage.INVALID, profile
    if profile.control_ratio > max_control_ratio:
        return TextLanguage.INVALID, profile

    if profile.khmer_ratio >= khmer_dominant:
        # Latin present but Khmer clearly dominant: still code-switching if the
        # Latin part is a real word rather than an incidental letter.
        if profile.latin_chars >= 2 and profile.latin_ratio >= 0.05:
            return TextLanguage.KHMER_ENGLISH, profile
        return TextLanguage.KHMER, profile

    if profile.khmer_ratio >= khmer_present:
        return TextLanguage.KHMER_ENGLISH, profile

    if profile.latin_ratio >= latin_dominant:
        return TextLanguage.ENGLISH, profile

    return TextLanguage.OTHER, profile
