"""Khmer-English code-switching analysis and protected-span extraction.

Two production needs are served here.

**Training data curation** - the SFT mixture reserves 10% for code-switching
(§Phase 7).  ``analyse_code_switching`` scores how much genuine switching a
sample contains so the mixture can be measured rather than guessed.

**Runtime answer validation** - a support answer must reproduce a model number,
SKU, URL, price or date *exactly* as it appears in the source document.
``extract_protected_spans`` finds those spans in the retrieved context, and
``verify_protected_spans`` checks that the generated answer did not corrupt
them (``ក្យូអិន-៤៥០០អា`` instead of ``QN-4500A`` is a real failure mode when a
Khmer-tuned model over-transliterates).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from preprocessing.khmer_detection import ScriptProfile, profile_text
from preprocessing.khmer_script import split_script_runs

__all__ = [
    "CodeSwitchAnalysis",
    "ProtectedSpan",
    "SpanKind",
    "analyse_code_switching",
    "extract_protected_spans",
    "verify_protected_spans",
]


class SpanKind(StrEnum):
    URL = "url"
    EMAIL = "email"
    MODEL_NUMBER = "model_number"
    SKU = "sku"
    CURRENCY = "currency"
    NUMBER = "number"
    DATE = "date"
    MEASUREMENT = "measurement"
    LATIN_TERM = "latin_term"


# Order matters - the first pattern that claims a span wins, so URL beats the
# generic number rule inside "https://example.com/qn4500a".
_SPAN_PATTERNS: tuple[tuple[SpanKind, re.Pattern[str]], ...] = (
    (SpanKind.URL, re.compile(r"https?://[^\s<>\"'ក-៿]+|www\.[^\s<>\"'ក-៿]+")),
    (SpanKind.EMAIL, re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    (
        SpanKind.CURRENCY,
        re.compile(
            r"(?:\$|USD|KHR|៛)\s?\d[\d,.]*|"
            r"\d[\d,.]*\s?(?:\$|USD|KHR|៛|ដុល្លារ|រៀល)",
            re.IGNORECASE,
        ),
    ),
    (
        SpanKind.DATE,
        re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"),
    ),
    # A model number is a Latin token containing at least one digit, e.g.
    # QN-4500A, RF-22B, X100, SKU-001.  Pure words are excluded.
    # This runs BEFORE the measurement rule: without that ordering the unit
    # alternation ("A" for ampere) claims the tail of "QN-4500A" and splits the
    # model number in half, which is exactly the corruption the whole module
    # exists to prevent.
    (
        SpanKind.MODEL_NUMBER,
        re.compile(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z]{1,6}[-]?\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b"),
    ),
    (
        SpanKind.MEASUREMENT,
        re.compile(
            r"(?<![A-Za-z0-9\-])\d[\d.,]*\s?"
            r"(?:mm|cm|km|kg|mg|ml|kW|GHz|MHz|Hz|GB|TB|MB|BTU|rpm|inch|°C|°F|[mgWVAL])"
            r"(?![A-Za-z0-9])"
        ),
    ),
    (SpanKind.NUMBER, re.compile(r"\b\d[\d,.]*\b")),
    (SpanKind.LATIN_TERM, re.compile(r"\b[A-Za-z][A-Za-z'\-]{1,}\b")),
)

_MIN_LATIN_TERM_LEN = 2


@dataclass(slots=True, frozen=True)
class ProtectedSpan:
    """A substring that must be reproduced verbatim in an answer."""

    kind: SpanKind
    text: str
    start: int
    end: int

    def normalised(self) -> str:
        return self.text.strip().rstrip(".,;:)").lower()


@dataclass(slots=True)
class CodeSwitchAnalysis:
    """How much Khmer/Latin switching a text contains."""

    profile: ScriptProfile
    switch_points: int = 0
    khmer_runs: int = 0
    latin_runs: int = 0
    protected_spans: list[ProtectedSpan] = field(default_factory=list)

    @property
    def is_code_switched(self) -> bool:
        """True when both scripts carry real content, not just an odd character."""
        return (
            self.khmer_runs >= 1
            and self.latin_runs >= 1
            and self.profile.khmer_chars >= 3
            and self.profile.latin_chars >= 2
        )

    @property
    def switch_density(self) -> float:
        """Switches per 100 characters - separates 'a product name' from 'salad'."""
        return 100.0 * self.switch_points / (self.profile.length or 1)


def analyse_code_switching(text: str) -> CodeSwitchAnalysis:
    runs = split_script_runs(text)
    khmer_runs = sum(1 for script, run in runs if script == "khmer" and run.strip())
    latin_runs = sum(
        1 for script, run in runs if script == "other" and any(c.isalnum() for c in run)
    )
    switch_points = 0
    previous: str | None = None
    for script, run in runs:
        if not run.strip():
            continue
        if script == "other" and not any(c.isalnum() for c in run):
            continue
        if previous is not None and previous != script:
            switch_points += 1
        previous = script

    return CodeSwitchAnalysis(
        profile=profile_text(text),
        switch_points=switch_points,
        khmer_runs=khmer_runs,
        latin_runs=latin_runs,
        protected_spans=extract_protected_spans(text),
    )


def extract_protected_spans(text: str, *, include_latin_terms: bool = False) -> list[ProtectedSpan]:
    """Find spans that must survive a Khmer answer unchanged.

    Overlapping matches are resolved by pattern priority, then by length, so a
    URL is never fragmented into a "model number" plus a "number".
    """
    claimed: list[tuple[int, int]] = []
    spans: list[ProtectedSpan] = []

    for kind, pattern in _SPAN_PATTERNS:
        if kind is SpanKind.LATIN_TERM and not include_latin_terms:
            continue
        for match in pattern.finditer(text):
            start, end = match.span()
            if kind is SpanKind.LATIN_TERM and (end - start) < _MIN_LATIN_TERM_LEN:
                continue
            if any(start < c_end and c_start < end for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            spans.append(ProtectedSpan(kind=kind, text=match.group(0), start=start, end=end))

    spans.sort(key=lambda s: s.start)
    return spans


def verify_protected_spans(
    source_text: str,
    answer: str,
    *,
    kinds: frozenset[SpanKind] | None = None,
    require_all: bool = False,
) -> tuple[bool, list[ProtectedSpan]]:
    """Check that protected spans from ``source_text`` appear intact in ``answer``.

    Returns ``(ok, missing)``.  With ``require_all=False`` (the default) the
    check only fails when the answer *mentions* a span in mangled form or omits
    a span it clearly tried to state - an answer is allowed to be selective
    about which facts it repeats.  With ``require_all=True`` every span must be
    present, which is what the grounding evaluator uses.
    """
    interesting = kinds or frozenset(
        {
            SpanKind.MODEL_NUMBER,
            SpanKind.SKU,
            SpanKind.URL,
            SpanKind.EMAIL,
            SpanKind.CURRENCY,
            SpanKind.MEASUREMENT,
        }
    )
    source_spans = [s for s in extract_protected_spans(source_text) if s.kind in interesting]
    if not source_spans:
        return True, []

    answer_lower = answer.lower()
    answer_spans = {s.normalised() for s in extract_protected_spans(answer)}

    missing: list[ProtectedSpan] = []
    for span in source_spans:
        key = span.normalised()
        if key in answer_lower or key in answer_spans:
            continue
        if require_all:
            missing.append(span)
            continue
        # Partial-mention heuristic: the answer used the alphabetic stem of the
        # model number but not the full token, i.e. it corrupted it.
        stem = re.sub(r"[^A-Za-z]", "", span.text).lower()
        if len(stem) >= 2 and stem in answer_lower:
            missing.append(span)
    return (not missing), missing
