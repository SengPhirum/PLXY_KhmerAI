"""Composite quality scoring with an auditable rejection reason.

Every record that leaves the pipeline carries a ``quality_score`` in [0, 1] and,
when rejected, the *first* rule that rejected it.  Storing the reason (rather
than a bare boolean) is what makes a corpus run debuggable: after a pass you can
answer "why did 40% of CulturaX disappear?" from
``data/manifests/preprocessing_report.json`` alone.

Signals combined
----------------
``khmer_ratio``        meaningful Khmer content, not an English page with a
                       Khmer menu item
``length``             too short to teach anything; absurdly long usually means
                       a concatenated dump
``repetition``         the classic crawl failure - the same line or n-gram over
                       and over
``entropy``            character entropy detects placeholder/keyboard-mash text
``punctuation``        abnormal punctuation density marks OCR and template noise
``url_density``        link farms
``duplicate_lines``    navigation remnants that survived boilerplate removal
``digit_ratio``        price tables and phone-number dumps are not prose
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from preprocessing.khmer_detection import ScriptProfile, TextLanguage, detect_language
from preprocessing.khmer_script import is_khmer_char, iter_clusters

__all__ = [
    "QualityAssessment",
    "QualityThresholds",
    "assess_quality",
    "repetition_ratio",
    "shannon_entropy",
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PUNCT_RE = re.compile(r"[!-/:-@\[-`{-~។៕៖៘៙៚]")
_WORD_SPLIT = re.compile(r"\s+")


@dataclass(slots=True)
class QualityThresholds:
    """Tunable acceptance criteria.  Defaults target a Khmer pretraining corpus."""

    min_chars: int = 60
    max_chars: int = 100_000
    min_khmer_syllables: int = 12
    min_khmer_ratio: float = 0.60
    max_latin_ratio: float = 0.35
    max_digit_ratio: float = 0.30
    max_symbol_ratio: float = 0.25
    max_punctuation_ratio: float = 0.20
    max_url_density: float = 0.02
    max_repetition_ratio: float = 0.30
    max_duplicate_line_ratio: float = 0.30
    min_entropy: float = 2.2
    max_entropy: float = 6.5
    min_score: float = 0.55
    allowed_languages: tuple[str, ...] = (TextLanguage.KHMER, TextLanguage.KHMER_ENGLISH)

    @classmethod
    def for_sft(cls) -> QualityThresholds:
        """Support dialogue is shorter and more code-switched than web prose."""
        return cls(
            min_chars=8,
            min_khmer_syllables=2,
            min_khmer_ratio=0.30,
            max_latin_ratio=0.70,
            max_digit_ratio=0.45,
            max_url_density=0.05,
            min_entropy=1.5,
            min_score=0.45,
        )

    @classmethod
    def for_company_documents(cls) -> QualityThresholds:
        """Company docs contain tables, SKUs and prices - digits are expected."""
        return cls(
            min_chars=20,
            min_khmer_syllables=0,
            min_khmer_ratio=0.0,
            max_latin_ratio=1.0,
            max_digit_ratio=0.7,
            max_punctuation_ratio=0.4,
            min_entropy=1.2,
            min_score=0.30,
            allowed_languages=(
                TextLanguage.KHMER,
                TextLanguage.KHMER_ENGLISH,
                TextLanguage.ENGLISH,
            ),
        )


@dataclass(slots=True)
class QualityAssessment:
    """Score plus every signal that produced it."""

    score: float
    accepted: bool
    rejection_reason: str | None = None
    language: str = TextLanguage.INVALID
    signals: dict[str, float] = field(default_factory=dict)
    profile: ScriptProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "language": str(self.language),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "profile": self.profile.to_dict() if self.profile else None,
        }

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.accepted


def shannon_entropy(text: str) -> float:
    """Per-character Shannon entropy in bits.

    Keyboard mash and single-character padding score near zero; natural Khmer
    prose sits around 4.0-5.5 bits because the script has a large alphabet.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def repetition_ratio(text: str, *, width: int = 4) -> float:
    """Fraction of character n-grams that are repeats of an earlier n-gram.

    Uses Khmer syllable clusters for Khmer text so that ``ក`` repeated inside
    unrelated words is not mistaken for repetition.
    """
    units = [c for c in iter_clusters(text) if not c.isspace()]
    if len(units) <= width:
        return 0.0
    grams = ["".join(units[i : i + width]) for i in range(len(units) - width + 1)]
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


def duplicate_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def _url_density(text: str) -> float:
    if not text:
        return 0.0
    matched = sum(len(m.group(0)) for m in _URL_RE.finditer(text))
    return matched / len(text)


def _punctuation_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_PUNCT_RE.findall(text)) / len(text)


def _longest_run(text: str) -> int:
    """Longest run of one repeated character - catches ``!!!!!!!!!!`` and ``ាាាា``."""
    best = run = 0
    previous = ""
    for ch in text:
        if ch == previous:
            run += 1
        else:
            run = 1
            previous = ch
        best = max(best, run)
    return best


def _score_from_signals(signals: dict[str, float], thresholds: QualityThresholds) -> float:
    """Weighted blend of normalised signals; each component is in [0, 1]."""

    def _inverse(value: float, limit: float) -> float:
        if limit <= 0:
            return 1.0
        return max(0.0, 1.0 - (value / limit))

    khmer = min(1.0, signals["khmer_ratio"] / max(thresholds.min_khmer_ratio, 1e-6))
    length = min(1.0, signals["khmer_syllables"] / max(thresholds.min_khmer_syllables or 1, 1))
    entropy_mid = (thresholds.min_entropy + thresholds.max_entropy) / 2
    entropy = max(
        0.0,
        1.0 - abs(signals["entropy"] - entropy_mid) / max(entropy_mid, 1e-6),
    )
    components = {
        "khmer": (khmer, 0.30),
        "length": (length, 0.15),
        "repetition": (_inverse(signals["repetition"], thresholds.max_repetition_ratio), 0.15),
        "duplicate_lines": (
            _inverse(signals["duplicate_lines"], thresholds.max_duplicate_line_ratio),
            0.10,
        ),
        "entropy": (entropy, 0.10),
        "punctuation": (
            _inverse(signals["punctuation"], thresholds.max_punctuation_ratio),
            0.08,
        ),
        "urls": (_inverse(signals["url_density"], thresholds.max_url_density), 0.07),
        "digits": (_inverse(signals["digit_ratio"], thresholds.max_digit_ratio), 0.05),
    }
    total_weight = sum(weight for _, weight in components.values())
    return (
        sum(min(1.0, max(0.0, value)) * weight for value, weight in components.values())
        / total_weight
    )


def assess_quality(text: str, thresholds: QualityThresholds | None = None) -> QualityAssessment:
    """Score ``text`` and decide whether it is fit for training."""
    th = thresholds or QualityThresholds()
    language, profile = detect_language(text)

    signals: dict[str, float] = {
        "chars": float(len(text)),
        "khmer_ratio": profile.khmer_ratio,
        "latin_ratio": profile.latin_ratio,
        "digit_ratio": profile.digit_ratio,
        "symbol_ratio": profile.symbol_ratio,
        "khmer_syllables": float(profile.khmer_syllables),
        "entropy": shannon_entropy(text),
        "repetition": repetition_ratio(text),
        "duplicate_lines": duplicate_line_ratio(text),
        "url_density": _url_density(text),
        "punctuation": _punctuation_ratio(text),
        "longest_char_run": float(_longest_run(text)),
        "words": float(len(_WORD_SPLIT.split(text.strip())) if text.strip() else 0),
    }

    score = _score_from_signals(signals, th)

    # Ordered checks - the first failure is the recorded reason.
    checks: list[tuple[bool, str]] = [
        (not text.strip(), "empty"),
        (len(text) < th.min_chars, "too_short"),
        (len(text) > th.max_chars, "too_long"),
        # Structural failures are checked before the language label so that a
        # keyboard-mash record is reported as such rather than as "english".
        (signals["longest_char_run"] > 30, "character_run"),
        (signals["entropy"] < th.min_entropy, "low_entropy"),
        (str(language) not in th.allowed_languages, f"language_{language}"),
        (profile.khmer_syllables < th.min_khmer_syllables, "insufficient_khmer"),
        (profile.khmer_ratio < th.min_khmer_ratio, "low_khmer_ratio"),
        (profile.latin_ratio > th.max_latin_ratio, "excess_latin"),
        (profile.digit_ratio > th.max_digit_ratio, "excess_digits"),
        (profile.symbol_ratio > th.max_symbol_ratio, "excess_symbols"),
        (signals["punctuation"] > th.max_punctuation_ratio, "excess_punctuation"),
        (signals["url_density"] > th.max_url_density, "link_farm"),
        (signals["repetition"] > th.max_repetition_ratio, "repetitive"),
        (signals["duplicate_lines"] > th.max_duplicate_line_ratio, "duplicate_lines"),
        (signals["entropy"] > th.max_entropy, "high_entropy"),
        (
            profile.khmer_syllables > 0 and not any(is_khmer_char(c) for c in text),
            "inconsistent_profile",
        ),
        (score < th.min_score, "low_score"),
    ]
    for failed, reason in checks:
        if failed:
            return QualityAssessment(
                score=score,
                accepted=False,
                rejection_reason=reason,
                language=str(language),
                signals=signals,
                profile=profile,
            )

    return QualityAssessment(
        score=score,
        accepted=True,
        rejection_reason=None,
        language=str(language),
        signals=signals,
        profile=profile,
    )


def summarise_rejections(assessments: list[QualityAssessment]) -> dict[str, Any]:
    """Aggregate for the pipeline audit report."""
    reasons = Counter(a.rejection_reason for a in assessments if not a.accepted)
    accepted = sum(1 for a in assessments if a.accepted)
    scores = [a.score for a in assessments]
    return {
        "total": len(assessments),
        "accepted": accepted,
        "rejected": len(assessments) - accepted,
        "acceptance_rate": round(accepted / len(assessments), 4) if assessments else 0.0,
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "rejection_reasons": dict(reasons.most_common()),
    }
