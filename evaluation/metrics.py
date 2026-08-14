"""Evaluation metrics - retrieval, text similarity, Khmer quality, latency.

Khmer-specific choices
----------------------
Standard text metrics assume whitespace tokenisation and therefore report
nonsense on Khmer.  Every metric here works on **orthographic syllables** from
``preprocessing.khmer_script``:

* ``token_f1`` over syllables rather than "words"
* ``chrf`` over character n-grams, which is the metric the MT community uses for
  low-resource and space-less languages precisely because BLEU degrades there
* ``khmer_fluency`` heuristics that look for the failure modes a Khmer-tuned
  model actually shows: orphan diacritics, Latin leakage, over-transliteration
  of product names, and repetition loops
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

from preprocessing.khmer_detection import TextLanguage, detect_language
from preprocessing.khmer_script import (
    CharClass,
    classify_char,
    is_khmer_char,
    iter_clusters,
    segment_syllables,
)
from preprocessing.language_mixing import SpanKind, extract_protected_spans
from preprocessing.unicode_normalization import normalize_for_hashing

__all__ = [
    "aggregate",
    "chrf",
    "contains_all",
    "exact_match",
    "khmer_fluency",
    "latency_percentiles",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "token_f1",
]


# --- retrieval --------------------------------------------------------------
def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant items present in the top ``k``."""
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    return sum(1 for item in top if item in set(relevant)) / len(top)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant hit."""
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in relevant_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_set), k) + 1))
    return dcg / ideal if ideal else 0.0


# --- text similarity --------------------------------------------------------
def _syllables(text: str) -> list[str]:
    return [s for s in segment_syllables(normalize_for_hashing(text)) if s.strip()]


def token_f1(prediction: str, reference: str) -> float:
    """F1 over Khmer syllables (and whole Latin tokens)."""
    predicted = Counter(_syllables(prediction))
    expected = Counter(_syllables(reference))
    if not predicted or not expected:
        return 1.0 if predicted == expected else 0.0
    overlap = sum((predicted & expected).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def _char_ngrams(text: str, n: int) -> Counter[str]:
    stripped = normalize_for_hashing(text)
    if len(stripped) < n:
        return Counter([stripped] if stripped else [])
    return Counter(stripped[i : i + n] for i in range(len(stripped) - n + 1))


def chrf(prediction: str, reference: str, *, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF - character n-gram F-score, the standard metric for Khmer MT/QA."""
    if not prediction.strip() or not reference.strip():
        return 1.0 if prediction.strip() == reference.strip() else 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    for n in range(1, max_n + 1):
        predicted = _char_ngrams(prediction, n)
        expected = _char_ngrams(reference, n)
        overlap = sum((predicted & expected).values())
        precisions.append(overlap / sum(predicted.values()) if predicted else 0.0)
        recalls.append(overlap / sum(expected.values()) if expected else 0.0)

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_for_hashing(prediction) == normalize_for_hashing(reference) else 0.0


def contains_all(text: str, required: Sequence[str]) -> tuple[bool, list[str]]:
    """Case/format-insensitive substring check.  Returns ``(ok, missing)``."""
    normalised = normalize_for_hashing(text)
    missing = [
        needle
        for needle in required
        if needle not in text and normalize_for_hashing(needle) not in normalised
    ]
    return (not missing), missing


# --- Khmer quality ----------------------------------------------------------
def _orphan_mark_ratio(text: str) -> float:
    """Share of clusters that begin with a combining mark - a rendering defect."""
    clusters = [c for c in iter_clusters(text) if is_khmer_char(c[0])]
    if not clusters:
        return 0.0
    orphans = sum(
        1
        for c in clusters
        if classify_char(c[0])
        in (CharClass.VOWEL, CharClass.SIGN, CharClass.SHIFTER, CharClass.COENG)
    )
    return orphans / len(clusters)


def _repetition_loop(text: str, *, window: int = 6) -> float:
    """Detects a decoder stuck in a loop (the classic low-temperature failure).

    The statistic is the *duplicate fraction* of syllable n-grams,
    ``1 - unique/total``.  Max-frequency-of-one-n-gram was the obvious choice and
    is wrong: a phrase repeated 25 times still contributes only ~25 of ~300
    n-grams, so it scores near zero.  The duplicate fraction goes to ~1.0 for
    that case while natural Khmer prose stays below ~0.2.
    """
    units = [c for c in iter_clusters(text) if not c.isspace()]
    if len(units) < window * 3:
        return 0.0
    grams = ["".join(units[i : i + window]) for i in range(len(units) - window + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def _transliterated_identifiers(text: str, source: str) -> list[str]:
    """Identifiers present in the source but rewritten in the answer."""
    source_ids = {
        s.normalised()
        for s in extract_protected_spans(source)
        if s.kind in (SpanKind.MODEL_NUMBER, SpanKind.SKU)
    }
    if not source_ids:
        return []
    lowered = text.lower()
    return sorted(i for i in source_ids if i not in lowered)


def khmer_fluency(text: str, *, source: str = "", expect_language: str = "km") -> dict[str, Any]:
    """Heuristic Khmer quality signals with a 0-1 composite score.

    This is a *screen*, not a replacement for native review: it reliably catches
    mechanical failures (wrong language, orphan marks, repetition loops,
    mangled model numbers) and deliberately says nothing about whether the Khmer
    is idiomatic.  Items it cannot judge are flagged ``needs_human_review``.
    """
    if not text.strip():
        return {
            "score": 0.0,
            "language": "invalid",
            "empty": True,
            "needs_human_review": False,
            "issues": ["empty"],
        }

    language, profile = detect_language(text, khmer_present=0.10)
    issues: list[str] = []

    correct_language = (
        str(language).startswith("khmer")
        if expect_language == "km"
        else language is TextLanguage.ENGLISH
    )
    if not correct_language:
        issues.append(f"wrong_language:{language}")

    orphans = _orphan_mark_ratio(text)
    if orphans > 0.02:
        issues.append(f"orphan_marks:{orphans:.2f}")

    repetition = _repetition_loop(text)
    if repetition > 0.50:
        issues.append(f"repetition_loop:{repetition:.2f}")

    latin_ratio = profile.latin_ratio
    if expect_language == "km" and latin_ratio > 0.45:
        issues.append(f"latin_leakage:{latin_ratio:.2f}")

    mangled = _transliterated_identifiers(text, source) if source else []
    if mangled:
        issues.append("identifiers_missing:" + ",".join(mangled[:3]))

    score = 1.0
    score -= 0.0 if correct_language else 0.5
    score -= min(0.25, orphans * 5)
    score -= min(0.25, repetition)
    score -= 0.10 if (expect_language == "km" and latin_ratio > 0.45) else 0.0
    score -= min(0.20, 0.10 * len(mangled))
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "language": str(language),
        "khmer_ratio": round(profile.khmer_ratio, 4),
        "latin_ratio": round(latin_ratio, 4),
        "syllables": profile.khmer_syllables,
        "orphan_mark_ratio": round(orphans, 4),
        "repetition": round(repetition, 4),
        "mangled_identifiers": mangled,
        "issues": issues,
        # A mechanically clean answer still needs a human to judge naturalness.
        "needs_human_review": not issues,
        "empty": False,
    }


# --- latency ----------------------------------------------------------------
def latency_percentiles(values: Sequence[float]) -> dict[str, float]:
    """p50/p90/p95/p99 plus mean/min/max, using nearest-rank percentiles."""
    if not values:
        return {}
    ordered = sorted(values)

    def _percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = max(1, math.ceil(p / 100.0 * len(ordered)))
        return ordered[min(rank, len(ordered)) - 1]

    return {
        "count": float(len(ordered)),
        "mean": round(statistics.fmean(ordered), 2),
        "min": round(ordered[0], 2),
        "p50": round(_percentile(50), 2),
        "p90": round(_percentile(90), 2),
        "p95": round(_percentile(95), 2),
        "p99": round(_percentile(99), 2),
        "max": round(ordered[-1], 2),
    }


def aggregate(scores: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of every key present across the score dicts."""
    if not scores:
        return {}
    keys = {k for s in scores for k in s}
    out: dict[str, float] = {}
    for key in sorted(keys):
        values = [float(s[key]) for s in scores if key in s]
        if values:
            out[key] = round(statistics.fmean(values), 4)
    return out
