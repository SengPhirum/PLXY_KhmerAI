"""Khmer-aware text preprocessing.

Public surface used by the rest of the platform::

    from preprocessing import (
        normalize_text,          # conservative Khmer Unicode normalisation
        detect_language,         # khmer / khmer_english / english / other / invalid
        assess_quality,          # composite score + rejection reason
        segment_syllables,       # orthographic clusters (Khmer has no spaces)
        tokenize_for_search,     # BM25 tokens
    )
"""

from preprocessing.khmer_detection import (
    ScriptProfile,
    TextLanguage,
    detect_language,
    khmer_ratio,
    profile_text,
)
from preprocessing.khmer_script import (
    count_syllables,
    iter_clusters,
    segment_syllables,
    split_script_runs,
    tokenize_for_search,
)
from preprocessing.language_mixing import (
    CodeSwitchAnalysis,
    ProtectedSpan,
    SpanKind,
    analyse_code_switching,
    extract_protected_spans,
    verify_protected_spans,
)
from preprocessing.quality_filter import QualityAssessment, QualityThresholds, assess_quality
from preprocessing.schemas import INTENTS, CleanRecord, SFTRecord
from preprocessing.unicode_normalization import (
    NormalizationConfig,
    NormalizationReport,
    normalize_for_hashing,
    normalize_khmer,
    normalize_text,
)

__all__ = [
    "INTENTS",
    "CleanRecord",
    "CodeSwitchAnalysis",
    "NormalizationConfig",
    "NormalizationReport",
    "ProtectedSpan",
    "QualityAssessment",
    "QualityThresholds",
    "SFTRecord",
    "ScriptProfile",
    "SpanKind",
    "TextLanguage",
    "analyse_code_switching",
    "assess_quality",
    "count_syllables",
    "detect_language",
    "extract_protected_spans",
    "iter_clusters",
    "khmer_ratio",
    "normalize_for_hashing",
    "normalize_khmer",
    "normalize_text",
    "profile_text",
    "segment_syllables",
    "split_script_runs",
    "tokenize_for_search",
    "verify_protected_spans",
]
