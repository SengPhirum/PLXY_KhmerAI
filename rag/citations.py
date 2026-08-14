"""Citation building and grounding verification.

Two jobs:

``build_context_block``
    Render retrieved chunks into the delimited, numbered block the runtime
    prompt expects.  Numbering matters: the model is instructed to cite ``[1]``,
    ``[2]`` and the API maps those back to real ``document_id``s, so a citation
    can be audited without trusting the model to reproduce an ID.

``verify_grounding``
    Check an answer against the context it was given.  Reports which sentences
    carry a business claim (a price, a period, a model number, a policy verb)
    that is *not* supported by the retrieved text.  This is what
    ``evaluation/evaluate_grounding.py`` scores and what
    ``server/guardrails.py`` uses to downgrade or flag a live answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from preprocessing.khmer_script import is_khmer_char, iter_clusters
from preprocessing.language_mixing import SpanKind, extract_protected_spans
from preprocessing.unicode_normalization import normalize_for_hashing
from rag.schemas import RetrievalResult, RetrievedChunk
from security.prompt_injection import wrap_context

__all__ = [
    "Citation",
    "GroundingReport",
    "build_citations",
    "build_context_block",
    "split_sentences",
    "verify_grounding",
]

_SENTENCE_SPLIT = re.compile(r"(?<=[។៕៖!?\.])\s+|\n+")
# Citation markers are metadata, not content.  They must be removed before claim
# extraction or the "1" in "[1]" is read as an unsupported numeric claim - which
# would make every correctly-cited answer fail grounding.
_CITATION_MARKER = re.compile(r"\[\d+\]")
_CLAIM_KINDS = frozenset(
    {
        SpanKind.CURRENCY,
        SpanKind.MEASUREMENT,
        SpanKind.MODEL_NUMBER,
        SpanKind.SKU,
        SpanKind.URL,
        SpanKind.EMAIL,
        SpanKind.NUMBER,
    }
)
# Khmer and English phrases that mark a *policy* claim - "must", "is covered",
# "is not included", "you are entitled to".  A sentence containing one of these
# asserts a company rule and therefore needs support even without a number.
_POLICY_MARKERS = (
    "ត្រូវតែ",
    "មិនរាប់បញ្ចូល",
    "គ្របដណ្តប់",
    "អាចទាមទារ",
    "មិនអាច",
    "តម្រូវឱ្យ",
    "ការធានា",
    "គោលការណ៍",
    "must",
    "is covered",
    "not covered",
    "entitled",
    "policy states",
    "guaranteed",
)
_HEDGE_MARKERS = (
    "ខ្ញុំមិនមានព័ត៌មាន",
    "មិនច្បាស់",
    "សូមទាក់ទង",
    "មិនមានក្នុងឯកសារ",
    "ខ្ញុំមិនអាចបញ្ជាក់",
    "i don't have",
    "i cannot confirm",
    "please contact",
)


@dataclass(slots=True, frozen=True)
class Citation:
    marker: str
    document_id: str
    title: str
    version: str
    effective_date: str
    source: str
    chunk_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "marker": self.marker,
            "document_id": self.document_id,
            "title": self.title,
            "version": self.version,
            "effective_date": self.effective_date,
            "source": self.source,
            "chunk_id": self.chunk_id,
        }


@dataclass(slots=True)
class GroundingReport:
    """Which parts of an answer the retrieved context does and does not support."""

    supported_sentences: list[str] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    unsupported_values: list[str] = field(default_factory=list)
    hedged: bool = False
    citation_markers: list[str] = field(default_factory=list)
    invalid_markers: list[str] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.supported_sentences) + len(self.unsupported_sentences)

    @property
    def grounding_precision(self) -> float:
        """Share of claim-bearing sentences that the context supports."""
        if self.total_claims == 0:
            return 1.0
        return len(self.supported_sentences) / self.total_claims

    @property
    def is_grounded(self) -> bool:
        return not self.unsupported_sentences and not self.invalid_markers

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.is_grounded,
            "grounding_precision": round(self.grounding_precision, 4),
            "claims": self.total_claims,
            "unsupported_sentences": self.unsupported_sentences[:10],
            "unsupported_values": self.unsupported_values[:20],
            "hedged": self.hedged,
            "citation_markers": self.citation_markers,
            "invalid_markers": self.invalid_markers,
        }


def split_sentences(text: str) -> list[str]:
    """Split on Khmer and Latin sentence terminators."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """One citation per chunk, numbered from 1 in retrieval order."""
    return [
        Citation(
            marker=f"[{index}]",
            document_id=chunk.document_id,
            title=chunk.document_title,
            version=chunk.version,
            effective_date=chunk.effective_date,
            source=chunk.source,
            chunk_id=chunk.chunk_id,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def build_context_block(
    result: RetrievalResult,
    *,
    open_tag: str = "<retrieved_company_context>",
    close_tag: str = "</retrieved_company_context>",
    include_metadata: bool = True,
) -> tuple[str, list[Citation]]:
    """Render the delimited context block plus its citation table.

    Each chunk is prefixed with its provenance so the model can honour the
    "prefer the newest active document" and "flag conflicts" instructions
    without a second retrieval call.
    """
    citations = build_citations(result.chunks)
    if not result.chunks:
        return "", citations

    rendered: list[str] = []
    for citation, chunk in zip(citations, result.chunks, strict=True):
        header = citation.marker
        if include_metadata:
            bits = [f"ឯកសារ៖ {chunk.document_title or chunk.document_id}"]
            if chunk.version:
                bits.append(f"កំណែ៖ {chunk.version}")
            if chunk.effective_date:
                bits.append(f"ចូលជាធរមាន៖ {chunk.effective_date}")
            if chunk.product_id:
                bits.append(f"ផលិតផល៖ {chunk.product_id}")
            bits.append(f"ស្ថានភាព៖ {chunk.status}")
            header = f"{citation.marker} ({' | '.join(bits)})"
        rendered.append(f"{header}\n{chunk.text.strip()}")

    body = wrap_context(rendered, open_tag=open_tag, close_tag=close_tag)

    if result.conflicts:
        warnings = "\n".join(
            f"- {conflict.describe()} (ឯកសារ៖ {', '.join(conflict.document_ids)})"
            for conflict in result.conflicts
        )
        body += f"\n\n<context_conflicts>\n{warnings}\n</context_conflicts>"
    return body, citations


def _claim_values(text: str) -> set[str]:
    return {
        span.normalised() for span in extract_protected_spans(text) if span.kind in _CLAIM_KINDS
    }


def _khmer_content_syllables(text: str) -> set[str]:
    return {c for c in iter_clusters(text) if is_khmer_char(c[0])}


def verify_grounding(
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    min_syllable_overlap: float = 0.55,
) -> GroundingReport:
    """Check every claim-bearing sentence of ``answer`` against ``chunks``.

    A sentence is *claim-bearing* when it contains a number, price, measurement,
    identifier, URL, or a policy marker.  Conversational sentences ("សូមអរគុណ")
    are ignored - requiring support for them would make every polite answer look
    ungrounded.

    Support is established when the sentence's concrete values all appear in the
    context, or when its Khmer content overlaps a context chunk strongly enough
    to be a paraphrase.
    """
    report = GroundingReport()
    if not answer.strip():
        return report

    report.hedged = any(marker in answer for marker in _HEDGE_MARKERS)
    report.citation_markers = sorted(set(re.findall(r"\[\d+\]", answer)))
    valid_markers = {f"[{i}]" for i in range(1, len(chunks) + 1)}
    report.invalid_markers = [m for m in report.citation_markers if m not in valid_markers]

    if not chunks:
        # With no context, any claim-bearing sentence is unsupported unless the
        # answer is explicitly hedged.
        for raw_sentence in split_sentences(answer):
            sentence = _CITATION_MARKER.sub("", raw_sentence).strip()
            if sentence and _is_claim(sentence) and not report.hedged:
                report.unsupported_sentences.append(sentence)
                report.unsupported_values.extend(sorted(_claim_values(sentence)))
        return report

    context_text = "\n".join(chunk.text for chunk in chunks)
    context_values = _claim_values(context_text)
    context_normalised = normalize_for_hashing(context_text)
    context_syllable_sets = [_khmer_content_syllables(chunk.text) for chunk in chunks]

    for raw_sentence in split_sentences(answer):
        sentence = _CITATION_MARKER.sub("", raw_sentence).strip()
        if not sentence or not _is_claim(sentence):
            continue
        values = _claim_values(sentence)
        missing = sorted(
            value
            for value in values
            if value not in context_values and value not in context_normalised
        )
        if missing:
            report.unsupported_sentences.append(sentence)
            report.unsupported_values.extend(missing)
            continue

        if values:
            report.supported_sentences.append(sentence)
            continue

        # Policy claim with no numbers: require lexical overlap with some chunk.
        sentence_syllables = _khmer_content_syllables(sentence)
        if not sentence_syllables:
            report.supported_sentences.append(sentence)
            continue
        best = max(
            (
                len(sentence_syllables & chunk_syllables) / len(sentence_syllables)
                for chunk_syllables in context_syllable_sets
            ),
            default=0.0,
        )
        if best >= min_syllable_overlap:
            report.supported_sentences.append(sentence)
        else:
            report.unsupported_sentences.append(sentence)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    report.unsupported_values = [
        v for v in report.unsupported_values if not (v in seen or seen.add(v))
    ]
    return report


def _is_claim(sentence: str) -> bool:
    if _claim_values(sentence):
        return True
    lowered = sentence.lower()
    return any(marker in sentence or marker in lowered for marker in _POLICY_MARKERS)
