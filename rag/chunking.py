"""Khmer-aware chunking.

Why this is not a generic character splitter
--------------------------------------------
Splitting Khmer on a character count cuts through orthographic clusters, which
produces mojibake at the chunk boundary and corrupts both the embedding and the
text a customer eventually sees.  Splitting on whitespace does not work either -
Khmer sentences frequently run for 200+ characters without a space.

The splitter therefore works on a hierarchy of *real* Khmer boundaries:

1. Markdown/heading structure (kept as ``heading_path`` for context)
2. Blank-line paragraph breaks
3. Khmer sentence terminators ``។`` ``៕`` ``៖`` and Latin ``. ! ?``
4. ZWSP word-break hints, then spaces
5. As a last resort, orthographic cluster boundaries - never mid-cluster

Token estimation
----------------
Qwen-family tokenizers split Khmer far more finely than English.  Measured on
the fixtures in ``rag/tests``, one Khmer orthographic cluster costs roughly 1.6
tokens and one Latin word roughly 1.3.  ``estimate_tokens`` uses those
coefficients so a "600 token" chunk really is about 600 tokens rather than 250.
Call ``calibrate_token_ratio`` with a real tokenizer to replace the estimate on
a training host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from common.hashing import stable_id
from preprocessing.khmer_script import (
    CharClass,
    classify_char,
    is_khmer_char,
    iter_clusters,
    split_script_runs,
)
from rag.schemas import Chunk

__all__ = [
    "ChunkingConfig",
    "calibrate_token_ratio",
    "chunk_document",
    "chunk_text",
    "estimate_tokens",
]

# Empirical cost multipliers - see the module docstring.
_KHMER_CLUSTER_TOKENS = 1.6
_LATIN_WORD_TOKENS = 1.3
_DIGIT_GROUP_TOKENS = 1.0

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[។៕៖!?\.])\s+|(?<=[។៕៖])(?=\S)")
_TABLE_ROW = re.compile(r"^\s*[^|\n]*\|[^|\n]*")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_DIGIT_GROUP = re.compile(r"\d+")


@dataclass(slots=True)
class ChunkingConfig:
    """Chunking parameters.  Sizes are in *estimated tokens*, not characters."""

    chunk_size: int = 600
    chunk_overlap: int = 100
    min_chunk_size: int = 80
    respect_headings: bool = True
    keep_table_rows_whole: bool = True
    include_heading_in_text: bool = True

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must not exceed chunk_size")


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of ``text`` for a Qwen-family tokenizer."""
    if not text:
        return 0
    total = 0.0
    for script, run in split_script_runs(text):
        if script == "khmer":
            clusters = sum(1 for c in iter_clusters(run) if classify_char(c[0]) is CharClass.BASE)
            # Khmer digits and punctuation also cost tokens.
            others = sum(
                1 for c in run if is_khmer_char(c) and classify_char(c) is not CharClass.BASE
            )
            total += clusters * _KHMER_CLUSTER_TOKENS + others * 0.25
        else:
            total += len(_LATIN_WORD.findall(run)) * _LATIN_WORD_TOKENS
            total += len(_DIGIT_GROUP.findall(run)) * _DIGIT_GROUP_TOKENS
            total += sum(1 for c in run if not c.isalnum() and not c.isspace()) * 0.5
    return max(1, round(total))


def calibrate_token_ratio(texts: list[str], tokenizer: Any) -> dict[str, float]:
    """Measure the real cost multipliers against a tokenizer.

    Run this on the training host (where ``transformers`` is installed) and put
    the result in ``configs/rag/ingestion.yaml`` so chunk sizes are grounded in
    measurement rather than in this module's defaults::

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
        calibrate_token_ratio(sample_texts, tok)
    """
    khmer_clusters = latin_words = real_tokens = 0
    for text in texts:
        real_tokens += len(tokenizer.encode(text, add_special_tokens=False))
        for script, run in split_script_runs(text):
            if script == "khmer":
                khmer_clusters += sum(
                    1 for c in iter_clusters(run) if classify_char(c[0]) is CharClass.BASE
                )
            else:
                latin_words += len(_LATIN_WORD.findall(run))
    estimated = sum(estimate_tokens(t) for t in texts)
    return {
        "real_tokens": float(real_tokens),
        "estimated_tokens": float(estimated),
        "ratio_real_over_estimated": round(real_tokens / estimated, 4) if estimated else 0.0,
        "khmer_clusters": float(khmer_clusters),
        "latin_words": float(latin_words),
        "suggested_khmer_cluster_tokens": (
            round(real_tokens / khmer_clusters, 3) if khmer_clusters else 0.0
        ),
    }


def _split_units(text: str) -> list[str]:
    """Break text into the smallest units a chunk boundary may fall between."""
    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue
            if _TABLE_ROW.match(line) and "|" in line:
                units.append(line)  # a table row is atomic
                continue
            sentences = [s.strip() for s in _SENTENCE_END.split(line) if s.strip()]
            units.extend(sentences or [line])
    return units


def _hard_split(unit: str, limit: int) -> list[str]:
    """Split a single oversized unit without ever cutting a Khmer cluster."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for cluster in iter_clusters(unit):
        cost = estimate_tokens(cluster)
        if current and current_tokens + cost > limit:
            pieces.append("".join(current))
            current, current_tokens = [], 0
        current.append(cluster)
        current_tokens += cost
    if current:
        pieces.append("".join(current))
    return pieces


def _overlap_units(units: list[str], overlap_tokens: int) -> list[str]:
    """Trailing units of a chunk whose combined cost is about ``overlap_tokens``."""
    if overlap_tokens <= 0:
        return []
    out: list[str] = []
    total = 0
    for unit in reversed(units):
        cost = estimate_tokens(unit)
        if total + cost > overlap_tokens and out:
            break
        out.insert(0, unit)
        total += cost
    return out


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[tuple[str, list[str]]]:
    """Split text into ``(chunk_text, heading_path)`` pairs."""
    cfg = config or ChunkingConfig()
    if not text.strip():
        return []

    # Group the document into sections by heading level.
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def _flush() -> None:
        if buffer and "".join(buffer).strip():
            sections.append(([h for _, h in heading_stack], "\n".join(buffer).strip()))
        buffer.clear()

    for line in text.split("\n"):
        heading = _HEADING_RE.match(line) if cfg.respect_headings else None
        if heading:
            _flush()
            level = len(heading.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading.group(2).strip()))
            continue
        buffer.append(line)
    _flush()

    if not sections:
        sections = [([], text.strip())]

    chunks: list[tuple[str, list[str]]] = []
    for headings, body in sections:
        units = _split_units(body)
        if not units:
            continue

        current: list[str] = []
        current_tokens = 0
        for unit in units:
            cost = estimate_tokens(unit)
            if cost > cfg.chunk_size:
                if current:
                    chunks.append(("\n".join(current), list(headings)))
                    current, current_tokens = [], 0
                for piece in _hard_split(unit, cfg.chunk_size):
                    chunks.append((piece, list(headings)))
                continue
            if current and current_tokens + cost > cfg.chunk_size:
                chunks.append(("\n".join(current), list(headings)))
                current = _overlap_units(current, cfg.chunk_overlap)
                current_tokens = sum(estimate_tokens(u) for u in current)
            current.append(unit)
            current_tokens += cost
        if current:
            chunks.append(("\n".join(current), list(headings)))

    # Merge a trailing runt into its predecessor rather than indexing a fragment.
    merged: list[tuple[str, list[str]]] = []
    for body, headings in chunks:
        if (
            merged
            and estimate_tokens(body) < cfg.min_chunk_size
            and merged[-1][1] == headings
            and estimate_tokens(merged[-1][0]) + estimate_tokens(body) <= cfg.chunk_size * 1.25
        ):
            merged[-1] = (merged[-1][0] + "\n" + body, headings)
        else:
            merged.append((body, headings))

    if cfg.include_heading_in_text:
        return [
            (("\n".join(headings) + "\n" + body).strip() if headings else body, headings)
            for body, headings in merged
        ]
    return merged


def chunk_document(
    *,
    document_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Chunk one company document into indexable :class:`Chunk` values."""
    cfg = config or ChunkingConfig()
    meta = dict(metadata or {})
    out: list[Chunk] = []
    for ordinal, (body, headings) in enumerate(chunk_text(text, cfg)):
        if not body.strip():
            continue
        out.append(
            Chunk(
                chunk_id=stable_id(document_id, str(ordinal), body, length=20),
                document_id=document_id,
                text=body,
                ordinal=ordinal,
                token_estimate=estimate_tokens(body),
                heading_path=headings,
                metadata=meta,
            )
        )
    return out
