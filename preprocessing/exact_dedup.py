"""Exact and normalised-form deduplication.

Two passes, because they catch different things:

``exact``       SHA-256 of the raw text - identical crawl copies.
``normalised``  SHA-256 of :func:`preprocessing.unicode_normalization.normalize_for_hashing`
                - the same text differing only in whitespace, ZWSP placement,
                punctuation or Latin case.  On Khmer web data this second pass
                typically removes several times as much as the first, because
                the same article is republished with different ZWSP conventions.

The deduplicator is streaming and keeps only the hashes, so a corpus far larger
than RAM can be processed.  ``keep`` decides which copy survives: ``"first"``
(cheapest) or ``"best"`` (highest quality score, requires buffering the winner
per hash but not the whole corpus).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from common.hashing import sha256_text
from preprocessing.unicode_normalization import normalize_for_hashing

__all__ = ["DedupStats", "ExactDeduplicator", "dedupe_exact", "content_hash", "normalised_hash"]

KeepPolicy = Literal["first", "best"]


def content_hash(text: str) -> str:
    """Stable hash of the exact text."""
    return sha256_text(text)


def normalised_hash(text: str) -> str:
    """Stable hash of the aggressive dedup normal form."""
    return sha256_text(normalize_for_hashing(text))


@dataclass(slots=True)
class DedupStats:
    seen: int = 0
    kept: int = 0
    exact_duplicates: int = 0
    normalised_duplicates: int = 0
    empty: int = 0
    duplicate_sources: dict[str, int] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return self.exact_duplicates + self.normalised_duplicates + self.empty

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "removed": self.removed,
            "exact_duplicates": self.exact_duplicates,
            "normalised_duplicates": self.normalised_duplicates,
            "empty": self.empty,
            "duplicate_rate": round(self.removed / self.seen, 4) if self.seen else 0.0,
            "duplicate_sources": dict(
                sorted(self.duplicate_sources.items(), key=lambda kv: -kv[1])[:20]
            ),
        }


class ExactDeduplicator:
    """Streaming exact + normalised-form deduplicator.

    >>> d = ExactDeduplicator()
    >>> d.is_new("សួស្តី")
    True
    >>> d.is_new("សួស្តី ")   # differs only in trailing space
    False
    """

    def __init__(self, *, use_normalised: bool = True) -> None:
        self.use_normalised = use_normalised
        self._exact: set[str] = set()
        self._normalised: set[str] = set()
        self.stats = DedupStats()

    def __len__(self) -> int:
        return len(self._exact)

    def is_new(self, text: str, *, source: str | None = None) -> bool:
        """Register ``text`` and report whether it had not been seen before."""
        self.stats.seen += 1
        if not text or not text.strip():
            self.stats.empty += 1
            return False

        exact = content_hash(text)
        if exact in self._exact:
            self.stats.exact_duplicates += 1
            self._note_source(source)
            return False

        if self.use_normalised:
            norm = normalised_hash(text)
            if norm in self._normalised:
                self.stats.normalised_duplicates += 1
                self._note_source(source)
                # Still record the exact hash so a third identical copy is cheap.
                self._exact.add(exact)
                return False
            self._normalised.add(norm)

        self._exact.add(exact)
        self.stats.kept += 1
        return True

    def contains(self, text: str) -> bool:
        """Membership test that does *not* mutate the state or the statistics."""
        if content_hash(text) in self._exact:
            return True
        return self.use_normalised and normalised_hash(text) in self._normalised

    def add_all(self, texts: Iterable[str]) -> None:
        """Preload hashes (used to seed a train-set deduper with the test set)."""
        for text in texts:
            if not text.strip():
                continue
            self._exact.add(content_hash(text))
            if self.use_normalised:
                self._normalised.add(normalised_hash(text))

    def _note_source(self, source: str | None) -> None:
        if source:
            self.stats.duplicate_sources[source] = self.stats.duplicate_sources.get(source, 0) + 1


def dedupe_exact(
    records: Iterable[dict[str, Any]],
    *,
    text_key: str = "text",
    source_key: str = "source",
    keep: KeepPolicy = "first",
    score_key: str = "quality_score",
    use_normalised: bool = True,
) -> tuple[list[dict[str, Any]], DedupStats]:
    """Deduplicate a record stream.

    With ``keep="best"`` the surviving copy is the one with the highest
    ``score_key``; ties keep the first.  That matters when the same article
    appears in both a clean source (Wikipedia) and a noisy one (CulturaX) - we
    want the clean copy in the corpus.
    """
    if keep == "first":
        deduper = ExactDeduplicator(use_normalised=use_normalised)
        kept: list[dict[str, Any]] = []
        for record in records:
            text = str(record.get(text_key, ""))
            if deduper.is_new(text, source=str(record.get(source_key, "")) or None):
                kept.append(record)
        return kept, deduper.stats

    # keep == "best": one pass, replacing the incumbent when a better copy shows up.
    stats = DedupStats()
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        stats.seen += 1
        text = str(record.get(text_key, ""))
        if not text.strip():
            stats.empty += 1
            continue
        key = normalised_hash(text) if use_normalised else content_hash(text)
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = record
            order.append(key)
            continue
        stats.normalised_duplicates += 1
        source = str(record.get(source_key, ""))
        if source:
            stats.duplicate_sources[source] = stats.duplicate_sources.get(source, 0) + 1
        if float(record.get(score_key, 0.0) or 0.0) > float(incumbent.get(score_key, 0.0) or 0.0):
            best[key] = record
    stats.kept = len(order)
    return [best[key] for key in order], stats


def iter_unique(
    texts: Iterable[str], *, key: Callable[[str], str] = normalised_hash
) -> Iterator[str]:
    """Yield only the first occurrence of each ``key(text)``."""
    seen: set[str] = set()
    for text in texts:
        digest = key(text)
        if digest in seen:
            continue
        seen.add(digest)
        yield text
