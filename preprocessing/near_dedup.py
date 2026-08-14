"""Near-duplicate detection (MinHash + banded LSH) and train/test leakage checks.

Exact deduplication misses the dominant duplication pattern in web corpora:
the same article republished with a changed headline, an added footer or a
reordered paragraph.  MinHash over Khmer syllable shingles catches those.

Shingling for Khmer
-------------------
English pipelines shingle over whitespace-separated words.  Khmer has no
spaces, so shingles are built over *orthographic syllable clusters* from
``preprocessing.khmer_script``.  A width of 5 clusters is roughly a 2-3 word
window in Khmer, which is the same granularity a word-5-gram gives in English.

Leakage
-------
``LeakageChecker`` is the Phase 7 guard: it holds the sealed evaluation/test set
and rejects any training record that is a near duplicate of it.  This runs
*before* the train/validation/test split so a paraphrase cannot cross splits.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from common.hashing import LSHIndex, MinHash, jaccard_estimate, shingles
from preprocessing.khmer_script import iter_clusters, is_khmer_char
from preprocessing.unicode_normalization import normalize_for_hashing

__all__ = [
    "NearDedupConfig",
    "NearDuplicateResult",
    "NearDeduplicator",
    "LeakageChecker",
    "khmer_shingle_tokens",
]


def khmer_shingle_tokens(text: str) -> list[str]:
    """Units for shingling: Khmer syllables plus lower-cased Latin/number runs."""
    normalised = normalize_for_hashing(text)
    units: list[str] = []
    buffer: list[str] = []
    for cluster in iter_clusters(normalised):
        if is_khmer_char(cluster[0]):
            if buffer:
                units.append("".join(buffer))
                buffer = []
            units.append(cluster)
        elif cluster.isalnum():
            buffer.append(cluster)
        elif buffer:
            units.append("".join(buffer))
            buffer = []
    if buffer:
        units.append("".join(buffer))
    return units


@dataclass(slots=True)
class NearDedupConfig:
    num_perm: int = 128
    shingle_width: int = 5
    threshold: float = 0.85
    seed: int = 20260814
    min_units: int = 8          # below this, MinHash is unreliable; fall back to exact

    @classmethod
    def for_sft(cls) -> NearDedupConfig:
        """Support turns are short - narrower shingles, stricter threshold."""
        return cls(shingle_width=3, threshold=0.80, min_units=4)


@dataclass(slots=True)
class NearDuplicateResult:
    is_duplicate: bool
    matched_key: str | None = None
    similarity: float = 0.0


@dataclass(slots=True)
class NearDedupStats:
    seen: int = 0
    kept: int = 0
    near_duplicates: int = 0
    too_short_for_minhash: int = 0
    matched_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "near_duplicates": self.near_duplicates,
            "too_short_for_minhash": self.too_short_for_minhash,
            "near_duplicate_rate": round(self.near_duplicates / self.seen, 4) if self.seen else 0.0,
            "sample_pairs": self.matched_pairs[:25],
        }


def _signature(units: Sequence[str], config: NearDedupConfig) -> tuple[int, ...]:
    sketch = MinHash(num_perm=config.num_perm, seed=config.seed)
    sketch.update_batch(shingles(units, width=config.shingle_width))
    return sketch.signature


class NearDeduplicator:
    """Streaming near-duplicate filter backed by banded LSH.

    >>> d = NearDeduplicator(NearDedupConfig(threshold=0.7, min_units=2))
    >>> d.add("a", "ការធានារយៈពេលពីរឆ្នាំសម្រាប់ម៉ូដែលនេះ").is_duplicate
    False
    >>> d.add("b", "ការធានារយៈពេលពីរឆ្នាំសម្រាប់ម៉ូដែលនេះ។").is_duplicate
    True
    """

    def __init__(self, config: NearDedupConfig | None = None) -> None:
        self.config = config or NearDedupConfig()
        self._index = LSHIndex(num_perm=self.config.num_perm, threshold=self.config.threshold)
        self._short_exact: dict[str, str] = {}
        self.stats = NearDedupStats()

    def __len__(self) -> int:
        return len(self._index) + len(self._short_exact)

    def check(self, text: str) -> NearDuplicateResult:
        """Test for a near duplicate without inserting."""
        units = khmer_shingle_tokens(text)
        if len(units) < self.config.min_units:
            key = "".join(units)
            match = self._short_exact.get(key)
            return NearDuplicateResult(match is not None, match, 1.0 if match else 0.0)

        signature = _signature(units, self.config)
        matches = self._index.query(signature)
        if not matches:
            return NearDuplicateResult(False)
        best = matches[0]
        similarity = jaccard_estimate(self._index._signatures[best], signature)  # noqa: SLF001
        return NearDuplicateResult(True, best, similarity)

    def add(self, key: str, text: str) -> NearDuplicateResult:
        """Check then insert.  Returns the check result for the *pre-insert* state."""
        self.stats.seen += 1
        units = khmer_shingle_tokens(text)

        if len(units) < self.config.min_units:
            self.stats.too_short_for_minhash += 1
            short_key = "".join(units)
            existing = self._short_exact.get(short_key)
            if existing is not None:
                self.stats.near_duplicates += 1
                self.stats.matched_pairs.append((key, existing, 1.0))
                return NearDuplicateResult(True, existing, 1.0)
            self._short_exact[short_key] = key
            self.stats.kept += 1
            return NearDuplicateResult(False)

        signature = _signature(units, self.config)
        matches = self._index.query(signature, exclude=key)
        if matches:
            best = matches[0]
            similarity = jaccard_estimate(self._index._signatures[best], signature)  # noqa: SLF001
            self.stats.near_duplicates += 1
            self.stats.matched_pairs.append((key, best, round(similarity, 4)))
            return NearDuplicateResult(True, best, similarity)

        self._index.add(key, signature)
        self.stats.kept += 1
        return NearDuplicateResult(False)

    def filter_records(
        self, records: Iterable[dict[str, Any]], *, text_key: str = "text", id_key: str = "id"
    ) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            key = str(record.get(id_key) or index)
            if not self.add(key, str(record.get(text_key, ""))).is_duplicate:
                kept.append(record)
        return kept


class LeakageChecker:
    """Reject training records that overlap the sealed evaluation set.

    Usage (Phase 7)::

        checker = LeakageChecker.from_texts(test_texts)
        clean_train = [r for r in train if not checker.leaks(r["text"])]
        report = checker.report()
    """

    def __init__(self, config: NearDedupConfig | None = None) -> None:
        self.config = config or NearDedupConfig(threshold=0.75)
        self._index = LSHIndex(num_perm=self.config.num_perm, threshold=self.config.threshold)
        self._exact: dict[str, str] = {}
        self.hits: list[tuple[str, str, float]] = []
        self.checked = 0

    @classmethod
    def from_texts(
        cls, texts: Iterable[str], config: NearDedupConfig | None = None
    ) -> LeakageChecker:
        checker = cls(config)
        for index, text in enumerate(texts):
            checker.protect(f"eval:{index}", text)
        return checker

    def protect(self, key: str, text: str) -> None:
        """Add an evaluation record that training data must not duplicate."""
        units = khmer_shingle_tokens(text)
        self._exact.setdefault("".join(units), key)
        if len(units) >= self.config.min_units:
            self._index.add(key, _signature(units, self.config))

    def leaks(self, text: str) -> bool:
        """True when ``text`` duplicates a protected evaluation record."""
        self.checked += 1
        units = khmer_shingle_tokens(text)
        exact_key = self._exact.get("".join(units))
        if exact_key is not None:
            self.hits.append((text[:120], exact_key, 1.0))
            return True
        if len(units) < self.config.min_units:
            return False
        signature = _signature(units, self.config)
        matches = self._index.query(signature)
        if not matches:
            return False
        best = matches[0]
        similarity = jaccard_estimate(self._index._signatures[best], signature)  # noqa: SLF001
        self.hits.append((text[:120], best, round(similarity, 4)))
        return True

    def report(self) -> dict[str, Any]:
        return {
            "protected_records": len(self._exact),
            "checked": self.checked,
            "leaks_found": len(self.hits),
            "leak_rate": round(len(self.hits) / self.checked, 6) if self.checked else 0.0,
            "threshold": self.config.threshold,
            "examples": [
                {"train_text": t, "matched_eval_key": k, "similarity": s}
                for t, k, s in self.hits[:20]
            ],
        }
