"""Hashing primitives for provenance, deduplication and cache keys.

Two families live here:

* **Content hashes** (``sha256_*``) - stable identifiers used in manifests and
  document IDs.  Cryptographic, collision-resistant, reproducible across runs.
* **Similarity hashes** (``MinHash``, ``simhash``) - used by
  ``preprocessing/near_dedup.py`` and by the train/test leakage checker.  These
  are deliberately *not* cryptographic.

The MinHash implementation is written against the standard library so the
preprocessing pipeline stays installable from ``requirements/base.txt``.  When
``datasketch`` is present the pipeline can use it instead (it is faster for very
large corpora) - the shingle definition is identical, so results are comparable.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence

__all__ = [
    "LSHIndex",
    "MinHash",
    "hamming_distance",
    "jaccard_estimate",
    "minhash_signature",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "shingles",
    "simhash",
    "stable_id",
]

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:  # noqa: PTH123 - streaming read of a large file
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: str, length: int = 16) -> str:
    """Deterministic short identifier derived from ``parts``.

    Used for ``document_id``/``chunk_id`` so that re-ingesting an unchanged
    document produces an unchanged ID (which in turn keeps the vector index
    diffable and makes reindex idempotent).
    """
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def shingles(tokens: Sequence[str], width: int = 5) -> set[str]:
    """Overlapping token n-grams ("shingles") used for near-duplicate detection."""
    if width <= 0:
        raise ValueError("width must be positive")
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def _permutations(num_perm: int, seed: int) -> list[tuple[int, int]]:
    """Deterministic (a, b) pairs for the universal hash family h(x) = a*x + b."""
    rng_state = hashlib.sha256(f"khmerai-minhash-{seed}".encode()).digest()
    out: list[tuple[int, int]] = []
    counter = 0
    while len(out) < num_perm:
        rng_state = hashlib.sha256(rng_state + struct.pack(">I", counter)).digest()
        counter += 1
        a = int.from_bytes(rng_state[:8], "big") % _MERSENNE_PRIME
        b = int.from_bytes(rng_state[8:16], "big") % _MERSENNE_PRIME
        if a == 0:
            continue
        out.append((a, b))
    return out


class MinHash:
    """Fixed-permutation MinHash sketch.

    >>> a = MinHash(num_perm=64); a.update_batch({"a b", "b c", "c d"})
    >>> b = MinHash(num_perm=64); b.update_batch({"a b", "b c", "c e"})
    >>> 0.0 < a.jaccard(b) < 1.0
    True
    """

    __slots__ = ("_params", "_sig", "num_perm", "seed")

    def __init__(self, num_perm: int = 128, seed: int = 20260814) -> None:
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        self.num_perm = num_perm
        self.seed = seed
        self._params = _permutations(num_perm, seed)
        self._sig = [_MAX_HASH] * num_perm

    def update(self, item: str) -> None:
        h = int.from_bytes(hashlib.sha1(item.encode("utf-8")).digest()[:8], "big")
        sig = self._sig
        for i, (a, b) in enumerate(self._params):
            value = ((a * h + b) % _MERSENNE_PRIME) & _MAX_HASH
            if value < sig[i]:
                sig[i] = value

    def update_batch(self, items: Iterable[str]) -> None:
        for item in items:
            self.update(item)

    @property
    def signature(self) -> tuple[int, ...]:
        return tuple(self._sig)

    def jaccard(self, other: MinHash) -> float:
        if self.num_perm != other.num_perm or self.seed != other.seed:
            raise ValueError("MinHash sketches must share num_perm and seed")
        matches = sum(1 for x, y in zip(self._sig, other._sig, strict=True) if x == y)
        return matches / self.num_perm


def minhash_signature(
    tokens: Sequence[str], *, num_perm: int = 128, width: int = 5, seed: int = 20260814
) -> tuple[int, ...]:
    sketch = MinHash(num_perm=num_perm, seed=seed)
    sketch.update_batch(shingles(tokens, width=width))
    return sketch.signature


def jaccard_estimate(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != len(b):
        raise ValueError("signatures must have the same length")
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


class LSHIndex:
    """Banded locality-sensitive hashing over MinHash signatures.

    ``threshold`` is approximated by choosing the number of bands so that the
    S-curve inflection ``(1/bands) ** (1/rows)`` sits near the requested value.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.85) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.num_perm = num_perm
        self.threshold = threshold
        self.bands, self.rows = self._optimal_bands(num_perm, threshold)
        self._buckets: dict[tuple[int, int], list[str]] = {}
        self._signatures: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _optimal_bands(num_perm: int, threshold: float) -> tuple[int, int]:
        best = (num_perm, 1)
        best_error = float("inf")
        for bands in range(1, num_perm + 1):
            if num_perm % bands:
                continue
            rows = num_perm // bands
            inflection = (1.0 / bands) ** (1.0 / rows)
            error = abs(inflection - threshold)
            if error < best_error:
                best_error = error
                best = (bands, rows)
        return best

    def __len__(self) -> int:
        return len(self._signatures)

    def add(self, key: str, signature: Sequence[int]) -> None:
        if len(signature) != self.num_perm:
            raise ValueError(f"signature length {len(signature)} != num_perm {self.num_perm}")
        sig = tuple(signature)
        self._signatures[key] = sig
        for band in range(self.bands):
            chunk = sig[band * self.rows : (band + 1) * self.rows]
            bucket = (band, hash(chunk))
            self._buckets.setdefault(bucket, []).append(key)

    def query(self, signature: Sequence[int], *, exclude: str | None = None) -> list[str]:
        """Candidate keys whose *estimated* Jaccard clears ``threshold``."""
        sig = tuple(signature)
        candidates: set[str] = set()
        for band in range(self.bands):
            chunk = sig[band * self.rows : (band + 1) * self.rows]
            candidates.update(self._buckets.get((band, hash(chunk)), ()))
        candidates.discard(exclude)  # type: ignore[arg-type]
        return [
            key
            for key in sorted(candidates)
            if jaccard_estimate(self._signatures[key], sig) >= self.threshold
        ]


def simhash(tokens: Iterable[str], bits: int = 64) -> int:
    """SimHash fingerprint - cheap complement to MinHash for short documents."""
    vector = [0] * bits
    seen = False
    for token in tokens:
        seen = True
        h = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest(), "big")
        for i in range(bits):
            vector[i] += 1 if (h >> i) & 1 else -1
    if not seen:
        return 0
    out = 0
    for i in range(bits):
        if vector[i] > 0:
            out |= 1 << i
    return out


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
