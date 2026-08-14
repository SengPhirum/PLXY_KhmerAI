"""BM25 over Khmer syllable n-grams.

Dense retrieval alone is weak exactly where customer support needs precision:
an exact model number (``QN-4500A``), an SKU, a price or a rarely-seen Khmer
technical term.  Embeddings smooth those into "some refrigerator", while BM25
matches them literally.  The hybrid fusion in ``rag/hybrid_search.py`` combines
the two.

Implemented here rather than pulled from ``rank_bm25`` for one substantive
reason: the tokenizer.  ``rank_bm25`` expects pre-tokenised input and every
off-the-shelf tokenizer splits Khmer on whitespace, which yields one enormous
"word" per sentence and destroys the ranking.  This implementation is built on
``preprocessing.khmer_script.tokenize_for_search``, which produces syllable
unigrams+bigrams for Khmer and whole tokens for Latin/numeric runs.

Uses BM25+ (the ``delta`` lower bound), which avoids BM25's known problem of
scoring very long documents below zero-information ones.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from preprocessing.khmer_script import tokenize_for_search

__all__ = ["BM25Index", "BM25Params", "default_tokenizer"]


def default_tokenizer(text: str) -> list[str]:
    """Khmer syllable unigrams + bigrams, plus whole Latin/numeric tokens."""
    return tokenize_for_search(text, khmer_ngrams=(1, 2))


@dataclass(slots=True)
class BM25Params:
    k1: float = 1.2      # term-frequency saturation
    b: float = 0.75      # length normalisation
    delta: float = 1.0   # BM25+ lower bound on the tf component


@dataclass(slots=True)
class BM25Index:
    """In-memory BM25+ index.

    >>> index = BM25Index()
    >>> index.add("a", "ការធានារយៈពេល ២៤ ខែ សម្រាប់ QN-4500A")
    >>> index.add("b", "សេវាកម្មដឹកជញ្ជូនទៅបណ្តាខេត្ត")
    >>> index.finalise()
    >>> index.search("QN-4500A")[0][0]
    'a'
    """

    params: BM25Params = field(default_factory=BM25Params)
    tokenizer: Callable[[str], list[str]] = default_tokenizer

    _doc_ids: list[str] = field(default_factory=list, init=False)
    _doc_lengths: list[int] = field(default_factory=list, init=False)
    _term_freqs: list[dict[str, int]] = field(default_factory=list, init=False)
    _postings: dict[str, list[int]] = field(default_factory=dict, init=False)
    _idf: dict[str, float] = field(default_factory=dict, init=False)
    _avg_length: float = field(default=0.0, init=False)
    _finalised: bool = field(default=False, init=False)

    # -- build --------------------------------------------------------------
    def add(self, doc_id: str, text: str) -> None:
        tokens = self.tokenizer(text)
        index = len(self._doc_ids)
        self._doc_ids.append(doc_id)
        self._doc_lengths.append(len(tokens))
        freqs = Counter(tokens)
        self._term_freqs.append(dict(freqs))
        for term in freqs:
            self._postings.setdefault(term, []).append(index)
        self._finalised = False

    def add_many(self, items: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in items:
            self.add(doc_id, text)

    def finalise(self) -> None:
        """Compute IDF and the average document length.  Idempotent."""
        n = len(self._doc_ids)
        self._avg_length = (sum(self._doc_lengths) / n) if n else 0.0
        self._idf = {}
        for term, postings in self._postings.items():
            df = len(postings)
            # Robertson-Sparck-Jones IDF with the standard +1 smoothing, which
            # keeps the value positive for terms appearing in most documents.
            self._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self._finalised = True

    # -- query --------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        allowed: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` sorted by descending BM25+ score."""
        if not self._finalised:
            self.finalise()
        if not self._doc_ids:
            return []

        query_terms = Counter(self.tokenizer(query))
        if not query_terms:
            return []

        k1, b, delta = self.params.k1, self.params.b, self.params.delta
        scores: dict[int, float] = {}
        for term, query_tf in query_terms.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_index in self._postings.get(term, ()):
                if allowed is not None and self._doc_ids[doc_index] not in allowed:
                    continue
                tf = self._term_freqs[doc_index].get(term, 0)
                length_norm = (
                    1 - b + b * (self._doc_lengths[doc_index] / self._avg_length)
                    if self._avg_length
                    else 1.0
                )
                component = (tf * (k1 + 1)) / (tf + k1 * length_norm) + delta
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * component * query_tf

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self._doc_ids[kv[0]]))
        return [(self._doc_ids[i], score) for i, score in ranked[:top_k]]

    # -- introspection ------------------------------------------------------
    def __len__(self) -> int:
        return len(self._doc_ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self._doc_ids),
            "vocabulary": self.vocabulary_size,
            "avg_tokens_per_document": round(self._avg_length, 2),
            "k1": self.params.k1,
            "b": self.params.b,
            "delta": self.params.delta,
        }
