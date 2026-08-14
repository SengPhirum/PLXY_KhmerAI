"""Embedding backends behind one interface.

Three implementations, chosen by ``KHMERAI_EMBEDDING_BACKEND``:

``ollama``
    Production default on the Mac Studio.  Calls the already-resident Ollama
    runtime (``/api/embed``), so no second copy of a model competes for unified
    memory.  Model: ``qwen3-embedding:0.6b``.

``sentence_transformers``
    For benchmarking on a Linux/Colab host, or when a reranking-quality
    embedder is wanted.  Loads ``Qwen/Qwen3-Embedding-0.6B`` (or any configured
    alternative) through sentence-transformers.

``hashing``
    Deterministic, dependency-free, Khmer-aware hashing embedder.  It is *not*
    semantic and must never be used in production - it exists so that the whole
    RAG stack (ingestion, index build, hybrid search, reindex, API) is testable
    in CI with no model download.  ``EmbeddingBackend.is_semantic`` is False for
    it, and ``rag.ingestion`` refuses to build a production index with it unless
    ``allow_non_semantic=True``.

All backends L2-normalise their output, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from common.logging import get_logger
from preprocessing.khmer_script import tokenize_for_search
from preprocessing.unicode_normalization import normalize_text

log = get_logger(__name__)

__all__ = [
    "EmbeddingBackend",
    "HashingEmbedder",
    "OllamaEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
    "cosine_similarity",
]


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors."""
    return _l2_normalise(np.atleast_2d(a)) @ _l2_normalise(np.atleast_2d(b)).T


class EmbeddingBackend(ABC):
    """Common interface for every embedder."""

    name: str = "abstract"
    is_semantic: bool = True

    def __init__(self, model: str, dim: int) -> None:
        self.model = model
        self.dim = dim

    @abstractmethod
    def _embed(self, texts: list[str], *, is_query: bool) -> np.ndarray: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalise(self._embed([self._prepare(t) for t in texts], is_query=False))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalise(self._embed([self._prepare(text)], is_query=True))[0]

    def _prepare(self, text: str) -> str:
        """Normalise Khmer before embedding so form variants map to one vector."""
        return normalize_text(text)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "dim": self.dim,
            "semantic": self.is_semantic,
        }

    def health(self) -> tuple[bool, str]:
        """Cheap liveness probe used by ``GET /ready``."""
        try:
            vector = self.embed_query("សួស្តី")
        except Exception as exc:  # noqa: BLE001 - reported, not raised, to the health endpoint
            return False, f"{type(exc).__name__}: {exc}"
        if vector.shape[0] != self.dim:
            return False, f"dimension mismatch: got {vector.shape[0]}, expected {self.dim}"
        return True, "ok"


class HashingEmbedder(EmbeddingBackend):
    """Deterministic Khmer-aware hashing embedder for tests and CI.

    Uses the signed hashing trick over Khmer syllable n-grams and Latin words,
    with sub-linear term weighting.  Two texts sharing many n-grams get a high
    cosine similarity, which is enough to exercise ranking logic - but it has no
    semantic knowledge whatsoever.
    """

    name = "hashing"
    is_semantic = False

    def __init__(self, model: str = "hashing-ngram", dim: int = 256) -> None:
        super().__init__(model, dim)

    def _embed(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in tokenize_for_search(text, khmer_ngrams=(1, 2, 3)):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = struct.unpack(">Q", digest)[0] % self.dim
                sign = 1.0 if digest[0] & 1 else -1.0
                out[row, bucket] += sign * (1.0 + math.log(count))
        return out


class OllamaEmbedder(EmbeddingBackend):
    """Production embedder backed by the local Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        dim: int = 1024,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        batch_size: int = 16,
    ) -> None:
        super().__init__(model, dim)
        self.base_url = (
            base_url or os.environ.get("KHMERAI_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx  # noqa: PLC0415 - optional at import time

            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def _embed(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        vectors: list[list[float]] = []
        client = self._http()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = client.post("/api/embed", json={"model": self.model, "input": batch})
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings")
            if embeddings is None and "embedding" in payload:
                embeddings = [payload["embedding"]]
            if not embeddings:
                raise RuntimeError(f"Ollama returned no embeddings for model {self.model!r}")
            vectors.extend(embeddings)

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape[1] != self.dim:
            log.warning(
                "rag.embedding.dim_mismatch",
                extra={"configured": self.dim, "actual": int(matrix.shape[1]), "model": self.model},
            )
            self.dim = int(matrix.shape[1])
        return matrix

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class SentenceTransformerEmbedder(EmbeddingBackend):
    """sentence-transformers backend (benchmarking, Colab, GPU hosts)."""

    name = "sentence_transformers"

    # Qwen3-Embedding expects an instruction prefix on the *query* side only.
    QUERY_INSTRUCTION = (
        "Instruct: Given a Khmer customer-support question, retrieve the company "
        "document passage that answers it\nQuery: "
    )

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        dim: int = 1024,
        *,
        device: str | None = None,
        batch_size: int = 16,
        use_query_instruction: bool = True,
    ) -> None:
        super().__init__(model, dim)
        self.device = device
        self.batch_size = batch_size
        self.use_query_instruction = use_query_instruction
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "sentence-transformers is not installed. Either "
                    "`pip install sentence-transformers==5.7.0` or set "
                    "KHMERAI_EMBEDDING_BACKEND=ollama"
                ) from exc
            self._model = SentenceTransformer(self.model, device=self.device)
            actual = self._model.get_sentence_embedding_dimension()
            if actual and actual != self.dim:
                log.warning(
                    "rag.embedding.dim_mismatch",
                    extra={"configured": self.dim, "actual": actual, "model": self.model},
                )
                self.dim = int(actual)
        return self._model

    def _embed(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        model = self._load()
        prepared = (
            [self.QUERY_INSTRUCTION + t for t in texts]
            if is_query and self.use_query_instruction
            else texts
        )
        return np.asarray(
            model.encode(
                prepared,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


_BACKENDS: dict[str, type[EmbeddingBackend]] = {
    "hashing": HashingEmbedder,
    "ollama": OllamaEmbedder,
    "sentence_transformers": SentenceTransformerEmbedder,
}


def build_embedder(
    backend: str | None = None,
    model: str | None = None,
    dim: int | None = None,
    **kwargs: Any,
) -> EmbeddingBackend:
    """Construct the configured embedder.

    Resolution order: explicit argument, environment variable, then the safe
    default (``hashing``) so an unconfigured test run never tries to hit a
    network service.
    """
    name = (backend or os.environ.get("KHMERAI_EMBEDDING_BACKEND", "hashing")).lower()
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown embedding backend {name!r}; expected one of {', '.join(_BACKENDS)}"
        )

    resolved_model = model or os.environ.get("KHMERAI_EMBEDDING_MODEL")
    resolved_dim = dim or int(os.environ.get("KHMERAI_EMBEDDING_DIM", "0") or 0)

    init: dict[str, Any] = dict(kwargs)
    if resolved_model:
        init["model"] = resolved_model
    if resolved_dim:
        init["dim"] = resolved_dim
    return cls(**init)
