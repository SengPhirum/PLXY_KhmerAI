"""Vector store abstraction with two backends.

Choice of vector database (Phase 10 asked for a justified choice)
-----------------------------------------------------------------
The workload is: a few thousand to a few hundred thousand chunks, single-node,
on-premise, read-heavy, with **metadata filtering that must be exact** (status,
effective date, confidentiality) and an **atomic build-then-swap** update model.

* ``LocalVectorStore`` (default) - a NumPy flat index persisted as ``.npy`` plus
  a JSONL sidecar.  At this corpus size an exact brute-force scan over
  normalised vectors is sub-millisecond per query on the Mac Studio, so an ANN
  index would add operational complexity and recall loss for no latency gain.
  It has no server to run, no daemon to supervise, and a build is a directory -
  which makes the atomic swap and the rollback in Phase 21/38 trivial.
* ``QdrantVectorStore`` - used when the corpus outgrows a flat scan or when
  multiple services need to share the index.  Qdrant is chosen over Chroma and
  raw FAISS because it supports server-side payload filtering (so the status /
  expiry filter runs *inside* the search rather than as post-filtering, which is
  what breaks top-k correctness) and it has a first-class collection-alias
  mechanism that gives the same atomic swap semantics as the local store.

Both implement :class:`VectorStore`, and ``rag.retrieval`` never knows which is
in use.
"""

from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from common.io import atomic_write_bytes, read_json, write_json
from common.logging import get_logger
from rag.schemas import Chunk

log = get_logger(__name__)

__all__ = ["VectorStore", "LocalVectorStore", "QdrantVectorStore", "build_vector_store"]

_VECTORS_FILE = "vectors.npy"
_CHUNKS_FILE = "chunks.jsonl"
_META_FILE = "store.json"


class VectorStore(ABC):
    """Persisted chunk + vector store with metadata filtering."""

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def persist(self) -> None: ...

    @abstractmethod
    def all_chunks(self) -> list[Chunk]: ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        return


class LocalVectorStore(VectorStore):
    """Exact flat index on disk.  Default backend."""

    def __init__(self, path: str | os.PathLike[str], dim: int) -> None:
        self.path = Path(path)
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._pending: list[np.ndarray] = []

    # -- persistence --------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> LocalVectorStore:
        root = Path(path)
        meta_path = root / _META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"no vector store at {root} (missing {_META_FILE})")
        meta = read_json(meta_path)
        store = cls(root, int(meta["dim"]))

        vectors_path = root / _VECTORS_FILE
        store._vectors = (
            np.load(vectors_path).astype(np.float32)
            if vectors_path.is_file()
            else np.zeros((0, store.dim), dtype=np.float32)
        )
        chunks_path = root / _CHUNKS_FILE
        if chunks_path.is_file():
            with chunks_path.open("r", encoding="utf-8") as handle:
                store._chunks = [
                    Chunk.model_validate(json.loads(line)) for line in handle if line.strip()
                ]
        if len(store._chunks) != store._vectors.shape[0]:
            raise ValueError(
                f"corrupt index at {root}: {len(store._chunks)} chunks vs "
                f"{store._vectors.shape[0]} vectors"
            )
        return store

    def persist(self) -> None:
        self._flush()
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / _VECTORS_FILE, self._vectors)
        payload = "\n".join(c.model_dump_json() for c in self._chunks)
        atomic_write_bytes(self.path / _CHUNKS_FILE, (payload + "\n").encode("utf-8"))
        write_json(
            self.path / _META_FILE,
            {"dim": self.dim, "count": len(self._chunks), "backend": "local"},
        )

    def destroy(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    # -- writes -------------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError(
                f"expected {len(chunks)} vectors of dim {self.dim}, got shape {matrix.shape}"
            )
        if matrix.shape[1] != self.dim:
            raise ValueError(f"vector dim {matrix.shape[1]} != store dim {self.dim}")
        self._chunks.extend(chunks)
        self._pending.append(matrix)

    def _flush(self) -> None:
        if not self._pending:
            return
        stacked = np.vstack([self._vectors, *self._pending]) if self._vectors.size else np.vstack(self._pending)
        self._vectors = stacked.astype(np.float32)
        self._pending.clear()

    # -- reads --------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[tuple[Chunk, float]]:
        self._flush()
        if self._vectors.shape[0] == 0 or top_k <= 0:
            return []

        # Filter first so top-k is computed over the eligible set only.  Doing it
        # the other way round silently returns fewer than k results (or none)
        # whenever the top matches happen to be expired or internal documents.
        if predicate is None:
            candidate_idx = np.arange(len(self._chunks))
        else:
            candidate_idx = np.fromiter(
                (i for i, c in enumerate(self._chunks) if predicate(c.metadata)),
                dtype=np.int64,
            )
        if candidate_idx.size == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self._vectors[candidate_idx] @ query
        k = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[int(candidate_idx[i])], float(scores[i])) for i in top]


class QdrantVectorStore(VectorStore):
    """Qdrant-backed store with server-side payload filtering.

    Used when ``KHMERAI_VECTOR_BACKEND=qdrant``.  The collection name carries the
    index version; ``rag.reindex`` builds into a new collection and then moves
    the ``khmer_company_kb`` alias, which is the atomic swap.
    """

    def __init__(
        self,
        collection: str,
        dim: int,
        *,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float = 15.0,
        recreate: bool = False,
    ) -> None:
        self.collection = collection
        self.dim = dim
        self.url = url or os.environ.get("KHMERAI_QDRANT_URL", "http://127.0.0.1:6333")
        self.api_key = api_key or os.environ.get("KHMERAI_QDRANT_API_KEY") or None
        self.timeout = timeout
        self._client: Any = None
        self._models: Any = None
        self._count = 0
        if recreate:
            self._ensure_collection(recreate=True)

    def _connect(self) -> tuple[Any, Any]:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient, models  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "qdrant-client is not installed. `pip install -r requirements/rag.txt` "
                    "or set KHMERAI_VECTOR_BACKEND=local"
                ) from exc
            self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)
            self._models = models
        return self._client, self._models

    def _ensure_collection(self, *, recreate: bool = False) -> None:
        client, models = self._connect()
        exists = client.collection_exists(self.collection)
        if exists and recreate:
            client.delete_collection(self.collection)
            exists = False
        if not exists:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE
                ),
            )
            # Indexed payload fields - these are the retrieval filters.
            for field, schema in (
                ("product_id", "keyword"),
                ("service_id", "keyword"),
                ("category", "keyword"),
                ("status", "keyword"),
                ("language", "keyword"),
                ("confidentiality", "keyword"),
                ("version", "keyword"),
                ("expiration_date", "keyword"),
            ):
                client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=schema
                )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        client, models = self._connect()
        self._ensure_collection()
        matrix = np.asarray(vectors, dtype=np.float32)
        points = [
            models.PointStruct(
                id=abs(hash(chunk.chunk_id)) % (1 << 63),
                vector=matrix[i].tolist(),
                payload={
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "ordinal": chunk.ordinal,
                    "heading_path": chunk.heading_path,
                    "token_estimate": chunk.token_estimate,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        client.upsert(collection_name=self.collection, points=points, wait=True)
        self._count += len(points)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        query_filter: Any = None,
    ) -> list[tuple[Chunk, float]]:
        client, _ = self._connect()
        # Over-fetch when a Python predicate is supplied, then apply it, so the
        # caller still receives up to top_k results after filtering.
        limit = top_k * 4 if predicate is not None and query_filter is None else top_k
        hits = client.query_points(
            collection_name=self.collection,
            query=np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist(),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points

        out: list[tuple[Chunk, float]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            metadata = {
                k: v
                for k, v in payload.items()
                if k not in ("text", "chunk_id", "document_id", "ordinal", "heading_path", "token_estimate")
            }
            if predicate is not None and not predicate(metadata):
                continue
            out.append(
                (
                    Chunk(
                        chunk_id=str(payload.get("chunk_id", hit.id)),
                        document_id=str(payload.get("document_id", "")),
                        text=str(payload.get("text", "")),
                        ordinal=int(payload.get("ordinal", 0)),
                        token_estimate=int(payload.get("token_estimate", 0)),
                        heading_path=list(payload.get("heading_path", []) or []),
                        metadata=metadata,
                    ),
                    float(hit.score),
                )
            )
            if len(out) >= top_k:
                break
        return out

    def all_chunks(self) -> list[Chunk]:
        """Full scan - used only to build the BM25 index at startup."""
        client, _ = self._connect()
        out: list[Chunk] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                out.append(
                    Chunk(
                        chunk_id=str(payload.get("chunk_id", point.id)),
                        document_id=str(payload.get("document_id", "")),
                        text=str(payload.get("text", "")),
                        ordinal=int(payload.get("ordinal", 0)),
                        token_estimate=int(payload.get("token_estimate", 0)),
                        heading_path=list(payload.get("heading_path", []) or []),
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k
                            not in ("text", "chunk_id", "document_id", "ordinal", "heading_path", "token_estimate")
                        },
                    )
                )
            if offset is None:
                break
        return out

    def set_alias(self, alias: str) -> None:
        """Point ``alias`` at this collection - the atomic production swap."""
        client, models = self._connect()
        client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=self.collection, alias_name=alias
                    )
                )
            ]
        )

    def persist(self) -> None:
        """Qdrant persists on write; this only records the count."""
        log.info(
            "rag.qdrant.persisted",
            extra={"collection": self.collection, "points": self._count},
        )

    @property
    def size(self) -> int:
        if self._count:
            return self._count
        client, _ = self._connect()
        return int(client.count(self.collection, exact=True).count)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def build_vector_store(
    backend: str | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    collection: str | None = None,
    dim: int = 1024,
    load_existing: bool = False,
    **kwargs: Any,
) -> VectorStore:
    """Construct the configured vector store."""
    name = (backend or os.environ.get("KHMERAI_VECTOR_BACKEND", "local")).lower()
    if name == "local":
        if path is None:
            raise ValueError("the local vector store requires `path`")
        if load_existing:
            return LocalVectorStore.load(path)
        return LocalVectorStore(path, dim)
    if name == "qdrant":
        return QdrantVectorStore(
            collection or os.environ.get("KHMERAI_QDRANT_COLLECTION", "khmer_company_kb"),
            dim,
            **kwargs,
        )
    raise ValueError(f"unknown vector backend {name!r}; expected 'local' or 'qdrant'")
