"""RAG lifecycle inside the API process.

Responsibilities:

* load the index that ``data/index/ACTIVE`` points at, at startup;
* run retrieval off the event loop (the flat vector scan and BM25 are CPU-bound,
  so they go to a thread executor rather than blocking every other request);
* enforce a retrieval timeout - a slow retriever must degrade to "no context"
  rather than holding a generation slot;
* hot-swap to a new index after a reindex, without restarting the process and
  without any request ever seeing a half-loaded index.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

from common.io import read_json
from common.logging import get_logger
from rag.embeddings import EmbeddingBackend, build_embedder
from rag.reranker import build_reranker
from rag.retrieval import RetrievalConfig, Retriever
from rag.schemas import RetrievalConfidence, RetrievalFilters, RetrievalResult
from rag.vector_store import LocalVectorStore, VectorStore, build_vector_store
from server.config import Settings

log = get_logger(__name__)

__all__ = ["RagService"]


class RagService:
    """Owns the retriever and the index version currently being served."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._retriever: Retriever | None = None
        self._embedder: EmbeddingBackend | None = None
        self._store: VectorStore | None = None
        self._index_version = "unknown"
        self._loaded_at = 0.0
        self._swap_lock = threading.Lock()
        self._config = RetrievalConfig.from_dict(settings.rag_policy())

    # -- state --------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._retriever is not None and (self._store is not None and self._store.size > 0)

    @property
    def index_version(self) -> str:
        return self._index_version

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "index_version": self._index_version,
            "chunks": self._store.size if self._store else 0,
            "embedding_backend": self._embedder.name if self._embedder else "",
            "embedding_model": self._embedder.model if self._embedder else "",
            "loaded_at": self._loaded_at,
            "top_k": self._config.top_k,
            "strategy": self._config.fusion.strategy,
        }

    # -- loading ------------------------------------------------------------
    def _resolve_active(self) -> tuple[Path, str]:
        """Resolve ``data/index/ACTIVE`` to a concrete version directory."""
        root = self.settings.index_root_path
        pointer = root / "ACTIVE"
        version = "unknown"

        if pointer.is_symlink():
            target = Path(str(pointer.readlink()))
            version = target.name
            directory = target if target.is_absolute() else root / target
        elif pointer.is_file():
            version = pointer.read_text(encoding="utf-8").strip()
            directory = root / version
        elif pointer.is_dir():
            directory = pointer
        else:
            raise FileNotFoundError(
                f"no active knowledge index at {pointer}. Build and activate one with:\n"
                "    make rag-reindex"
            )

        manifest = directory / "manifest.json"
        if manifest.is_file():
            try:
                version = str(read_json(manifest).get("index_version", version))
            except Exception:  # noqa: BLE001 - a damaged manifest is not fatal
                pass
        return directory, version

    def load(self) -> None:
        """Load the active index.  Safe to call again to reload in place."""
        started = time.perf_counter()
        directory, version = self._resolve_active()

        embedder = self._embedder or build_embedder(
            self.settings.embedding_backend,
            self.settings.embedding_model,
            self.settings.embedding_dim,
        )
        store = (
            LocalVectorStore.load(directory / "vectors")
            if self.settings.vector_backend == "local"
            else build_vector_store(
                "qdrant",
                collection=self.settings.qdrant_collection,
                dim=embedder.dim,
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
            )
        )
        retriever = Retriever(
            store,
            embedder,
            config=self._config,
            reranker=build_reranker("cross_encoder" if self._config.use_reranker else "heuristic")
            if self._config.use_reranker
            else None,
            index_version=version,
        )

        # Publish the fully-built retriever atomically: a concurrent request
        # either sees the old one or the new one, never a partially wired state.
        with self._swap_lock:
            self._embedder = embedder
            self._store = store
            self._retriever = retriever
            self._index_version = version
            self._loaded_at = time.time()

        log.info(
            "rag_service.loaded",
            extra={
                "index_version": version,
                "chunks": store.size,
                "embedding_backend": embedder.name,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

    def try_load(self) -> bool:
        """Load, converting a missing index into a warning instead of a crash.

        The API must start even with no index: ``/health`` stays green, ``/ready``
        reports the RAG component as unhealthy, and general (non-company)
        questions are still answerable.
        """
        try:
            self.load()
            return True
        except FileNotFoundError as exc:
            log.warning("rag_service.no_index", extra={"detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 - never block startup on RAG
            log.error("rag_service.load_failed", extra={"error": f"{type(exc).__name__}: {exc}"})
        return False

    def reload(self) -> bool:
        """Hot-swap to whatever ACTIVE now points at (called after a reindex)."""
        return self.try_load()

    # -- query --------------------------------------------------------------
    async def search(
        self, query: str, *, filters: RetrievalFilters | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        """Retrieve, off the event loop, with a hard timeout."""
        retriever = self._retriever
        if retriever is None:
            return RetrievalResult(query=query, confidence=RetrievalConfidence.NONE)

        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: retriever.retrieve(query, filters=filters, top_k=top_k)
                ),
                timeout=self.settings.retrieval_timeout_s,
            )
        except TimeoutError:
            log.warning(
                "rag_service.timeout",
                extra={"timeout_s": self.settings.retrieval_timeout_s, "query_chars": len(query)},
            )
            return RetrievalResult(
                query=query,
                confidence=RetrievalConfidence.NONE,
                index_version=self._index_version,
                strategy="timeout",
            )
        except Exception as exc:  # noqa: BLE001 - retrieval must never 500 a chat turn
            log.error("rag_service.search_failed", extra={"error": f"{type(exc).__name__}: {exc}"})
            return RetrievalResult(
                query=query, confidence=RetrievalConfidence.NONE, strategy="error"
            )

    def health(self) -> tuple[bool, str]:
        if self._retriever is None:
            return False, "no index loaded"
        if self._store is None or self._store.size == 0:
            return False, "index is empty"
        if self._embedder is not None:
            ok, detail = self._embedder.health()
            if not ok:
                return False, f"embedder: {detail}"
        return True, f"index {self._index_version}, {self._store.size} chunks"

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
        self._retriever = None
        self._store = None
