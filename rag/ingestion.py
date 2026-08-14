"""Build a versioned index from canonical company records.

An index build never touches the live index.  It creates a fresh directory
``data/index/<index_version>/`` containing the vector store, the manifest and a
copy of the chunk texts, and leaves activation to ``rag.reindex``.

Run it::

    python -m rag.ingestion \
        --config configs/rag/ingestion.yaml \
        --input data/interim/company_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from common.config import load_config
from common.hashing import sha256_file, sha256_text
from common.io import read_jsonl, write_json
from common.logging import get_logger
from common.paths import INDEX_ROOT
from common.versions import git_commit
from company_data.schema import CompanyDocument
from rag.chunking import ChunkingConfig, chunk_document
from rag.embeddings import EmbeddingBackend, build_embedder
from rag.schemas import Chunk, IndexManifest
from rag.vector_store import LocalVectorStore, VectorStore, build_vector_store

log = get_logger(__name__)

__all__ = ["IngestionSettings", "build_index", "load_records", "main"]


@dataclass(slots=True)
class IngestionSettings:
    chunk_size: int = 600
    chunk_overlap: int = 100
    min_chunk_size: int = 80
    embedding_backend: str = "ollama"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dim: int = 1024
    embed_batch_size: int = 16
    vector_backend: str = "local"
    index_root: Path = INDEX_ROOT
    allow_non_semantic_embedder: bool = False
    include_non_retrievable: bool = False
    as_of: date | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IngestionSettings:
        ingestion = config.get("ingestion", {}) or {}
        chunking = ingestion.get("chunking", {}) or {}
        embedding = config.get("embedding", ingestion.get("embedding", {})) or {}
        store = ingestion.get("vector_store", {}) or {}
        return cls(
            chunk_size=int(chunking.get("chunk_size", 600)),
            chunk_overlap=int(chunking.get("chunk_overlap", 100)),
            min_chunk_size=int(chunking.get("min_chunk_size", 80)),
            embedding_backend=str(embedding.get("backend", "ollama")),
            embedding_model=str(embedding.get("model", "qwen3-embedding:0.6b")),
            embedding_dim=int(embedding.get("dim", 1024)),
            embed_batch_size=int(embedding.get("batch_size", 16)),
            vector_backend=str(store.get("backend", "local")),
            index_root=Path(store.get("index_root", str(INDEX_ROOT))),
            allow_non_semantic_embedder=bool(
                ingestion.get("allow_non_semantic_embedder", False)
            ),
            include_non_retrievable=bool(ingestion.get("include_non_retrievable", False)),
        )

    def fingerprint(self) -> str:
        return sha256_text(
            json.dumps(
                {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "min_chunk_size": self.min_chunk_size,
                    "embedding_backend": self.embedding_backend,
                    "embedding_model": self.embedding_model,
                    "embedding_dim": self.embedding_dim,
                },
                sort_keys=True,
            )
        )[:16]


def new_index_version(prefix: str | None = None) -> str:
    """Date-sequenced version, e.g. ``2026-08-14.1`` (§37 knowledge_index)."""
    stamp = prefix or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing = sorted(p.name for p in INDEX_ROOT.glob(f"{stamp}.*") if p.is_dir())
    sequence = 1
    for name in existing:
        try:
            sequence = max(sequence, int(name.rsplit(".", 1)[-1]) + 1)
        except ValueError:
            continue
    return f"{stamp}.{sequence}"


def load_records(path: str | Path) -> list[CompanyDocument]:
    """Read canonical records produced by ``company_data.validate``."""
    documents: list[CompanyDocument] = []
    for row in read_jsonl(path):
        try:
            documents.append(CompanyDocument.model_validate(row))
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the build
            log.error(
                "rag.ingestion.invalid_record",
                extra={"error": str(exc), "id": str(row.get("document_id", "?"))},
            )
    return documents


def build_index(
    documents: list[CompanyDocument],
    *,
    settings: IngestionSettings | None = None,
    index_version: str | None = None,
    embedder: EmbeddingBackend | None = None,
    store: VectorStore | None = None,
    source_file: str = "",
) -> tuple[VectorStore, IndexManifest]:
    """Chunk, embed and persist a new index version.  Does not activate it."""
    cfg = settings or IngestionSettings()
    started = time.perf_counter()
    version = index_version or new_index_version()

    embed = embedder or build_embedder(
        cfg.embedding_backend, cfg.embedding_model, cfg.embedding_dim
    )
    if not embed.is_semantic and not cfg.allow_non_semantic_embedder:
        raise RuntimeError(
            f"embedding backend {embed.name!r} is not semantic and must not build a production "
            "index. Set KHMERAI_EMBEDDING_BACKEND=ollama (or sentence_transformers), or pass "
            "allow_non_semantic_embedder=true for tests."
        )

    eligible = [
        d
        for d in documents
        if cfg.include_non_retrievable or d.is_retrievable(cfg.as_of)
    ]
    skipped = len(documents) - len(eligible)
    if skipped:
        log.info("rag.ingestion.skipped_non_retrievable", extra={"count": skipped})

    chunking = ChunkingConfig(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        min_chunk_size=cfg.min_chunk_size,
    )

    chunks: list[Chunk] = []
    for document in eligible:
        metadata = document.to_index_metadata()
        produced = chunk_document(
            document_id=document.document_id,
            text=document.text,
            metadata=metadata,
            config=chunking,
        )
        for chunk in produced:
            chunk.metadata["token_estimate"] = chunk.token_estimate
        chunks.extend(produced)

    index_dir = Path(cfg.index_root) / version
    target = store or build_vector_store(
        cfg.vector_backend,
        path=index_dir / "vectors",
        collection=f"khmer_company_kb_{version.replace('.', '_').replace('-', '_')}",
        dim=embed.dim,
    )

    for start in range(0, len(chunks), cfg.embed_batch_size):
        batch = chunks[start : start + cfg.embed_batch_size]
        vectors = embed.embed_documents([c.contextual_text() for c in batch])
        target.add(batch, vectors)
    target.persist()

    manifest = IndexManifest(
        index_version=version,
        embedding_backend=embed.name,
        embedding_model=embed.model,
        embedding_dim=embed.dim,
        chunk_size_tokens=cfg.chunk_size,
        chunk_overlap_tokens=cfg.chunk_overlap,
        documents=len(eligible),
        chunks=len(chunks),
        source_file=str(source_file),
        source_sha256=sha256_file(str(source_file)) if source_file and Path(source_file).is_file() else "",
        code_commit=git_commit(),
        config_fingerprint=cfg.fingerprint(),
        build_seconds=round(time.perf_counter() - started, 2),
        notes=f"{skipped} document(s) excluded as non-retrievable",
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    write_json(index_dir / "manifest.json", manifest.model_dump(mode="json"))

    log.info(
        "rag.ingestion.built",
        extra={
            "index_version": version,
            "documents": manifest.documents,
            "chunks": manifest.chunks,
            "seconds": manifest.build_seconds,
        },
    )
    return target, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag.ingestion", description="Build a new company knowledge index"
    )
    parser.add_argument("--config", default="configs/rag/ingestion.yaml")
    parser.add_argument("--input", required=True, help="canonical records JSONL")
    parser.add_argument("--index-version", default=None)
    parser.add_argument(
        "--allow-non-semantic",
        action="store_true",
        help="permit the hashing embedder (tests only - never for production)",
    )
    args = parser.parse_args(argv)

    settings = IngestionSettings.from_config(load_config(args.config))
    if args.allow_non_semantic:
        settings.allow_non_semantic_embedder = True

    documents = load_records(args.input)
    if not documents:
        print(f"no valid records in {args.input}", file=sys.stderr)
        return 1

    store, manifest = build_index(
        documents, settings=settings, index_version=args.index_version, source_file=args.input
    )
    if isinstance(store, LocalVectorStore):
        store.persist()
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(
        f"\nIndex {manifest.index_version} built but NOT activated. "
        f"Activate with:\n    python -m rag.reindex --activate-version {manifest.index_version}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
