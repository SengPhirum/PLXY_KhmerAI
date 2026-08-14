"""Retrieval-augmented company knowledge layer.

Typical wiring (see ``server/rag_service.py`` for the production version)::

    from rag import LocalVectorStore, Retriever, build_embedder

    store = LocalVectorStore.load("data/index/ACTIVE/vectors")
    retriever = Retriever(store, build_embedder("ollama"))
    result = retriever.retrieve("តើម៉ូដែល QN-4500A ធានាប៉ុន្មានឆ្នាំ?")
"""

from rag.bm25 import BM25Index, BM25Params
from rag.chunking import ChunkingConfig, chunk_document, chunk_text, estimate_tokens
from rag.citations import Citation, GroundingReport, build_context_block, verify_grounding
from rag.embeddings import EmbeddingBackend, HashingEmbedder, OllamaEmbedder, build_embedder
from rag.hybrid_search import FusionConfig, fuse
from rag.reranker import HeuristicReranker, Reranker, build_reranker
from rag.retrieval import RetrievalConfig, Retriever, detect_conflicts
from rag.schemas import (
    Chunk,
    ConflictGroup,
    IndexManifest,
    RetrievalConfidence,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)
from rag.vector_store import LocalVectorStore, VectorStore, build_vector_store

__all__ = [
    "BM25Index",
    "BM25Params",
    "Chunk",
    "ChunkingConfig",
    "Citation",
    "ConflictGroup",
    "EmbeddingBackend",
    "FusionConfig",
    "GroundingReport",
    "HashingEmbedder",
    "HeuristicReranker",
    "IndexManifest",
    "LocalVectorStore",
    "OllamaEmbedder",
    "Reranker",
    "RetrievalConfidence",
    "RetrievalConfig",
    "RetrievalFilters",
    "RetrievalResult",
    "RetrievedChunk",
    "Retriever",
    "VectorStore",
    "build_context_block",
    "build_embedder",
    "build_reranker",
    "build_vector_store",
    "chunk_document",
    "chunk_text",
    "detect_conflicts",
    "estimate_tokens",
    "fuse",
    "verify_grounding",
]
