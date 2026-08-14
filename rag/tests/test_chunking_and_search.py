"""Chunking, BM25, fusion and the embedding backends."""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.khmer_script import iter_clusters
from rag.bm25 import BM25Index
from rag.chunking import ChunkingConfig, chunk_document, chunk_text, estimate_tokens
from rag.embeddings import HashingEmbedder, build_embedder, cosine_similarity
from rag.hybrid_search import FusionConfig, fuse, reciprocal_rank_fusion, weighted_fusion

KHMER_DOC = (
    "# គោលការណ៍ធានា\n"
    "ទូរទឹកកកម៉ូដែល QN-4500A មានការធានារយៈពេល ២៤ ខែ ចាប់ពីថ្ងៃទិញ។ "
    "ការធានាគ្របដណ្តប់លើកំហុសផលិតកម្ម ប៉ុន្តែមិនរាប់បញ្ចូលការខូចខាតដោយសារការប្រើប្រាស់មិនត្រឹមត្រូវ។\n\n"
    "## ការទាមទារ\n"
    "អតិថិជនត្រូវបង្ហាញវិក្កយបត្រដើម។ សេវាកម្មជួសជុលមានផ្តល់ជូននៅគ្រប់សាខា។"
)


# --- token estimation -------------------------------------------------------
def test_khmer_costs_more_tokens_than_its_character_count_suggests() -> None:
    khmer = "ការធានារយៈពេលពីរឆ្នាំ"
    english = "two year warranty period"
    # Similar meaning, but Khmer tokenises far more finely.
    assert estimate_tokens(khmer) > estimate_tokens(english)


def test_estimate_tokens_is_monotonic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("ក") >= 1
    assert estimate_tokens(KHMER_DOC) > estimate_tokens(KHMER_DOC[:50])


# --- chunking ---------------------------------------------------------------
def test_chunks_never_split_a_khmer_cluster() -> None:
    long_text = "ការធានារយៈពេលពីរឆ្នាំសម្រាប់ផលិតផលអេឡិចត្រូនិកទាំងអស់។" * 40
    chunks = chunk_text(long_text, ChunkingConfig(chunk_size=120, chunk_overlap=20))
    assert len(chunks) > 1
    for body, _ in chunks:
        # A chunk that begins with a combining mark means a cluster was cut.
        assert body[0] not in "ាិីឹឺុូួើឿៀេែៃោៅំះ៉៊់៌៍៎៏័្"
        rebuilt = "".join(iter_clusters(body))
        assert rebuilt == body


def test_headings_become_the_heading_path() -> None:
    chunks = chunk_text(KHMER_DOC, ChunkingConfig(chunk_size=80, chunk_overlap=10))
    paths = [tuple(path) for _, path in chunks]
    assert ("គោលការណ៍ធានា",) in paths
    assert ("គោលការណ៍ធានា", "ការទាមទារ") in paths


def test_chunk_document_produces_stable_ids() -> None:
    first = chunk_document(document_id="doc1", text=KHMER_DOC)
    second = chunk_document(document_id="doc1", text=KHMER_DOC)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_chunk_metadata_is_carried_through() -> None:
    chunks = chunk_document(
        document_id="doc1", text=KHMER_DOC, metadata={"product_id": "QN-4500A", "status": "active"}
    )
    assert all(c.metadata["product_id"] == "QN-4500A" for c in chunks)
    assert all(c.product_id == "QN-4500A" for c in chunks)


def test_table_rows_stay_whole() -> None:
    table = "\n".join(f"QN-450{i} | 500 USD | 24 ខែ" for i in range(20))
    chunks = chunk_text(table, ChunkingConfig(chunk_size=60, chunk_overlap=0, min_chunk_size=10))
    for body, _ in chunks:
        for line in body.split("\n"):
            if "|" in line:
                assert line.count("|") == 2, f"table row was split: {line!r}"


def test_overlap_creates_shared_content() -> None:
    text = " ".join(f"ប្រយោគទី {i} អំពីការធានាផលិតផលរបស់យើង។" for i in range(30))
    no_overlap = chunk_text(text, ChunkingConfig(chunk_size=100, chunk_overlap=0))
    with_overlap = chunk_text(text, ChunkingConfig(chunk_size=100, chunk_overlap=40))
    assert len(with_overlap) >= len(no_overlap)


def test_invalid_chunk_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_document(document_id="d", text="   ") == []


# --- BM25 -------------------------------------------------------------------
def test_bm25_finds_exact_model_number() -> None:
    index = BM25Index()
    index.add("a", "ការធានារយៈពេល ២៤ ខែ សម្រាប់ម៉ូដែល QN-4500A")
    index.add("b", "ការធានារយៈពេល ១២ ខែ សម្រាប់ម៉ូដែល RF-22B")
    index.add("c", "សេវាកម្មដឹកជញ្ជូនទៅបណ្តាខេត្ត")
    index.finalise()
    assert index.search("QN-4500A")[0][0] == "a"
    assert index.search("RF-22B")[0][0] == "b"


def test_bm25_matches_khmer_without_spaces() -> None:
    index = BM25Index()
    index.add("a", "សេវាកម្មដឹកជញ្ជូនទៅបណ្តាខេត្តចំណាយពេលបីថ្ងៃ")
    index.add("b", "ការធានាផលិតផលអេឡិចត្រូនិក")
    index.finalise()
    results = index.search("ដឹកជញ្ជូន")
    assert results and results[0][0] == "a"


def test_bm25_respects_the_allowed_set() -> None:
    index = BM25Index()
    index.add("a", "ការធានា QN-4500A")
    index.add("b", "ការធានា QN-4500A ជំនាន់ចាស់")
    index.finalise()
    results = index.search("QN-4500A", allowed={"b"})
    assert [doc for doc, _ in results] == ["b"]


def test_bm25_unknown_query_returns_nothing() -> None:
    index = BM25Index()
    index.add("a", "ការធានា")
    index.finalise()
    assert index.search("zzzzqqqq") == []


def test_bm25_stats() -> None:
    index = BM25Index()
    index.add("a", "ការធានា QN-4500A")
    index.finalise()
    stats = index.stats()
    assert stats["documents"] == 1
    assert stats["vocabulary"] > 0


# --- fusion -----------------------------------------------------------------
def test_rrf_prefers_a_document_ranked_well_by_both() -> None:
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    lexical = [("c", 12.0), ("a", 10.0), ("d", 1.0)]
    fused = fuse(dense, lexical, FusionConfig(strategy="rrf", top_k=4))
    assert fused[0][0] == "a"
    assert {i for i, _ in fused} == {"a", "b", "c", "d"}


def test_weighted_fusion_normalises_incomparable_scales() -> None:
    """A raw BM25 score of 100 must not swamp a cosine similarity of 0.9.

    With equal weights and exactly opposite rankings the two retrievers cancel
    out.  Without min-max normalisation the unbounded lexical score would decide
    the ranking on its own, which is the bug this guards against.
    """
    fused = weighted_fusion(
        {"a": 0.9, "b": 0.1}, {"a": 1.0, "b": 100.0}, dense_weight=0.5, lexical_weight=0.5
    )
    assert all(0.0 <= v <= 1.0 for v in fused.values())
    assert fused["a"] == pytest.approx(fused["b"])

    # Shift the weights and the dense retriever's preference decides.
    dense_led = weighted_fusion(
        {"a": 0.9, "b": 0.1}, {"a": 1.0, "b": 100.0}, dense_weight=0.9, lexical_weight=0.1
    )
    assert dense_led["a"] > dense_led["b"]


def test_dense_only_and_lexical_only_strategies() -> None:
    dense = [("a", 0.9)]
    lexical = [("b", 5.0)]
    assert fuse(dense, lexical, FusionConfig(strategy="dense_only"))[0][0] == "a"
    assert fuse(dense, lexical, FusionConfig(strategy="lexical_only"))[0][0] == "b"


def test_fusion_is_deterministic_on_ties() -> None:
    fused = reciprocal_rank_fusion([(["x", "y"], 1.0), (["y", "x"], 1.0)])
    assert fused["x"] == fused["y"]
    ordered = fuse([("x", 1.0), ("y", 1.0)], [("y", 1.0), ("x", 1.0)], FusionConfig())
    assert [i for i, _ in ordered] == ["x", "y"]


def test_invalid_fusion_config() -> None:
    with pytest.raises(ValueError, match="weight"):
        FusionConfig(dense_weight=0.0, lexical_weight=0.0)


# --- embeddings -------------------------------------------------------------
def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dim=128)
    a = embedder.embed_query("ការធានារយៈពេលពីរឆ្នាំ")
    b = embedder.embed_query("ការធានារយៈពេលពីរឆ្នាំ")
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_hashing_embedder_similarity_tracks_overlap() -> None:
    embedder = HashingEmbedder(dim=512)
    query = embedder.embed_query("ការធានា QN-4500A")
    related = embedder.embed_documents(["ការធានារយៈពេល ២៤ ខែ សម្រាប់ QN-4500A"])[0]
    unrelated = embedder.embed_documents(["ម៉ោងធ្វើការរបស់សាខានៅភ្នំពេញ"])[0]
    assert float(query @ related) > float(query @ unrelated)


def test_hashing_embedder_is_marked_non_semantic() -> None:
    assert HashingEmbedder().is_semantic is False


def test_embed_documents_handles_empty_input() -> None:
    embedder = HashingEmbedder(dim=64)
    assert embedder.embed_documents([]).shape == (0, 64)


def test_cosine_similarity_helper() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    sims = cosine_similarity(a, b)[0]
    assert np.isclose(sims[0], 1.0)
    assert np.isclose(sims[1], 0.0)


def test_build_embedder_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown embedding backend"):
        build_embedder("does-not-exist")


def test_build_embedder_defaults_to_hashing_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KHMERAI_EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("KHMERAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("KHMERAI_EMBEDDING_DIM", raising=False)
    assert build_embedder().name == "hashing"


def test_embedder_health_probe() -> None:
    ok, message = HashingEmbedder(dim=64).health()
    assert ok and message == "ok"
