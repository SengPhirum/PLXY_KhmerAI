"""A small, fully-Khmer company knowledge base used across the RAG tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from company_data.schema import (
    CompanyDocument,
    Confidentiality,
    DocumentStatus,
    ValidationStatus,
)
from rag.embeddings import HashingEmbedder
from rag.ingestion import IngestionSettings, build_index
from rag.retrieval import RetrievalConfig, Retriever
from rag.vector_store import LocalVectorStore


def _document(
    *,
    title: str,
    text: str,
    product_id: str | None = None,
    service_id: str | None = None,
    category: str = "general",
    version: str = "1.0",
    effective: str = "2026-01-01",
    expires: str | None = None,
    status: DocumentStatus = DocumentStatus.ACTIVE,
    confidentiality: Confidentiality = Confidentiality.PUBLIC,
) -> CompanyDocument:
    from datetime import date

    return CompanyDocument(
        document_title=title,
        text=text,
        product_id=product_id,
        service_id=service_id,
        category=category,
        version=version,
        effective_date=date.fromisoformat(effective),
        expiration_date=date.fromisoformat(expires) if expires else None,
        status=status,
        confidentiality=confidentiality,
        access_level="customer",
        validation_status=ValidationStatus.VALID,
        owner="after-sales",
        source_path=f"company/{category}/{title}.md",
    )


@pytest.fixture
def documents() -> list[CompanyDocument]:
    """Six documents covering warranty, price, delivery, expiry and confidentiality."""
    return [
        _document(
            title="គោលការណ៍ធានា QN-4500A",
            product_id="QN-4500A",
            category="warranty",
            version="2.0",
            text=(
                "# គោលការណ៍ធានា\n"
                "ទូរទឹកកកម៉ូដែល QN-4500A មានការធានារយៈពេល ២៤ ខែ ចាប់ពីថ្ងៃទិញ។ "
                "ការធានាគ្របដណ្តប់លើកំហុសផលិតកម្ម ប៉ុន្តែមិនរាប់បញ្ចូលការខូចខាតដោយសារការប្រើប្រាស់មិនត្រឹមត្រូវឡើយ។ "
                "អតិថិជនត្រូវតែបង្ហាញវិក្កយបត្រដើមនៅពេលទាមទារសេវាកម្មធានា។"
            ),
        ),
        _document(
            title="តម្លៃលក់រាយ QN-4500A",
            product_id="QN-4500A",
            category="pricing",
            text=(
                "ទូរទឹកកក QN-4500A មានតម្លៃលក់រាយ 520 USD។ "
                "តម្លៃនេះរួមបញ្ចូលពន្ធអាករតម្លៃបន្ថែម ប៉ុន្តែមិនរួមបញ្ចូលថ្លៃដឹកជញ្ជូនទេ។"
            ),
        ),
        _document(
            title="លក្ខណៈបច្ចេកទេស RF-22B",
            product_id="RF-22B",
            category="product",
            text=(
                "ទូរទឹកកក RF-22B មានទំហំ 350 L និងប្រើថាមពល 180 W។ "
                "ម៉ូដែលនេះមានមុខងារបង្កកលឿន និងប្រព័ន្ធបញ្ជាសីតុណ្ហភាពឌីជីថល។"
            ),
        ),
        _document(
            title="សេវាកម្មដឹកជញ្ជូន",
            service_id="DELIVERY",
            category="policy",
            text=(
                "សេវាកម្មដឹកជញ្ជូនក្នុងរាជធានីភ្នំពេញចំណាយពេល 1 ថ្ងៃធ្វើការ។ "
                "ការដឹកជញ្ជូនទៅបណ្តាខេត្តចំណាយពេល 3 ថ្ងៃធ្វើការ។ "
                "ថ្លៃដឹកជញ្ជូនគឺឥតគិតថ្លៃសម្រាប់ការបញ្ជាទិញលើសពី 300 USD។"
            ),
        ),
        _document(
            title="ការផ្តល់ជូនពិសេសចូលឆ្នាំ",
            product_id="QN-4500A",
            category="pricing",
            version="1.0",
            effective="2025-04-01",
            expires="2025-04-30",
            text="បញ្ចុះតម្លៃ QN-4500A មកត្រឹម 420 USD ក្នុងអំឡុងពេលបុណ្យចូលឆ្នាំ។",
        ),
        _document(
            title="នីតិវិធីផ្ទៃក្នុងសម្រាប់បុគ្គលិក",
            category="policy",
            confidentiality=Confidentiality.INTERNAL,
            text="ការបញ្ចុះតម្លៃអតិបរមាដែលបុគ្គលិកអាចផ្តល់ជូនគឺ 15 ភាគរយ។ ឯកសារនេះសម្រាប់ផ្ទៃក្នុងតែប៉ុណ្ណោះ។",
        ),
    ]


@pytest.fixture
def conflicting_documents(documents: list[CompanyDocument]) -> list[CompanyDocument]:
    """Adds a second ACTIVE warranty document stating a different period."""
    return [
        *documents,
        _document(
            title="គោលការណ៍ធានា QN-4500A",
            product_id="QN-4500A",
            category="warranty",
            version="1.0",
            effective="2024-01-01",
            text="ទូរទឹកកកម៉ូដែល QN-4500A មានការធានារយៈពេល ១២ ខែ ចាប់ពីថ្ងៃទិញ។",
        ),
    ]


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    return tmp_path / "index"


@pytest.fixture
def built_index(
    documents: list[CompanyDocument], embedder: HashingEmbedder, index_dir: Path
) -> tuple[LocalVectorStore, str]:
    settings = IngestionSettings(
        index_root=index_dir, allow_non_semantic_embedder=True, embed_batch_size=4
    )
    store, manifest = build_index(
        documents, settings=settings, index_version="2026-08-14.1", embedder=embedder
    )
    assert isinstance(store, LocalVectorStore)
    store.persist()
    return store, manifest.index_version


@pytest.fixture
def retriever(
    built_index: tuple[LocalVectorStore, str], embedder: HashingEmbedder
) -> Retriever:
    store, version = built_index
    # The hashing embedder produces lower absolute similarities than a real
    # model, so the confidence floor is lowered for tests.  Threshold behaviour
    # itself is tested explicitly in test_retrieval.py.
    config = RetrievalConfig(
        top_k=4,
        candidate_k=12,
        min_score_to_answer=0.05,
        medium_confidence_score=0.15,
        high_confidence_score=0.30,
    )
    return Retriever(store, embedder, config=config, index_version=version)
