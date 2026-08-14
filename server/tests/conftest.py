"""Test harness for the API: a fake Ollama plus a real, tiny knowledge index.

The fake Ollama implements the same interface and the same admission-control
semantics as the real client, so concurrency, queueing, timeouts and streaming
are all exercised without a model.  Its ``script`` lets a test decide exactly
what the "model" says, which is what makes guardrail behaviour testable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from company_data.schema import CompanyDocument
from rag.embeddings import HashingEmbedder
from rag.ingestion import IngestionSettings, build_index
from rag.reindex import activate
from server.config import Settings
from server.ollama_client import CapacityExceeded, GenerationChunk, GenerationResult

ADMIN_KEY = "test-admin-key-do-not-use-in-production"

WARRANTY_TEXT = (
    "ទូរទឹកកកម៉ូដែល QN-4500A មានការធានារយៈពេល ២៤ ខែ ចាប់ពីថ្ងៃទិញ។ "
    "ការធានាគ្របដណ្តប់លើកំហុសផលិតកម្ម ប៉ុន្តែមិនរាប់បញ្ចូលការខូចខាតដោយសារការប្រើប្រាស់មិនត្រឹមត្រូវ។"
)
PRICE_TEXT = "ទូរទឹកកក QN-4500A មានតម្លៃលក់រាយ 520 USD រួមបញ្ចូលពន្ធអាករតម្លៃបន្ថែម។"
DELIVERY_TEXT = (
    "សេវាកម្មដឹកជញ្ជូនក្នុងរាជធានីភ្នំពេញចំណាយពេល 1 ថ្ងៃធ្វើការ "
    "ហើយទៅបណ្តាខេត្តចំណាយពេល 3 ថ្ងៃធ្វើការ។"
)


class FakeOllamaClient:
    """Drop-in replacement for :class:`server.ollama_client.OllamaClient`."""

    def __init__(
        self,
        *,
        model: str = "khmer-support-9b",
        max_active: int = 4,
        max_queue: int = 8,
        queue_timeout: float = 0.5,
    ) -> None:
        self.model = model
        self.max_active = max_active
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.base_url = "http://fake-ollama"

        # Test controls.
        self.script: str = "ម៉ូដែល QN-4500A មានការធានារយៈពេល ២៤ ខែ ចាប់ពីថ្ងៃទិញ។ [1]"
        self.raise_on_call: Exception | None = None
        self.delay: float = 0.0
        self.healthy: bool = True
        self.available_models: list[dict[str, Any]] = [
            {
                "name": "khmer-support-9b",
                "size": 6_000_000_000,
                "details": {"parameter_size": "9B", "quantization_level": "Q4_K_M"},
            }
        ]
        self.calls: list[list[dict[str, str]]] = []

        self._semaphore = asyncio.Semaphore(max_active)
        self._active = 0
        self._queued = 0
        self.peak_active = 0

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None: ...
    async def close(self) -> None: ...

    @property
    def active_generations(self) -> int:
        return self._active

    @property
    def queued_requests(self) -> int:
        return self._queued

    async def health(self) -> tuple[bool, str, float]:
        return (self.healthy, "ok" if self.healthy else "connection refused", 1.0)

    async def list_models(self) -> list[dict[str, Any]]:
        return self.available_models

    async def model_available(self, model: str) -> bool:
        return any(m["name"] == model for m in self.available_models)

    def stats(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "active_generations": self._active,
            "queued_requests": self._queued,
            "max_active": self.max_active,
            "max_queue": self.max_queue,
        }

    # -- admission control (mirrors the real client) ------------------------
    async def _acquire(self) -> float:
        if self._queued >= self.max_queue:
            raise CapacityExceeded("queue full", retry_after=1.0)
        self._queued += 1
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout)
        except TimeoutError as exc:
            self._queued -= 1
            raise CapacityExceeded("slot wait timed out", retry_after=1.0) from exc
        self._queued -= 1
        self._active += 1
        self.peak_active = max(self.peak_active, self._active)
        return 0.0

    def _release(self) -> None:
        self._active = max(0, self._active - 1)
        self._semaphore.release()

    # -- generation ---------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> GenerationResult:
        self.calls.append(messages)
        await self._acquire()
        try:
            if self.raise_on_call is not None:
                raise self.raise_on_call
            if self.delay:
                await asyncio.sleep(self.delay)
            return GenerationResult(
                text=self.script,
                model=model or self.model,
                prompt_tokens=sum(len(m["content"]) // 3 for m in messages),
                completion_tokens=max(1, len(self.script) // 3),
                total_duration_ms=12.0,
                time_to_first_token_ms=4.0,
            )
        finally:
            self._release()

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        self.calls.append(messages)
        await self._acquire()
        try:
            if self.raise_on_call is not None:
                raise self.raise_on_call
            words = self.script.split(" ")
            for index, word in enumerate(words):
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield GenerationChunk(text=word + (" " if index < len(words) - 1 else ""))
            yield GenerationChunk(
                text="",
                done=True,
                metrics={
                    "model": model or self.model,
                    "prompt_tokens": 100,
                    "completion_tokens": len(words),
                    "total_duration_ms": 15.0,
                    "time_to_first_token_ms": 3.0,
                    "queue_wait_ms": 0.0,
                    "done_reason": "stop",
                },
            )
        finally:
            self._release()


def _documents() -> list[CompanyDocument]:
    common = {
        "status": "active",
        "confidentiality": "public",
        "access_level": "customer",
        "validation_status": "valid",
        "effective_date": date(2026, 1, 1),
        "owner": "after-sales",
    }
    return [
        CompanyDocument(
            document_title="គោលការណ៍ធានា QN-4500A",
            text=WARRANTY_TEXT,
            product_id="QN-4500A",
            category="warranty",
            version="2.0",
            source_path="company/warranty/qn4500a.md",
            **common,  # type: ignore[arg-type]
        ),
        CompanyDocument(
            document_title="តម្លៃលក់រាយ QN-4500A",
            text=PRICE_TEXT,
            product_id="QN-4500A",
            category="pricing",
            source_path="company/pricing/qn4500a.md",
            **common,  # type: ignore[arg-type]
        ),
        CompanyDocument(
            document_title="សេវាកម្មដឹកជញ្ជូន",
            text=DELIVERY_TEXT,
            service_id="DELIVERY",
            category="policy",
            source_path="company/policy/delivery.md",
            **common,  # type: ignore[arg-type]
        ),
    ]


@pytest.fixture
def index_root(tmp_path: Path) -> Path:
    """A built and activated knowledge index."""
    root = tmp_path / "index"
    settings = IngestionSettings(index_root=root, allow_non_semantic_embedder=True)
    store, manifest = build_index(
        _documents(),
        settings=settings,
        index_version="2026-08-14.1",
        embedder=HashingEmbedder(dim=256),
    )
    store.persist()
    activate(manifest.index_version, index_root=root)
    return root


@pytest.fixture
def settings(index_root: Path) -> Settings:
    return Settings(
        env="development",
        admin_api_key=ADMIN_KEY,
        rag_enabled=True,
        index_root=str(index_root),
        embedding_backend="hashing",
        embedding_dim=256,
        embedding_model="hashing-ngram",
        vector_backend="local",
        max_active_generations=2,
        max_queue_depth=8,
        queue_timeout_s=0.5,
        rate_limit_enabled=False,
        metrics_enabled=True,
        conversation_ttl_seconds=60,
        support_hotline="000-000-000",
        support_email="support@example.com",
        num_ctx=4096,
        max_output_tokens=256,
        log_format="console",
    )


@pytest.fixture
def fake_ollama() -> FakeOllamaClient:
    return FakeOllamaClient(max_active=2, max_queue=8, queue_timeout=0.5)


@pytest.fixture
def client(
    settings: Settings, fake_ollama: FakeOllamaClient, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A TestClient whose app uses the test settings and the fake Ollama."""
    import server.dependencies as deps
    from server.main import create_app

    original_build = deps.build_state

    def _build(_settings: Settings | None = None, **kwargs: object) -> deps.AppState:
        state = original_build(settings, **kwargs)  # type: ignore[arg-type]
        state.ollama = fake_ollama  # type: ignore[assignment]
        state.chat.ollama = fake_ollama  # type: ignore[assignment]
        return state

    monkeypatch.setattr(deps, "build_state", _build)
    monkeypatch.setattr("server.main.build_state", _build)

    application = create_app(settings)
    with TestClient(application) as test_client:
        # Retrieval uses the hashing embedder, whose absolute similarities are
        # lower than a real model's; relax the confidence floor for the tests.
        state: deps.AppState = application.state.app_state
        if state.rag is not None and state.rag._retriever is not None:  # noqa: SLF001
            state.rag._retriever.config.min_score_to_answer = 0.02  # noqa: SLF001
            state.rag._retriever.config.medium_confidence_score = 0.05  # noqa: SLF001
            state.rag._retriever.config.high_confidence_score = 0.20  # noqa: SLF001
        yield test_client


@pytest.fixture
def app_state(client: TestClient) -> Any:
    return client.app.state.app_state  # type: ignore[union-attr]
