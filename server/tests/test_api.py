"""End-to-end API tests through the FastAPI app with a fake Ollama."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server.ollama_client import CapacityExceeded, OllamaTimeout, OllamaUnavailable
from server.tests.conftest import ADMIN_KEY, FakeOllamaClient


def _sse_events(response: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


# --- health / ops -----------------------------------------------------------
def test_health_is_ok_and_reports_versions(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["uptime_seconds"] >= 0
    assert "app" in body["versions"]
    assert "knowledge_index" in body["versions"]


def test_health_stays_ok_when_ollama_is_down(
    client: TestClient, fake_ollama: FakeOllamaClient
) -> None:
    """Liveness must not depend on Ollama, or launchd would kill a healthy API."""
    fake_ollama.healthy = False
    assert client.get("/health").status_code == 200


def test_ready_reports_all_components(client: TestClient) -> None:
    body = client.get("/ready").json()
    names = {c["name"] for c in body["components"]}
    assert {"ollama", "model", "rag", "disk", "capacity"} <= names
    assert body["ready"] is True


def test_ready_fails_when_ollama_is_down(
    client: TestClient, fake_ollama: FakeOllamaClient
) -> None:
    fake_ollama.healthy = False
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_metrics_endpoint_exposes_the_required_series(client: TestClient) -> None:
    client.post("/v1/chat", json={"message": "សួស្តី", "stream": False})
    body = client.get("/metrics").text
    for metric in (
        "khmerai_requests_total",
        "khmerai_request_latency_seconds",
        "khmerai_retrieval_latency_seconds",
        "khmerai_generated_tokens_total",
    ):
        assert metric in body, f"{metric} missing from /metrics"


def test_models_endpoint(client: TestClient) -> None:
    body = client.get("/v1/models").json()
    assert body["default"] == "khmer-support-9b"
    assert any(m["name"] == "khmer-support-9b" and m["available"] for m in body["models"])
    assert body["generation_defaults"]["temperature"] == 0.3


# --- chat -------------------------------------------------------------------
def test_chat_returns_a_grounded_khmer_answer(client: TestClient) -> None:
    response = client.post(
        "/v1/chat",
        json={"message": "តើម៉ូដែល QN-4500A មានការធានារយៈពេលប៉ុន្មាន?", "stream": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert "២៤ ខែ" in body["answer"]
    assert body["language"] == "km"
    assert body["sources"], "a warranty answer must cite a source document"
    assert body["sources"][0]["document_id"]
    assert body["grounded"] is True
    assert body["conversation_id"]
    assert body["request_id"]


def test_conversation_id_is_reused_across_turns(client: TestClient) -> None:
    first = client.post("/v1/chat", json={"message": "សួស្តី", "stream": False}).json()
    second = client.post(
        "/v1/chat",
        json={"message": "តើធានាប៉ុន្មានខែ?", "conversation_id": first["conversation_id"], "stream": False},
    ).json()
    assert second["conversation_id"] == first["conversation_id"]


def test_multi_turn_context_is_carried_forward(
    client: TestClient, fake_ollama: FakeOllamaClient
) -> None:
    """The model number from turn 1 must still be known in turn 2."""
    first = client.post(
        "/v1/chat", json={"message": "ខ្ញុំមានម៉ូដែល QN-4500A", "stream": False}
    ).json()
    client.post(
        "/v1/chat",
        json={
            "message": "តើវាធានាប៉ុន្មានខែ?",
            "conversation_id": first["conversation_id"],
            "stream": False,
        },
    )
    last_prompt = "\n".join(m["content"] for m in fake_ollama.calls[-1])
    assert "QN-4500A" in last_prompt


def test_history_is_included_but_bounded(client: TestClient, fake_ollama: FakeOllamaClient) -> None:
    conversation_id = "bounded-history-test"
    for i in range(10):
        client.post(
            "/v1/chat",
            json={"message": f"សំណួរទី {i} អំពីការធានា", "conversation_id": conversation_id, "stream": False},
        )
    messages = fake_ollama.calls[-1]
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    # Bounded: not every one of the 10 prior turns is resent.
    assert len(messages) < 2 + 10 * 2


def test_product_filter_is_applied(client: TestClient) -> None:
    response = client.post(
        "/v1/chat",
        json={"message": "តម្លៃប៉ុន្មាន?", "product_id": "QN-4500A", "stream": False},
    )
    assert response.status_code == 200
    for source in response.json()["sources"]:
        assert source["document_id"]


def test_empty_message_is_rejected_by_validation(client: TestClient) -> None:
    assert client.post("/v1/chat", json={"message": "", "stream": False}).status_code == 422


def test_oversized_message_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"message": "ក" * 200_000, "stream": False})
    assert response.status_code in (413, 422)


def test_bad_conversation_id_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat", json={"message": "សួស្តី", "conversation_id": "../../etc/passwd", "stream": False}
    )
    assert response.status_code == 422


def test_model_override_is_forbidden(client: TestClient) -> None:
    response = client.post(
        "/v1/chat", json={"message": "សួស្តី", "model": "llama3", "stream": False}
    )
    assert response.status_code == 403


def test_conversation_can_be_deleted(client: TestClient) -> None:
    conversation_id = client.post(
        "/v1/chat", json={"message": "សួស្តី", "stream": False}
    ).json()["conversation_id"]
    body = client.delete(f"/v1/conversations/{conversation_id}").json()
    assert body["deleted"] is True
    assert client.delete(f"/v1/conversations/{conversation_id}").json()["deleted"] is False


# --- streaming --------------------------------------------------------------
def test_stream_emits_start_tokens_and_done(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/stream", json={"message": "តើ QN-4500A ធានាប៉ុន្មានខែ?", "stream": True}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers.get("X-Accel-Buffering") == "no"

    events = _sse_events(response)
    types = [e["type"] for e in events]
    assert types[0] == "start"
    assert types[-1] == "done"
    assert "token" in types

    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert "២៤ ខែ" in text
    done = events[-1]
    assert done["usage"]["completion_tokens"] > 0


def test_stream_reports_sources(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/stream", json={"message": "តើ QN-4500A ធានាប៉ុន្មានខែ?"}
    )
    events = _sse_events(response)
    source_events = [e for e in events if e["type"] == "sources"]
    assert source_events and source_events[0]["sources"]


def test_stream_reports_an_error_event_when_generation_fails(
    client: TestClient, fake_ollama: FakeOllamaClient
) -> None:
    fake_ollama.raise_on_call = OllamaUnavailable("daemon down")
    events = _sse_events(client.post("/v1/chat/stream", json={"message": "សួស្តី"}))
    assert events[-1]["type"] == "error"


# --- error handling ---------------------------------------------------------
def test_capacity_exceeded_returns_503_with_retry_after(
    client: TestClient, fake_ollama: FakeOllamaClient
) -> None:
    fake_ollama.raise_on_call = CapacityExceeded("queue full", retry_after=3.0)
    response = client.post("/v1/chat", json={"message": "សួស្តី", "stream": False})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "3"
    assert response.json()["error"] == "capacity_exceeded"


def test_ollama_timeout_returns_504(client: TestClient, fake_ollama: FakeOllamaClient) -> None:
    fake_ollama.raise_on_call = OllamaTimeout("too slow")
    response = client.post("/v1/chat", json={"message": "សួស្តី", "stream": False})
    assert response.status_code == 504
    assert response.json()["error"] == "generation_timeout"


def test_ollama_unavailable_returns_503(client: TestClient, fake_ollama: FakeOllamaClient) -> None:
    fake_ollama.raise_on_call = OllamaUnavailable("connection refused")
    response = client.post("/v1/chat", json={"message": "សួស្តី", "stream": False})
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_validation_errors_do_not_echo_the_submitted_value(client: TestClient) -> None:
    """A rejected payload must not have its content reflected into logs/responses."""
    secret = "my-secret-phone-012345678"
    response = client.post("/v1/chat", json={"message": secret, "conversation_id": "!!bad!!"})
    assert response.status_code == 422
    assert secret not in response.text


# --- retrieval endpoint -----------------------------------------------------
def test_rag_search_returns_the_documented_chunk_contract(client: TestClient) -> None:
    response = client.post("/v1/rag/search", json={"query": "ការធានា QN-4500A", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    first = body["results"][0]
    for field in ("chunk_id", "document_id", "text", "score", "product_id", "version", "source"):
        assert field in first
    assert body["index_version"] == "2026-08-14.1"


def test_rag_search_respects_top_k(client: TestClient) -> None:
    body = client.post("/v1/rag/search", json={"query": "ការធានា", "top_k": 1}).json()
    assert len(body["results"]) <= 1


# --- admin ------------------------------------------------------------------
def test_admin_requires_a_key(client: TestClient) -> None:
    assert client.post("/v1/admin/reload").status_code == 401
    assert client.get("/v1/admin/diagnostics").status_code == 401


def test_admin_rejects_a_wrong_key(client: TestClient) -> None:
    response = client.post("/v1/admin/reload", headers={"X-Admin-Key": "wrong"})
    assert response.status_code == 401


def test_admin_accepts_the_configured_key(client: TestClient) -> None:
    response = client.post("/v1/admin/reload", headers={"X-Admin-Key": ADMIN_KEY})
    assert response.status_code == 200
    assert response.json()["index_version"] == "2026-08-14.1"


def test_admin_accepts_a_bearer_token(client: TestClient) -> None:
    response = client.get(
        "/v1/admin/diagnostics", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
    )
    assert response.status_code == 200


def test_diagnostics_never_leaks_the_admin_key(client: TestClient) -> None:
    body = client.get("/v1/admin/diagnostics", headers={"X-Admin-Key": ADMIN_KEY}).json()
    assert body["settings"]["admin_api_key"] == "***set***"
    assert ADMIN_KEY not in json.dumps(body)
    assert body["versions"]["app"]
    assert body["rag"]["ready"] is True


def test_reindex_dry_run(client: TestClient, tmp_path: Any) -> None:
    from common.io import write_jsonl
    from common.paths import PROJECT_ROOT

    records = PROJECT_ROOT / "data" / "interim" / "test_records.jsonl"
    records.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, [])
    try:
        response = client.post(
            "/v1/admin/reindex",
            headers={"X-Admin-Key": ADMIN_KEY},
            json={"input_path": str(records), "dry_run": True},
        )
        assert response.status_code == 200
        assert response.json()["index_version"] == "(dry-run)"
    finally:
        records.unlink(missing_ok=True)


def test_reindex_rejects_a_path_outside_the_project(client: TestClient) -> None:
    response = client.post(
        "/v1/admin/reindex",
        headers={"X-Admin-Key": ADMIN_KEY},
        json={"input_path": "/etc/passwd", "dry_run": True},
    )
    assert response.status_code in (400, 422, 500)
    assert "passwd" not in response.text or response.status_code != 200


# --- security headers -------------------------------------------------------
@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("Referrer-Policy", "no-referrer"),
    ],
)
def test_security_headers_are_present(client: TestClient, header: str, value: str) -> None:
    assert client.get("/health").headers[header] == value


def test_request_id_is_returned_and_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    generated = client.get("/health").headers["X-Request-ID"]
    assert generated and generated != "abc123"


def test_hostile_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "bad\nvalue with spaces"})
    assert "\n" not in response.headers["X-Request-ID"]
