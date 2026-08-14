"""Prometheus metrics (Phase 19).

Every metric named in the specification is implemented here.  ``prometheus_client``
is an optional dependency: when it is absent the module degrades to no-op stubs
so the API still runs, and ``/metrics`` reports that metrics are disabled rather
than 500ing.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = [
    "METRICS_AVAILABLE",
    "CONTENT_TYPE",
    "render",
    "record_request",
    "track_latency",
    "observe_ttft",
    "observe_tokens",
    "observe_retrieval",
    "record_error",
    "record_escalation",
    "record_unknown_answer",
    "record_rate_limit",
    "set_active",
    "set_queued",
    "record_injection_block",
    "record_guardrail_block",
    "set_index_info",
]

try:  # pragma: no cover - exercised by whichever branch the environment takes
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    METRICS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

CONTENT_TYPE = CONTENT_TYPE_LATEST


class _NoopMetric:
    """Stand-in used when prometheus_client is not installed."""

    def labels(self, *args: Any, **kwargs: Any) -> _NoopMetric:
        return self

    def inc(self, amount: float = 1) -> None: ...
    def dec(self, amount: float = 1) -> None: ...
    def set(self, value: float) -> None: ...
    def observe(self, value: float) -> None: ...
    def info(self, values: dict[str, str]) -> None: ...


if METRICS_AVAILABLE:
    REGISTRY = CollectorRegistry(auto_describe=True)

    # Latency buckets are chosen around the SLOs in configs/base.yaml
    # (TTFT p95 2.5s, end-to-end p95 12s) so the histogram has resolution
    # exactly where the alerting thresholds sit.
    _LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60)
    _TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 4, 6, 10)
    _RETRIEVAL_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.5, 3)

    requests_total = Counter(
        "khmerai_requests_total", "API requests", ["endpoint", "status"], registry=REGISTRY
    )
    requests_active = Gauge(
        "khmerai_requests_active", "In-flight generations", registry=REGISTRY
    )
    requests_queued = Gauge(
        "khmerai_requests_queued", "Requests waiting for a generation slot", registry=REGISTRY
    )
    request_latency_seconds = Histogram(
        "khmerai_request_latency_seconds",
        "End-to-end request latency",
        ["endpoint"],
        buckets=_LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    time_to_first_token_seconds = Histogram(
        "khmerai_time_to_first_token_seconds",
        "Time to first generated token",
        buckets=_TTFT_BUCKETS,
        registry=REGISTRY,
    )
    generated_tokens_total = Counter(
        "khmerai_generated_tokens_total", "Tokens generated", ["model"], registry=REGISTRY
    )
    retrieval_latency_seconds = Histogram(
        "khmerai_retrieval_latency_seconds",
        "Retrieval latency",
        buckets=_RETRIEVAL_BUCKETS,
        registry=REGISTRY,
    )
    retrieval_empty_total = Counter(
        "khmerai_retrieval_empty_total",
        "Retrievals that returned no usable context",
        ["reason"],
        registry=REGISTRY,
    )
    ollama_errors_total = Counter(
        "khmerai_ollama_errors_total", "Ollama failures", ["kind"], registry=REGISTRY
    )
    api_errors_total = Counter(
        "khmerai_api_errors_total", "API errors", ["endpoint", "kind"], registry=REGISTRY
    )
    escalations_total = Counter(
        "khmerai_escalations_total", "Escalations to a human", ["reason"], registry=REGISTRY
    )
    unknown_answers_total = Counter(
        "khmerai_unknown_answers_total",
        "Answers where the assistant stated it did not know",
        ["intent"],
        registry=REGISTRY,
    )
    rate_limit_total = Counter(
        "khmerai_rate_limit_total", "Rate-limited requests", ["scope"], registry=REGISTRY
    )
    injection_blocks_total = Counter(
        "khmerai_injection_blocks_total",
        "Prompt-injection attempts blocked",
        ["surface"],
        registry=REGISTRY,
    )
    guardrail_blocks_total = Counter(
        "khmerai_guardrail_blocks_total", "Answers blocked by guardrails", ["reason"], registry=REGISTRY
    )
    index_info = Info("khmerai_index", "Active knowledge index", registry=REGISTRY)
else:  # pragma: no cover
    REGISTRY = None  # type: ignore[assignment]
    requests_total = requests_active = requests_queued = _NoopMetric()
    request_latency_seconds = time_to_first_token_seconds = _NoopMetric()
    generated_tokens_total = retrieval_latency_seconds = _NoopMetric()
    retrieval_empty_total = ollama_errors_total = api_errors_total = _NoopMetric()
    escalations_total = unknown_answers_total = rate_limit_total = _NoopMetric()
    injection_blocks_total = guardrail_blocks_total = index_info = _NoopMetric()


# --- recording helpers ------------------------------------------------------
def record_request(endpoint: str, status: int | str) -> None:
    requests_total.labels(endpoint=endpoint, status=str(status)).inc()


@contextmanager
def track_latency(endpoint: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        request_latency_seconds.labels(endpoint=endpoint).observe(time.perf_counter() - started)


def observe_ttft(seconds: float) -> None:
    time_to_first_token_seconds.observe(max(0.0, seconds))


def observe_tokens(model: str, count: int) -> None:
    if count > 0:
        generated_tokens_total.labels(model=model).inc(count)


def observe_retrieval(seconds: float, *, empty: bool = False, reason: str = "no_match") -> None:
    retrieval_latency_seconds.observe(max(0.0, seconds))
    if empty:
        retrieval_empty_total.labels(reason=reason).inc()


def record_error(endpoint: str, kind: str, *, ollama: bool = False) -> None:
    api_errors_total.labels(endpoint=endpoint, kind=kind).inc()
    if ollama:
        ollama_errors_total.labels(kind=kind).inc()


def record_escalation(reason: str) -> None:
    escalations_total.labels(reason=reason).inc()


def record_unknown_answer(intent: str) -> None:
    unknown_answers_total.labels(intent=intent).inc()


def record_rate_limit(scope: str = "client") -> None:
    rate_limit_total.labels(scope=scope).inc()


def record_injection_block(surface: str) -> None:
    injection_blocks_total.labels(surface=surface).inc()


def record_guardrail_block(reason: str) -> None:
    guardrail_blocks_total.labels(reason=reason).inc()


def set_active(value: int) -> None:
    requests_active.set(value)


def set_queued(value: int) -> None:
    requests_queued.set(value)


def set_index_info(versions: dict[str, str]) -> None:
    index_info.info({k: str(v) for k, v in versions.items()})


def render() -> bytes:
    """Prometheus exposition payload."""
    if not METRICS_AVAILABLE:  # pragma: no cover
        return (
            b"# prometheus_client is not installed; metrics are disabled.\n"
            b"# Install it with: pip install -r requirements/server.txt\n"
        )
    return generate_latest(REGISTRY)
