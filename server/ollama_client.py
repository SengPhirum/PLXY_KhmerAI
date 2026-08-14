"""Async Ollama client with admission control.

The Mac Studio has one Ollama daemon with a fixed ``OLLAMA_NUM_PARALLEL``.
Sending more concurrent generations than that does not make anything faster - it
makes every request slower and eventually produces timeouts.  So the API layer
does its own admission control:

* a semaphore caps *active* generations at ``max_active_generations``;
* additional requests wait in a bounded queue for ``queue_timeout_s``;
* beyond ``max_queue_depth`` the server returns 503 with ``Retry-After`` instead
  of accepting work it cannot finish.

This is the mechanism behind the "10+ connected clients on 4-6 active
generations" decision in Phase 13: connections are cheap, generations are not.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from common.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "CapacityExceeded",
    "GenerationChunk",
    "GenerationResult",
    "OllamaClient",
    "OllamaError",
    "OllamaTimeout",
    "OllamaUnavailable",
]


class OllamaError(RuntimeError):
    """Base class for Ollama failures."""


class OllamaUnavailable(OllamaError):
    """The daemon is unreachable or returned 5xx."""


class OllamaTimeout(OllamaError):
    """Generation exceeded the configured timeout."""


class CapacityExceeded(OllamaError):
    """The queue is full or the wait exceeded ``queue_timeout_s``."""

    def __init__(self, message: str, retry_after: float = 5.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(slots=True)
class GenerationChunk:
    text: str
    done: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    queue_wait_ms: float = 0.0
    done_reason: str = ""

    @property
    def tokens_per_second(self) -> float:
        seconds = self.total_duration_ms / 1000.0
        return round(self.completion_tokens / seconds, 2) if seconds > 0 else 0.0

    def usage(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 1),
            "queue_wait_ms": round(self.queue_wait_ms, 1),
            "tokens_per_second": self.tokens_per_second,
        }


class OllamaClient:
    """Admission-controlled async client for the local Ollama daemon."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        model: str = "khmer-support-9b",
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
        total_timeout: float = 180.0,
        max_active: int = 4,
        max_queue: int = 64,
        queue_timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_active = max_active
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self._timeout = httpx.Timeout(
            timeout=total_timeout, connect=connect_timeout, read=read_timeout
        )
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(max_active)
        self._queued = 0
        self._active = 0
        self._queue_lock = asyncio.Lock()

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                limits=httpx.Limits(
                    max_connections=self.max_active + 8, max_keepalive_connections=self.max_active
                ),
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise OllamaError("OllamaClient.start() was not called")
        return self._client

    # -- introspection ------------------------------------------------------
    @property
    def active_generations(self) -> int:
        return self._active

    @property
    def queued_requests(self) -> int:
        return self._queued

    async def health(self) -> tuple[bool, str, float]:
        """Liveness probe: is the daemon answering ``/api/tags``?"""
        started = time.perf_counter()
        try:
            response = await self._http().get("/api/tags", timeout=httpx.Timeout(3.0))
            response.raise_for_status()
        except (httpx.HTTPError, OllamaError) as exc:
            return False, f"{type(exc).__name__}: {exc}", (time.perf_counter() - started) * 1000
        return True, "ok", (time.perf_counter() - started) * 1000

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            response = await self._http().get("/api/tags", timeout=httpx.Timeout(5.0))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"could not list models: {exc}") from exc
        return list(response.json().get("models", []))

    async def model_available(self, model: str) -> bool:
        try:
            names = {m.get("name", "") for m in await self.list_models()}
        except OllamaUnavailable:
            return False
        return model in names or any(n.split(":")[0] == model.split(":")[0] for n in names)

    # -- admission control --------------------------------------------------
    async def _acquire(self) -> float:
        """Wait for a generation slot.  Returns the queue wait in milliseconds."""
        async with self._queue_lock:
            if self._queued >= self.max_queue:
                raise CapacityExceeded(
                    f"generation queue is full ({self._queued}/{self.max_queue})",
                    retry_after=max(2.0, self.queue_timeout / 2),
                )
            self._queued += 1

        started = time.perf_counter()
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout)
        except TimeoutError as exc:
            async with self._queue_lock:
                self._queued -= 1
            raise CapacityExceeded(
                f"waited {self.queue_timeout:.0f}s for a generation slot",
                retry_after=self.queue_timeout,
            ) from exc

        async with self._queue_lock:
            self._queued -= 1
            self._active += 1
        return (time.perf_counter() - started) * 1000

    async def _release(self) -> None:
        async with self._queue_lock:
            self._active = max(0, self._active - 1)
        self._semaphore.release()

    # -- generation ---------------------------------------------------------
    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None,
        options: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": options or {},
            "keep_alive": -1,
        }

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Non-streaming generation."""
        queue_wait = await self._acquire()
        started = time.perf_counter()
        try:
            response = await self._http().post(
                "/api/chat",
                json=self._payload(messages, model=model, options=options, stream=False),
            )
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeout(f"generation timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaUnavailable(
                f"Ollama returned {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Ollama request failed: {exc}") from exc
        finally:
            await self._release()

        elapsed_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            text=(body.get("message") or {}).get("content", ""),
            model=body.get("model", model or self.model),
            prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(body.get("eval_count", 0) or 0),
            total_duration_ms=float(body.get("total_duration", 0) or 0) / 1e6 or elapsed_ms,
            time_to_first_token_ms=float(body.get("prompt_eval_duration", 0) or 0) / 1e6,
            queue_wait_ms=queue_wait,
            done_reason=str(body.get("done_reason", "")),
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        """Streaming generation.

        The slot is held for the whole stream and released in ``finally``, so a
        client that disconnects mid-stream (which raises ``CancelledError`` here)
        cannot leak a permit and starve the queue.
        """
        queue_wait = await self._acquire()
        started = time.perf_counter()
        first_token_at: float | None = None
        completion_tokens = 0
        try:
            async with self._http().stream(
                "POST",
                "/api/chat",
                json=self._payload(messages, model=model, options=options, stream=True),
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")[:200]
                    raise OllamaUnavailable(f"Ollama returned {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("ollama.stream.bad_json", extra={"line": line[:120]})
                        continue

                    piece = (event.get("message") or {}).get("content", "")
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        completion_tokens += 1
                        yield GenerationChunk(text=piece)

                    if event.get("done"):
                        total_ms = (time.perf_counter() - started) * 1000
                        ttft = (first_token_at - started) * 1000 if first_token_at else total_ms
                        yield GenerationChunk(
                            text="",
                            done=True,
                            metrics={
                                "model": event.get("model", model or self.model),
                                "prompt_tokens": int(event.get("prompt_eval_count", 0) or 0),
                                "completion_tokens": int(
                                    event.get("eval_count", 0) or completion_tokens
                                ),
                                "total_duration_ms": round(total_ms, 1),
                                "time_to_first_token_ms": round(ttft, 1),
                                "queue_wait_ms": round(queue_wait, 1),
                                "done_reason": str(event.get("done_reason", "")),
                            },
                        )
                        return
        except httpx.TimeoutException as exc:
            raise OllamaTimeout(f"streaming generation timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Ollama streaming failed: {exc}") from exc
        finally:
            await self._release()

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Embeddings through the same daemon (used by ``OllamaEmbedder``)."""
        try:
            response = await self._http().post("/api/embed", json={"model": model, "input": texts})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"embedding request failed: {exc}") from exc
        payload = response.json()
        embeddings = payload.get("embeddings")
        if embeddings is None and "embedding" in payload:
            embeddings = [payload["embedding"]]
        if not embeddings:
            raise OllamaError(f"no embeddings returned for model {model!r}")
        return embeddings

    def stats(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "active_generations": self._active,
            "queued_requests": self._queued,
            "max_active": self.max_active,
            "max_queue": self.max_queue,
        }
