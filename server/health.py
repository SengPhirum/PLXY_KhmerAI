"""Liveness and readiness checks.

``/health`` is a **liveness** probe: is the process itself alive and able to
answer?  It must stay cheap and must not depend on Ollama, otherwise a
restarting model daemon would cause launchd to kill an otherwise healthy API.

``/ready`` is a **readiness** probe: can this instance actually serve a customer
request right now?  It checks Ollama, the model, the index and the embedder.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

from common.logging import get_logger
from server.dependencies import AppState
from server.schemas import ComponentHealth, HealthResponse, ReadyResponse

log = get_logger(__name__)

__all__ = ["liveness", "readiness", "host_stats"]


def liveness(state: AppState) -> HealthResponse:
    return HealthResponse(
        status="ok",
        uptime_seconds=round(state.uptime_seconds, 1),
        versions=state.version_dict(),
        active_generations=state.ollama.active_generations,
        queued_requests=state.ollama.queued_requests,
    )


async def _check_ollama(state: AppState) -> list[ComponentHealth]:
    healthy, detail, latency_ms = await state.ollama.health()
    checks = [
        ComponentHealth(
            name="ollama", healthy=healthy, detail=detail, latency_ms=round(latency_ms, 1)
        )
    ]
    if healthy:
        model = state.settings.ollama_model
        available = await state.ollama.model_available(model)
        checks.append(
            ComponentHealth(
                name="model",
                healthy=available,
                detail=(
                    f"{model} loaded"
                    if available
                    else f"{model} not found; create it with `bash ollama/create_model.sh`"
                ),
            )
        )
    return checks


def _check_rag(state: AppState) -> ComponentHealth:
    if state.rag is None:
        return ComponentHealth(name="rag", healthy=True, detail="disabled")
    healthy, detail = state.rag.health()
    return ComponentHealth(name="rag", healthy=healthy, detail=detail)


def _check_disk(state: AppState) -> ComponentHealth:
    try:
        usage = shutil.disk_usage(state.settings.index_root_path.parent)
    except OSError as exc:
        return ComponentHealth(name="disk", healthy=False, detail=str(exc))
    free_gb = usage.free / 1e9
    return ComponentHealth(
        name="disk",
        healthy=free_gb > 5.0,
        detail=f"{free_gb:.1f} GB free of {usage.total / 1e9:.0f} GB",
    )


def _check_capacity(state: AppState) -> ComponentHealth:
    queued = state.ollama.queued_requests
    limit = state.settings.max_queue_depth
    return ComponentHealth(
        name="capacity",
        healthy=queued < limit,
        detail=(
            f"{state.ollama.active_generations}/{state.settings.max_active_generations} active, "
            f"{queued}/{limit} queued"
        ),
    )


async def readiness(state: AppState) -> ReadyResponse:
    started = time.perf_counter()
    components = await _check_ollama(state)
    components.append(_check_rag(state))
    components.append(_check_disk(state))
    components.append(_check_capacity(state))

    # RAG being unready is a degradation, not an outage: general questions still
    # work, so it must not take the instance out of the load-balancer rotation.
    blocking = {"ollama", "model", "disk"}
    ready = all(c.healthy for c in components if c.name in blocking)

    log.debug(
        "health.ready",
        extra={
            "ready": ready,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "unhealthy": [c.name for c in components if not c.healthy],
        },
    )
    return ReadyResponse(ready=ready, components=components, versions=state.version_dict())


def host_stats(index_root: Path | None = None) -> dict[str, Any]:
    """Host telemetry for the operations runbook; degrades if psutil is absent."""
    stats: dict[str, Any] = {}
    try:
        import psutil  # noqa: PLC0415 - optional dependency

        memory = psutil.virtual_memory()
        stats.update(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_total_gb=round(memory.total / 1e9, 2),
            memory_used_gb=round(memory.used / 1e9, 2),
            memory_percent=memory.percent,
            load_average=list(psutil.getloadavg()),
        )
    except (ImportError, OSError, AttributeError):
        stats["note"] = "psutil not available; install requirements/server.txt for host metrics"

    if index_root is not None:
        try:
            usage = shutil.disk_usage(index_root)
            stats.update(
                disk_total_gb=round(usage.total / 1e9, 2),
                disk_free_gb=round(usage.free / 1e9, 2),
            )
        except OSError:
            pass
    return stats


async def wait_until_ready(state: AppState, *, timeout: float = 60.0, interval: float = 2.0) -> bool:
    """Poll readiness - used by ``scripts/smoke_test.sh`` and the installer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (await readiness(state)).ready:
            return True
        await asyncio.sleep(interval)
    return False
