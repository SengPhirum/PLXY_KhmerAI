"""Application state container and FastAPI dependencies."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from common.logging import get_logger
from common.versions import PlatformVersions, load_versions
from server.chat_service import ChatService, PromptBuilder
from server.config import Settings, get_settings
from server.guardrails import InputGuard, OutputGuard
from server.models import ConversationStore
from server.ollama_client import OllamaClient
from server.rag_service import RagService
from server.rate_limit import InMemoryRateLimiter, NullRateLimiter, RateLimiter

log = get_logger(__name__)

__all__ = [
    "AppState",
    "get_state",
    "get_chat_service",
    "get_rag_service",
    "require_admin",
    "require_client",
]


@dataclass
class AppState:
    """Everything the app owns for its whole lifetime."""

    settings: Settings
    versions: PlatformVersions
    ollama: OllamaClient
    rag: RagService | None
    conversations: ConversationStore
    chat: ChatService
    rate_limiter: RateLimiter
    started_at: float = field(default_factory=time.time)
    _sweeper: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    def version_dict(self) -> dict[str, str]:
        versions = self.versions
        if self.rag is not None and self.rag.index_version != "unknown":
            versions = versions.with_index(self.rag.index_version)
        return {k: str(v) for k, v in versions.to_dict().items()}

    async def start(self) -> None:
        await self.ollama.start()
        if self.rag is not None:
            # Loading reads files and may download an embedder; keep it off the
            # event loop so startup does not block the health endpoint.
            await asyncio.get_running_loop().run_in_executor(None, self.rag.try_load)
        self._sweeper = asyncio.create_task(self._sweep_conversations())
        log.info(
            "app.started",
            extra={
                "env": self.settings.env,
                "model": self.settings.ollama_model,
                "rag_ready": bool(self.rag and self.rag.ready),
                "versions": self.version_dict(),
            },
        )

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
            self._sweeper = None
        await self.ollama.close()
        if self.rag is not None:
            self.rag.close()
        self.conversations.clear()
        log.info("app.stopped", extra={"uptime_seconds": round(self.uptime_seconds, 1)})

    async def _sweep_conversations(self) -> None:
        """Drop expired conversations so memory does not grow without bound."""
        interval = max(60, self.settings.conversation_ttl_seconds // 4)
        try:
            while True:
                await asyncio.sleep(interval)
                removed = self.conversations.purge_expired()
                if removed:
                    log.info("conversations.purged", extra={"count": removed})
        except asyncio.CancelledError:
            raise


def build_state(settings: Settings | None = None) -> AppState:
    """Construct the whole object graph.  Called once by the lifespan handler."""
    resolved = settings or get_settings()

    ollama = OllamaClient(
        resolved.ollama_base_url,
        model=resolved.ollama_model,
        connect_timeout=resolved.ollama_connect_timeout_s,
        read_timeout=resolved.ollama_read_timeout_s,
        total_timeout=resolved.ollama_total_timeout_s,
        max_active=resolved.max_active_generations,
        max_queue=resolved.max_queue_depth,
        queue_timeout=resolved.queue_timeout_s,
    )
    rag = RagService(resolved) if resolved.rag_enabled else None
    conversations = ConversationStore(
        ttl_seconds=resolved.conversation_ttl_seconds,
        max_turns=resolved.conversation_max_turns,
        enabled=resolved.conversation_store == "memory",
    )
    rag_policy = resolved.rag_policy()
    grounding = (rag_policy.get("grounding") or {}) if isinstance(rag_policy, dict) else {}

    chat = ChatService(
        settings=resolved,
        ollama=ollama,
        rag=rag,
        conversations=conversations,
        prompts=PromptBuilder(resolved),
        input_guard=InputGuard(max_chars=resolved.max_message_chars),
        output_guard=OutputGuard(
            min_grounding_precision=float(grounding.get("min_grounding_precision", 0.90))
        ),
    )
    limiter: RateLimiter = (
        InMemoryRateLimiter(
            requests=resolved.rate_limit_requests,
            window_seconds=resolved.rate_limit_window_seconds,
            burst=resolved.rate_limit_burst,
        )
        if resolved.rate_limit_enabled
        else NullRateLimiter()
    )

    return AppState(
        settings=resolved,
        versions=load_versions(index_root=resolved.index_root_path),
        ollama=ollama,
        rag=rag,
        conversations=conversations,
        chat=chat,
        rate_limiter=limiter,
    )


# --- FastAPI dependencies ---------------------------------------------------
def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "app_state", None)
    if state is None:  # pragma: no cover - only reachable on a misconfigured app
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="application not initialised"
        )
    return state


def get_chat_service(state: Annotated[AppState, Depends(get_state)]) -> ChatService:
    return state.chat


def get_rag_service(state: Annotated[AppState, Depends(get_state)]) -> RagService:
    if state.rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG is disabled (KHMERAI_RAG_ENABLED=false)",
        )
    return state.rag


def _constant_time_equal(a: str, b: str) -> bool:
    """Compare without leaking length or position through timing."""
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def require_admin(
    state: Annotated[AppState, Depends(get_state)],
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Guard for ``/v1/admin/*``.

    Accepts either ``X-Admin-Key`` or ``Authorization: Bearer <key>``.  Refuses
    outright when no admin key is configured - an unset key must never mean
    "anyone may reindex".
    """
    configured = state.settings.admin_api_key
    if not configured or configured.startswith("CHANGE_ME"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin endpoints are disabled: KHMERAI_ADMIN_API_KEY is not configured",
        )

    presented = x_admin_key
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    if not presented or not _constant_time_equal(presented, configured):
        log.warning("auth.admin_denied", extra={"presented": bool(presented)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing admin credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_client(
    state: Annotated[AppState, Depends(get_state)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Optional client authentication.  Returns the client identity for rate limiting."""
    settings = state.settings
    presented = x_api_key
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()

    if not settings.require_client_auth:
        return presented[:16] if presented else "anonymous"

    keys = settings.client_key_set
    if not keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="client authentication is required but KHMERAI_CLIENT_API_KEYS is empty",
        )
    if not presented or not any(_constant_time_equal(presented, k) for k in keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return presented[:16]
