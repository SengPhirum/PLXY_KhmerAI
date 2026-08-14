"""FastAPI application (Phase 14).

Endpoints::

    GET  /health              liveness - cheap, never touches Ollama
    GET  /ready               readiness - Ollama, model, index, disk, capacity
    GET  /metrics             Prometheus exposition
    POST /v1/chat             non-streaming answer
    POST /v1/chat/stream      server-sent events
    POST /v1/rag/search       retrieval only (debugging + the ops runbook)
    POST /v1/admin/reindex    build/activate a knowledge index   [admin]
    POST /v1/admin/reload     hot-swap to the active index       [admin]
    GET  /v1/admin/diagnostics versions, config, host stats      [admin]
    GET  /v1/models           served models and versions
    DELETE /v1/conversations/{id}  forget a conversation (privacy)

Run it::

    make serve                        # development, autoreload
    make serve-prod                   # as launchd runs it
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from common.logging import configure_logging, conversation_id_var, get_logger
from rag.schemas import RetrievalFilters
from server import metrics
from server.chat_service import ChatService
from server.config import Settings, get_settings
from server.dependencies import (
    AppState,
    build_state,
    get_chat_service,
    get_rag_service,
    get_state,
    require_admin,
    require_client,
)
from server.health import host_stats, liveness, readiness
from server.middleware import (
    RateLimitMiddleware,
    RequestContextMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from server.ollama_client import CapacityExceeded, OllamaError, OllamaTimeout, OllamaUnavailable
from server.rag_service import RagService
from server.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    ReadyResponse,
    ReindexRequest,
    ReindexResponse,
    SearchRequest,
    SearchResponse,
)

log = get_logger(__name__)

__all__ = ["app", "create_app"]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings: Settings = getattr(application.state, "settings", None) or get_settings()
    configure_logging(settings.log_level, fmt=settings.log_format, force=True)

    state = build_state(settings)
    application.state.app_state = state
    await state.start()
    metrics.set_index_info(state.version_dict())
    try:
        yield
    finally:
        await state.stop()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.  Tests call this with an overridden ``Settings``."""
    resolved = settings or get_settings()

    application = FastAPI(
        title="Khmer Customer-Support LLM API",
        description=(
            "Khmer-first, RAG-grounded customer-support assistant served by a local Ollama "
            "runtime. Company facts come only from the retrieval index; the assistant states "
            "uncertainty rather than guessing."
        ),
        version=resolved.policy().get("versions", {}).get("app", "1.0.0"),
        lifespan=lifespan,
        docs_url="/docs" if not resolved.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not resolved.is_production else None,
    )
    application.state.settings = resolved

    # Middleware runs bottom-up: security headers outermost, rate limiting last.
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(RequestSizeLimitMiddleware, max_bytes=resolved.max_request_bytes)
    application.add_middleware(RequestContextMiddleware, header=resolved.request_id_header)
    if resolved.cors_origin_list:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_origin_list,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "X-API-Key", "Authorization", resolved.request_id_header],
            expose_headers=[resolved.request_id_header, "X-Response-Time-Ms"],
        )

    _register_routes(application)
    _register_error_handlers(application)
    return application


def _register_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(CapacityExceeded)
    async def _capacity(request: Request, exc: CapacityExceeded) -> JSONResponse:
        metrics.record_error(request.url.path, "capacity", ollama=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="capacity_exceeded",
                detail=str(exc),
                request_id=getattr(request.state, "request_id", ""),
                retry_after_seconds=exc.retry_after,
            ).model_dump(mode="json"),
            headers={"Retry-After": str(max(1, int(exc.retry_after)))},
        )

    @application.exception_handler(OllamaTimeout)
    async def _timeout(request: Request, exc: OllamaTimeout) -> JSONResponse:
        metrics.record_error(request.url.path, "timeout", ollama=True)
        log.error("api.ollama_timeout", extra={"path": request.url.path, "error": str(exc)})
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=ErrorResponse(
                error="generation_timeout",
                detail="the model did not respond in time; please try again",
                request_id=getattr(request.state, "request_id", ""),
            ).model_dump(mode="json"),
        )

    @application.exception_handler(OllamaUnavailable)
    async def _unavailable(request: Request, exc: OllamaUnavailable) -> JSONResponse:
        metrics.record_error(request.url.path, "ollama_unavailable", ollama=True)
        log.error("api.ollama_unavailable", extra={"path": request.url.path, "error": str(exc)})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="model_unavailable",
                detail="the language model service is not reachable",
                request_id=getattr(request.state, "request_id", ""),
                retry_after_seconds=10.0,
            ).model_dump(mode="json"),
            headers={"Retry-After": "10"},
        )

    @application.exception_handler(OllamaError)
    async def _ollama_error(request: Request, exc: OllamaError) -> JSONResponse:
        metrics.record_error(request.url.path, "ollama_error", ollama=True)
        log.error("api.ollama_error", extra={"path": request.url.path, "error": str(exc)})
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorResponse(
                error="generation_failed",
                detail="the model could not complete this request",
                request_id=getattr(request.state, "request_id", ""),
            ).model_dump(mode="json"),
        )

    @application.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        metrics.record_error(request.url.path, "validation")
        # Echo only field names and messages - never the submitted values, which
        # may contain customer PII.
        problems = [
            {"field": ".".join(str(p) for p in err.get("loc", [])), "message": err.get("msg", "")}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": problems,
                "request_id": getattr(request.state, "request_id", ""),
            },
        )


def _register_routes(application: FastAPI) -> None:
    # --- health ------------------------------------------------------------
    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(state: Annotated[AppState, Depends(get_state)]) -> HealthResponse:
        metrics.set_active(state.ollama.active_generations)
        metrics.set_queued(state.ollama.queued_requests)
        return liveness(state)

    @application.get("/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready(
        response: Response, state: Annotated[AppState, Depends(get_state)]
    ) -> ReadyResponse:
        result = await readiness(state)
        if not result.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    @application.get("/metrics", tags=["ops"])
    async def prometheus(state: Annotated[AppState, Depends(get_state)]) -> Response:
        if not state.settings.metrics_enabled:
            raise HTTPException(status_code=404, detail="metrics are disabled")
        metrics.set_active(state.ollama.active_generations)
        metrics.set_queued(state.ollama.queued_requests)
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)

    # --- chat --------------------------------------------------------------
    @application.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(
        request: Request,
        payload: ChatRequest,
        state: Annotated[AppState, Depends(get_state)],
        service: Annotated[ChatService, Depends(get_chat_service)],
        _client: Annotated[str, Depends(require_client)],
    ) -> ChatResponse:
        request_id = getattr(request.state, "request_id", "")
        if payload.model and payload.model != state.settings.ollama_model:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="overriding the model is not permitted on this endpoint",
            )
        with metrics.track_latency("/v1/chat"):
            outcome = await service.complete(payload)

        conversation_id_var.set(outcome.conversation_id)
        _record_outcome_metrics(outcome, state)
        return service.to_response(
            outcome, request_id=request_id, versions=state.version_dict()
        )

    @application.post("/v1/chat/stream", tags=["chat"])
    async def chat_stream(
        request: Request,
        payload: ChatRequest,
        state: Annotated[AppState, Depends(get_state)],
        service: Annotated[ChatService, Depends(get_chat_service)],
        _client: Annotated[str, Depends(require_client)],
    ) -> StreamingResponse:
        request_id = getattr(request.state, "request_id", "")

        async def event_stream() -> AsyncIterator[bytes]:
            try:
                async for event in service.stream(payload, request_id=request_id):
                    if event.type == "done":
                        metrics.record_request("/v1/chat/stream", 200)
                    body = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
                    yield f"event: {event.type}\ndata: {body}\n\n".encode()
            except CapacityExceeded as exc:
                metrics.record_error("/v1/chat/stream", "capacity", ollama=True)
                payload_json = json.dumps(
                    {"type": "error", "error": "capacity_exceeded", "detail": str(exc)},
                    ensure_ascii=False,
                )
                yield f"event: error\ndata: {payload_json}\n\n".encode()
            except OllamaError as exc:
                metrics.record_error("/v1/chat/stream", "ollama_error", ollama=True)
                log.error("api.stream_failed", extra={"error": str(exc)})
                payload_json = json.dumps(
                    {"type": "error", "error": "generation_failed"}, ensure_ascii=False
                )
                yield f"event: error\ndata: {payload_json}\n\n".encode()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # nginx must not buffer SSE
            },
        )

    @application.delete("/v1/conversations/{conversation_id}", tags=["chat"])
    async def forget_conversation(
        conversation_id: str,
        state: Annotated[AppState, Depends(get_state)],
        _client: Annotated[str, Depends(require_client)],
    ) -> dict[str, Any]:
        """Erase a conversation immediately (privacy / right to be forgotten)."""
        deleted = state.conversations.delete(conversation_id)
        return {"conversation_id": conversation_id, "deleted": deleted}

    # --- retrieval ---------------------------------------------------------
    @application.post("/v1/rag/search", response_model=SearchResponse, tags=["rag"])
    async def rag_search(
        payload: SearchRequest,
        rag: Annotated[RagService, Depends(get_rag_service)],
        _client: Annotated[str, Depends(require_client)],
    ) -> SearchResponse:
        filters = RetrievalFilters(
            product_id=payload.product_id,
            category=payload.category,
            include_expired=payload.include_expired,
            status=payload.status or ["active"],
        )
        result = await rag.search(payload.query, filters=filters, top_k=payload.top_k)
        metrics.observe_retrieval(
            result.latency_ms / 1000.0, empty=result.is_empty, reason="low_confidence"
        )
        return SearchResponse(
            query=payload.query,
            results=[c.model_dump(mode="json") for c in result.chunks],
            confidence=str(result.confidence),
            confidence_score=result.confidence_score,
            conflicts=[c.model_dump(mode="json") for c in result.conflicts],
            latency_ms=result.latency_ms,
            index_version=result.index_version,
            dropped_for_injection=result.dropped_for_injection,
        )

    # --- models ------------------------------------------------------------
    @application.get("/v1/models", response_model=ModelsResponse, tags=["ops"])
    async def models(state: Annotated[AppState, Depends(get_state)]) -> ModelsResponse:
        settings = state.settings
        try:
            available = await state.ollama.list_models()
        except OllamaError:
            available = []
        by_name = {m.get("name", ""): m for m in available}

        infos: list[ModelInfo] = []
        for name, role in ((settings.ollama_model, "primary"), (settings.ollama_fallback_model, "fallback")):
            if not name:
                continue
            raw = by_name.get(name, {})
            details = raw.get("details", {}) or {}
            infos.append(
                ModelInfo(
                    name=name,
                    role=role,
                    available=name in by_name,
                    parameter_size=str(details.get("parameter_size", "")),
                    quantization=str(details.get("quantization_level", "")),
                    size_bytes=int(raw.get("size", 0) or 0),
                )
            )
        return ModelsResponse(
            default=settings.ollama_model,
            fallback=settings.ollama_fallback_model,
            models=infos,
            versions=state.version_dict(),
            generation_defaults=settings.generation_options(),
        )

    # --- admin -------------------------------------------------------------
    @application.post(
        "/v1/admin/reindex",
        response_model=ReindexResponse,
        tags=["admin"],
        dependencies=[Depends(require_admin)],
    )
    async def admin_reindex(
        payload: ReindexRequest, state: Annotated[AppState, Depends(get_state)]
    ) -> ReindexResponse:
        import asyncio  # noqa: PLC0415 - only needed on this path

        from common.paths import PROJECT_ROOT, resolve_under_root
        from rag.ingestion import IngestionSettings
        from rag.reindex import reindex as run_reindex

        rag_config = state.settings.rag_policy()
        settings = IngestionSettings.from_config(rag_config)
        settings.index_root = state.settings.index_root_path

        default_input = PROJECT_ROOT / "data" / "interim" / "company_records.jsonl"
        # An admin-supplied path is still untrusted input: a traversal attempt is
        # a client error, not a server error.
        try:
            input_path = (
                resolve_under_root(payload.input_path, PROJECT_ROOT)
                if payload.input_path
                else default_input
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="input_path must be inside the project directory"
            ) from exc
        if not input_path.is_file():
            raise HTTPException(status_code=400, detail=f"input not found: {input_path}")

        if payload.dry_run:
            return ReindexResponse(
                index_version="(dry-run)",
                activated=False,
                regression_passed=True,
                message=f"would rebuild from {input_path}",
            )

        reindex_cfg = (rag_config.get("reindex") or {}) if isinstance(rag_config, dict) else {}
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: run_reindex(
                input_path,
                settings=settings,
                index_version=payload.index_version,
                golden_path=reindex_cfg.get("golden"),
                activate_on_success=payload.activate,
                min_recall=float(reindex_cfg.get("min_recall", 0.80)),
                keep_versions=int(reindex_cfg.get("keep_versions", 3)),
            ),
        )
        if result.activated and state.rag is not None:
            state.rag.reload()
            metrics.set_index_info(state.version_dict())

        return ReindexResponse(
            index_version=result.index_version,
            activated=result.activated,
            regression_passed=result.regression_passed,
            documents=int(result.manifest.get("documents", 0)),
            chunks=int(result.manifest.get("chunks", 0)),
            regression=result.regression,
            message=(
                "activated and hot-swapped"
                if result.activated
                else "built; not activated (pass activate=true, and the regression gate must pass)"
            ),
        )

    @application.post("/v1/admin/reload", tags=["admin"], dependencies=[Depends(require_admin)])
    async def admin_reload(state: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
        """Hot-swap to whatever ``data/index/ACTIVE`` currently points at."""
        if state.rag is None:
            raise HTTPException(status_code=503, detail="RAG is disabled")
        ok = state.rag.reload()
        state.chat.prompts.clear_cache()
        metrics.set_index_info(state.version_dict())
        return {"reloaded": ok, "index_version": state.rag.index_version, "prompts_reloaded": True}

    @application.get("/v1/admin/diagnostics", tags=["admin"], dependencies=[Depends(require_admin)])
    async def admin_diagnostics(state: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
        return {
            "versions": state.version_dict(),
            "uptime_seconds": round(state.uptime_seconds, 1),
            "settings": state.settings.redacted(),
            "ollama": state.ollama.stats(),
            "rag": state.rag.stats() if state.rag else {"enabled": False},
            "conversations": state.conversations.stats(),
            "host": host_stats(state.settings.index_root_path),
        }


def _record_outcome_metrics(outcome: Any, state: AppState) -> None:
    if outcome.usage:
        ttft_ms = float(outcome.usage.get("time_to_first_token_ms", 0) or 0)
        if ttft_ms:
            metrics.observe_ttft(ttft_ms / 1000.0)
        metrics.observe_tokens(
            outcome.model or state.settings.ollama_model,
            int(outcome.usage.get("completion_tokens", 0) or 0),
        )
    if outcome.retrieval is not None:
        metrics.observe_retrieval(
            outcome.retrieval.latency_ms / 1000.0,
            empty=outcome.retrieval.is_empty,
            reason="low_confidence" if outcome.retrieval.is_empty else "none",
        )
    if outcome.escalation_required:
        metrics.record_escalation(str(outcome.escalation_reason))
    if outcome.blocked_reason:
        metrics.record_guardrail_block(outcome.blocked_reason)
        if outcome.blocked_reason == "prompt_injection":
            metrics.record_injection_block("user_message")
    if ChatService.is_unknown_answer(outcome.answer):
        metrics.record_unknown_answer(outcome.intent)


app = create_app()
