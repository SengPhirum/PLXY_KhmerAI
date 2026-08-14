"""Request middleware: IDs, size limits, rate limiting, timing, security headers."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from common.logging import conversation_id_var, get_logger, new_request_id, request_id_var
from server import metrics
from server.rate_limit import RateLimiter

log = get_logger(__name__)

__all__ = [
    "RequestContextMiddleware",
    "RequestSizeLimitMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
]

_EXEMPT_PATHS = frozenset({"/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, bind it to the log context, and time the request."""

    def __init__(self, app: ASGIApp, *, header: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header = header

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(self.header, "")
        # Never trust a client-supplied ID verbatim - it lands in logs.
        request_id = (
            incoming[:64]
            if incoming and incoming.replace("-", "").replace("_", "").isalnum()
            else new_request_id()
        )
        token_r = request_id_var.set(request_id)
        token_c = conversation_id_var.set("-")
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            log.exception(
                "request.unhandled_error",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(elapsed, 1),
                },
            )
            metrics.record_error(request.url.path, "unhandled")
            raise
        finally:
            request_id_var.reset(token_r)
            conversation_id_var.reset(token_c)

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[self.header] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        if request.url.path not in _EXEMPT_PATHS:
            metrics.record_request(request.url.path, response.status_code)
            log.info(
                "request.completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed_ms, 1),
                },
            )
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies before they are parsed (Phase 16)."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = 65_536) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    return self._too_large(request)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "bad_request", "detail": "invalid Content-Length header"},
                )
        elif request.headers.get("transfer-encoding", "").lower() == "chunked":
            # No declared length: read once with a cap, then hand the buffered
            # body to the app so the downstream handler still sees it.
            body = await request.body()
            if len(body) > self.max_bytes:
                return self._too_large(request)
        return await call_next(request)

    def _too_large(self, request: Request) -> JSONResponse:
        log.warning("request.too_large", extra={"path": request.url.path, "limit": self.max_bytes})
        metrics.record_error(request.url.path, "payload_too_large")
        return JSONResponse(
            status_code=413,
            content={
                "error": "payload_too_large",
                "detail": f"request body exceeds {self.max_bytes} bytes",
            },
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiting keyed by API key, else by client IP."""

    def __init__(self, app: ASGIApp, *, limiter: RateLimiter, exempt: frozenset[str] = _EXEMPT_PATHS) -> None:
        super().__init__(app)
        self.limiter = limiter
        self.exempt = exempt

    @staticmethod
    def _client_key(request: Request) -> str:
        api_key = request.headers.get("X-API-Key") or ""
        if api_key:
            return f"key:{api_key[:32]}"
        # X-Forwarded-For is only trustworthy behind the documented reverse proxy
        # (deployment/nginx/nginx.conf sets it); take the left-most entry.
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()[:45]}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in self.exempt or request.method == "OPTIONS":
            return await call_next(request)

        decision = self.limiter.check(self._client_key(request))
        if not decision.allowed:
            metrics.record_rate_limit("client")
            log.warning(
                "request.rate_limited",
                extra={"path": request.url.path, "retry_after": round(decision.retry_after, 1)},
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": "too many requests; please slow down",
                    "retry_after_seconds": round(decision.retry_after, 1),
                },
                headers=decision.headers(),
            )

        response = await call_next(request)
        for header, value in decision.headers().items():
            response.headers.setdefault(header, value)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative security headers for a browser-facing deployment."""

    _HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        # The API returns JSON and SSE only; nothing should ever be rendered.
        "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    }

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for header, value in self._HEADERS.items():
            response.headers.setdefault(header, value)
        return response
