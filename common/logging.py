"""Structured logging with automatic secret/PII redaction and request IDs.

Every log record is emitted as a single JSON object (Phase 19 requirement) with
a stable field set:

``ts, level, logger, msg, request_id, conversation_id, event, <extras...>``

The formatter runs :func:`security.data_redaction.redact` over the rendered
message and over every string extra, so a stack trace containing an API key or a
customer phone number is scrubbed before it reaches disk.  This is enforced in
``security/tests/test_logging_redaction.py``.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from security.data_redaction import redact

__all__ = [
    "JsonFormatter",
    "bind_request",
    "configure_logging",
    "conversation_id_var",
    "get_logger",
    "new_request_id",
    "request_id_var",
]

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
conversation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "conversation_id", default="-"
)

_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

_CONFIGURED = False


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def bind_request(
    request_id: str | None = None, conversation_id: str | None = None
) -> Iterator[str]:
    """Bind a request/conversation ID for the duration of the block."""
    rid = request_id or new_request_id()
    token_r = request_id_var.set(rid)
    token_c = conversation_id_var.set(conversation_id or "-")
    try:
        yield rid
    finally:
        request_id_var.reset(token_r)
        conversation_id_var.reset(token_c)


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON line, redacting secrets and PII."""

    def __init__(self, *, service: str = "khmer-support-llm", redact_output: bool = True) -> None:
        super().__init__()
        self.service = service
        self.redact_output = redact_output

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        conversation = conversation_id_var.get()
        if conversation and conversation != "-":
            payload["conversation_id"] = conversation

        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = _coerce(value)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        line = json.dumps(payload, ensure_ascii=False, default=str)
        if self.redact_output:
            line = redact(line).text
        return line


class ConsoleFormatter(logging.Formatter):
    """Human-readable format for local development (still redacted)."""

    _FMT = "%(asctime)s %(levelname)-7s %(name)-28s [%(request_id)s] %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = request_id_var.get()  # type: ignore[attr-defined]
        rendered = super().format(record)
        extras = {
            k: _coerce(v)
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_") and k != "request_id"
        }
        if extras:
            rendered += " " + " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
        return redact(rendered).text


def _coerce(value: Any) -> Any:
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, dict | list | tuple):
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    return str(value)


def configure_logging(
    level: str | int | None = None,
    *,
    fmt: str | None = None,
    service: str = "khmer-support-llm",
    force: bool = False,
) -> None:
    """Install the root handler.  Idempotent unless ``force`` is set."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved_level = level or os.environ.get("KHMERAI_LOG_LEVEL", "INFO")
    resolved_fmt = (fmt or os.environ.get("KHMERAI_LOG_FORMAT", "json")).lower()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        JsonFormatter(service=service) if resolved_fmt == "json" else ConsoleFormatter()
    )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(
        resolved_level if isinstance(resolved_level, int) else str(resolved_level).upper()
    )

    # Uvicorn duplicates records through its own handlers; route them to ours.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore"):
        logger = logging.getLogger(noisy)
        logger.handlers.clear()
        logger.propagate = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root handler on first use."""
    configure_logging()
    return logging.getLogger(name)
