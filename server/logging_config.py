"""Uvicorn/launchd logging configuration.

``common.logging`` owns the formatter and the redaction rule.  This module is the
thin adapter that (a) hands uvicorn a dictConfig which routes everything through
that formatter, and (b) adds an optional rotating file handler for the launchd
deployment, where stdout goes to a plist-managed log path.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any

from common.logging import ConsoleFormatter, JsonFormatter, configure_logging

__all__ = ["uvicorn_log_config", "add_file_handler", "setup_from_settings"]


def uvicorn_log_config(level: str = "INFO", fmt: str = "json") -> dict[str, Any]:
    """dictConfig for ``uvicorn --log-config``.

    Uvicorn installs its own colourised handlers by default, which would bypass
    redaction.  This replaces them so that every line - including access logs -
    goes through :class:`common.logging.JsonFormatter`.
    """
    formatter = (
        "server.logging_config._json_formatter"
        if fmt == "json"
        else "server.logging_config._console_formatter"
    )
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"()": formatter}},
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": level, "propagate": False},
            # Access logs duplicate `request.completed` from RequestContextMiddleware,
            # which carries the request ID and the duration - keep only ours.
            "uvicorn.access": {"handlers": [], "level": "WARNING", "propagate": False},
        },
        "root": {"handlers": ["default"], "level": level},
    }


def _json_formatter() -> logging.Formatter:
    return JsonFormatter()


def _console_formatter() -> logging.Formatter:
    return ConsoleFormatter()


def add_file_handler(
    path: str | Path,
    *,
    level: str = "INFO",
    fmt: str = "json",
    max_bytes: int = 50 * 1024 * 1024,
    backups: int = 5,
) -> logging.Handler:
    """Attach a size-rotating file handler (used by the launchd deployment)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        target, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    return handler


def setup_from_settings(settings: Any, *, log_file: str | Path | None = None) -> None:
    """Configure logging from a :class:`server.config.Settings`."""
    configure_logging(settings.log_level, fmt=settings.log_format, force=True)
    if log_file:
        add_file_handler(log_file, level=settings.log_level, fmt=settings.log_format)
