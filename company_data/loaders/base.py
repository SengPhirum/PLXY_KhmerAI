"""Shared loader types and file-safety checks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_FILE_BYTES",
    "FileTooLargeError",
    "LoadedDocument",
    "LoaderDependencyError",
    "LoaderError",
    "UnsupportedFormatError",
    "check_file",
    "require",
]

MAX_FILE_BYTES = int(os.environ.get("KHMERAI_COMPANY_MAX_FILE_MB", "50")) * 1024 * 1024


class LoaderError(RuntimeError):
    """A document could not be read."""


class LoaderDependencyError(LoaderError):
    """A required optional parser is not installed."""


class UnsupportedFormatError(LoaderError):
    """The file extension has no registered loader."""


class FileTooLargeError(LoaderError):
    """The file exceeds the configured ingestion size limit."""


@dataclass(slots=True)
class LoadedDocument:
    """Raw output of a loader, before normalisation."""

    text: str
    source_path: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    part: str = ""  # sheet name, page range, CSV row key, ...
    loader: str = ""

    def __post_init__(self) -> None:
        if not self.title:
            self.title = Path(self.source_path).stem.replace("_", " ").strip()

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def require(module: str, extra: str = "rag") -> Any:
    """Import an optional parser or raise a helpful error."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LoaderDependencyError(
            f"{module!r} is required to read this format. Install it with:\n"
            f"    pip install -r requirements/{extra}.txt"
        ) from exc


def check_file(path: str | os.PathLike[str], *, max_bytes: int = MAX_FILE_BYTES) -> Path:
    """Validate a file before opening it.

    Guards against the ingestion threats in Phase 16: symlink escapes, device
    files, and oversized uploads that would exhaust memory during parsing.
    """
    target = Path(path)
    if not target.exists():
        raise LoaderError(f"file not found: {target}")
    if target.is_symlink():
        raise LoaderError(f"refusing to read a symlink: {target}")
    if not target.is_file():
        raise LoaderError(f"not a regular file: {target}")
    size = target.stat().st_size
    if size == 0:
        raise LoaderError(f"file is empty: {target}")
    if size > max_bytes:
        raise FileTooLargeError(
            f"{target} is {size / 1e6:.1f} MB, over the {max_bytes / 1e6:.0f} MB ingestion limit"
        )
    return target
