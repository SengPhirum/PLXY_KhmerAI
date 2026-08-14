"""Crash-safe file IO helpers.

The knowledge-update pipeline (Phase 21) requires that a partially written index
or manifest can never replace a good one, so every writer here goes through a
temporary file in the *same directory* followed by ``os.replace`` (atomic on
POSIX, including APFS).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    "append_jsonl",
    "atomic_symlink",
    "atomic_write_bytes",
    "atomic_write_text",
    "count_lines",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(path: str | os.PathLike[str], text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def write_json(path: str | os.PathLike[str], obj: Any, *, indent: int = 2) -> Path:
    payload = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=False, default=str)
    return atomic_write_text(path, payload + "\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> int:
    """Write ``rows`` as JSON Lines atomically.  Returns the number of records."""
    count = 0
    chunks: list[str] = []
    for row in rows:
        chunks.append(json.dumps(row, ensure_ascii=False, default=str))
        count += 1
    atomic_write_text(path, "\n".join(chunks) + ("\n" if chunks else ""))
    return count


def append_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> int:
    """Append records to a JSONL file (not atomic - use for streaming pipelines)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def read_jsonl(path: str | os.PathLike[str], *, skip_invalid: bool = False) -> Iterator[Any]:
    """Stream a JSONL file.

    ``skip_invalid`` tolerates truncated final lines produced by an interrupted
    writer; it is off by default so data corruption is loud during ingestion.
    """
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_invalid:
                    continue
                raise ValueError(f"{target}:{lineno}: invalid JSON ({exc.msg})") from exc


def count_lines(path: str | os.PathLike[str]) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    with target.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def atomic_symlink(link: str | os.PathLike[str], target: str | os.PathLike[str]) -> Path:
    """Point ``link`` at ``target`` atomically (used by the index ACTIVE pointer)."""
    link_path = Path(link)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = link_path.with_name(f".{link_path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    os.symlink(str(target), str(tmp))
    os.replace(tmp, link_path)
    return link_path


@contextmanager
def temporary_directory(prefix: str = "khmerai-") -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix) as name:
        yield Path(name)
