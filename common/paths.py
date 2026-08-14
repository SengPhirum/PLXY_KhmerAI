"""Canonical filesystem locations.

Every module resolves paths through here so that the repository can be relocated
(for example to `/usr/local/opt/khmerai` on the Mac Studio) by setting a single
environment variable, `KHMERAI_PROJECT_ROOT`.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_ROOT",
    "PROMPT_DIR",
    "REPORT_DIR",
    "ensure_dir",
    "resolve_under_root",
]


def _detect_project_root() -> Path:
    env = os.environ.get("KHMERAI_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # common/paths.py -> common/ -> <repo root>
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _detect_project_root()
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
PROMPT_DIR: Path = PROJECT_ROOT / "prompts"
REPORT_DIR: Path = PROJECT_ROOT / "reports"
EVAL_GOLDEN_DIR: Path = PROJECT_ROOT / "evaluation" / "golden"
EVAL_REPORT_DIR: Path = PROJECT_ROOT / "evaluation" / "reports"


def _detect_data_root() -> Path:
    env = os.environ.get("KHMERAI_DATA_ROOT")
    if env:
        p = Path(env).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    return PROJECT_ROOT / "data"


DATA_ROOT: Path = _detect_data_root()
RAW_DIR: Path = DATA_ROOT / "raw"
INTERIM_DIR: Path = DATA_ROOT / "interim"
CLEANED_DIR: Path = DATA_ROOT / "cleaned"
DEDUPED_DIR: Path = DATA_ROOT / "deduped"
SFT_DIR: Path = DATA_ROOT / "sft"
PREFERENCE_DIR: Path = DATA_ROOT / "preference"
EVALUATION_DIR: Path = DATA_ROOT / "evaluation"
MANIFEST_DIR: Path = DATA_ROOT / "manifests"
INDEX_ROOT: Path = DATA_ROOT / "index"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) if missing and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_under_root(candidate: str | os.PathLike[str], root: Path) -> Path:
    """Resolve ``candidate`` and guarantee it stays inside ``root``.

    Used by every path that can be influenced by user input (document ingestion,
    admin reindex requests, backup restore).  Raises ``ValueError`` on traversal.
    """
    root_resolved = Path(root).expanduser().resolve()
    target = Path(candidate).expanduser()
    if not target.is_absolute():
        target = root_resolved / target
    target = target.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"path escapes permitted root: {candidate!r} not under {root_resolved}")
    return target
