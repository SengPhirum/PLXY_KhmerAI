"""Independent version tracking for application, prompt, model, index and dataset.

Implements §37 of the implementation specification: each artefact is versioned
separately and every version is surfaced by ``GET /v1/models`` and
``GET /health`` so that a production incident can be tied to an exact
combination without reading logs.

The source of truth is ``configs/base.yaml -> versions``.  The knowledge index
version is *not* stored there: it is read live from the active index manifest,
because the index changes without a code deploy.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common.config import ConfigError, load_config
from common.paths import PROJECT_ROOT

__all__ = ["PlatformVersions", "git_commit", "load_versions"]

_UNKNOWN = "unknown"


def git_commit(short: bool = True) -> str:
    """Return the current commit SHA, or ``"unknown"`` outside a git checkout.

    Recorded in every training run, evaluation report and release bundle so a
    result can always be traced back to the exact code that produced it.
    """
    env_commit = os.environ.get("KHMERAI_GIT_COMMIT")
    if env_commit:
        return env_commit
    argv = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
    if not short:
        argv = ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(  # noqa: S603
            argv,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return _UNKNOWN
    if result.returncode != 0:
        return _UNKNOWN
    return result.stdout.strip() or _UNKNOWN


@dataclass(frozen=True, slots=True)
class PlatformVersions:
    """Every independently deployable version in the platform."""

    app: str = "0.0.0"
    prompt: str = "0.0"
    model: str = _UNKNOWN
    adapter: str = _UNKNOWN
    dataset: str = _UNKNOWN
    knowledge_index: str = _UNKNOWN
    code_commit: str = field(default_factory=git_commit)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_index(self, index_version: str) -> PlatformVersions:
        return PlatformVersions(
            app=self.app,
            prompt=self.prompt,
            model=self.model,
            adapter=self.adapter,
            dataset=self.dataset,
            knowledge_index=index_version,
            code_commit=self.code_commit,
        )

    def summary(self) -> str:
        return (
            f"app={self.app} model={self.model} adapter={self.adapter} "
            f"prompt={self.prompt} index={self.knowledge_index} "
            f"dataset={self.dataset} commit={self.code_commit}"
        )


def _read_active_index_version(index_root: Path) -> str:
    """Resolve the ACTIVE pointer written by ``rag.reindex``."""
    pointer = index_root / "ACTIVE"
    try:
        if pointer.is_symlink():
            return Path(os.readlink(pointer)).name
        if pointer.is_file():
            return pointer.read_text(encoding="utf-8").strip() or _UNKNOWN
    except OSError:
        return _UNKNOWN
    return _UNKNOWN


def load_versions(
    config_path: str | os.PathLike[str] = "configs/base.yaml",
    *,
    index_root: str | os.PathLike[str] | None = None,
) -> PlatformVersions:
    """Build a :class:`PlatformVersions` from configuration + live index state."""
    try:
        config = load_config(config_path)
    except ConfigError:
        config = {}
    raw = config.get("versions", {}) if isinstance(config, dict) else {}

    root = Path(index_root) if index_root else PROJECT_ROOT / "data" / "index"
    index_version = _read_active_index_version(root)
    if index_version == _UNKNOWN:
        index_version = str(raw.get("knowledge_index", _UNKNOWN))

    return PlatformVersions(
        app=str(raw.get("app", "0.0.0")),
        prompt=str(raw.get("prompt", "0.0")),
        model=str(raw.get("model", _UNKNOWN)),
        adapter=str(raw.get("adapter", _UNKNOWN)),
        dataset=str(raw.get("dataset", _UNKNOWN)),
        knowledge_index=index_version,
        code_commit=git_commit(),
    )
