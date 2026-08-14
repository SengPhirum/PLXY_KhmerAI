"""YAML configuration loading with layering, environment expansion and includes.

Design rules
------------
* ``configs/base.yaml`` holds every platform-wide default.  Specialised files
  (``configs/rag/retrieval.yaml`` and friends) declare ``extends: ../base.yaml``
  and only override what differs, so a value is defined in exactly one place.
* ``${VAR}`` and ``${VAR:-default}`` inside string values are expanded from the
  process environment at load time.  A missing variable with no default raises,
  which turns a mis-configured deployment into a startup failure instead of a
  silent production surprise.
* Loaded configs are cached by (path, mtime) so hot paths can call
  ``load_config`` freely.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from common.paths import PROJECT_ROOT

__all__ = [
    "ConfigError",
    "load_config",
    "load_yaml",
    "deep_merge",
    "expand_env",
    "get_path",
    "clear_cache",
]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_CACHE: dict[tuple[str, float], dict[str, Any]] = {}
_MAX_EXTENDS_DEPTH = 8


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing, malformed or incomplete."""


def clear_cache() -> None:
    _CACHE.clear()


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            env = os.environ.get(name)
            if env is not None:
                return env
            if default is not None:
                return default
            raise ConfigError(
                f"environment variable {name!r} is referenced by configuration but is not set "
                f"(add it to .env or provide a ${{{name}:-default}})"
            )

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either.

    Mappings merge key-by-key; every other type (including lists) is replaced.
    Replacing lists is deliberate - a partially overridden list of retrieval
    filters is almost always a bug.
    """
    out = deepcopy(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = deep_merge(current, value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse a single YAML file (no ``extends`` handling, no env expansion)."""
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.is_file():
        raise ConfigError(f"configuration file not found: {target}")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {target}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {target} must be a mapping, got {type(data).__name__}")
    return data


def load_config(
    path: str | os.PathLike[str],
    *,
    overrides: dict[str, Any] | None = None,
    expand: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load a config file, following ``extends`` chains and expanding env vars."""
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.is_file():
        raise ConfigError(f"configuration file not found: {target}")

    cache_key = (str(target.resolve()), target.stat().st_mtime)
    if use_cache and overrides is None and expand and cache_key in _CACHE:
        return deepcopy(_CACHE[cache_key])

    merged = _load_with_extends(target, depth=0, seen=[])
    if overrides:
        merged = deep_merge(merged, overrides)
    if expand:
        merged = expand_env(merged)
    if use_cache and overrides is None and expand:
        _CACHE[cache_key] = deepcopy(merged)
    return merged


def _load_with_extends(target: Path, depth: int, seen: list[Path]) -> dict[str, Any]:
    if depth > _MAX_EXTENDS_DEPTH:
        raise ConfigError(f"`extends` chain deeper than {_MAX_EXTENDS_DEPTH} starting at {target}")
    resolved = target.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in [*seen, resolved])
        raise ConfigError(f"circular `extends` chain: {chain}")

    data = load_yaml(target)
    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data

    if not isinstance(parent_ref, str):
        raise ConfigError(f"`extends` in {target} must be a string path")
    parent_path = (target.parent / parent_ref).resolve()
    parent = _load_with_extends(parent_path, depth + 1, [*seen, resolved])
    return deep_merge(parent, data)


def get_path(config: dict[str, Any], dotted: str, default: Any = ...) -> Any:
    """Read ``config`` by dotted key, e.g. ``get_path(cfg, "retrieval.top_k")``."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is ...:
                raise ConfigError(f"missing required configuration key: {dotted!r}")
            return default
        node = node[part]
    return node
