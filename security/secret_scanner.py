"""Repository and dataset secret scanner.

Runs in three places:

* ``scripts/lint.sh`` - blocks a commit that adds a credential to the repo.
* ``scripts/prepare_all_data.sh`` - refuses to build a cloud-training bundle
  that contains one.
* CI (``.github/workflows/ci.yml``) - the security stage.

Entropy is used only as a *secondary* signal.  A high-entropy string alone is
not reported (Khmer text, base64 images and UUIDs would all trip it); it is
reported when it also sits next to a secret-ish assignment keyword.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SecretFinding",
    "scan_text",
    "scan_bytes",
    "scan_path",
    "scan_repository",
    "shannon_entropy",
]

_SKIP_DIRS = frozenset(
    {
        ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", "data", "models", "checkpoints",
        "outputs", "htmlcov", "dist", "build", ".hypothesis",
    }
)
_SKIP_SUFFIXES = frozenset(
    {
        ".gguf", ".safetensors", ".bin", ".pt", ".ckpt", ".png", ".jpg", ".jpeg",
        ".gif", ".pdf", ".zip", ".gz", ".tar", ".woff", ".woff2", ".ico", ".so",
        ".dylib", ".parquet", ".arrow", ".npy", ".npz",
    }
)
_MAX_FILE_BYTES = 4 * 1024 * 1024

# (rule, severity, pattern)
_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("aws_access_key", "critical", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    ("aws_secret_key", "critical", re.compile(r"(?i)aws_?secret_?access_?key\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})")),
    ("github_token", "critical", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("gitlab_token", "critical", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b")),
    ("slack_token", "critical", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("openai_key", "critical", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_key", "critical", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("hf_token", "critical", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("google_api_key", "critical", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("private_key", "critical", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("jwt", "high", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("connection_string", "high", re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:@/]+:[^\s:@/]+@")),
    (
        "assigned_secret",
        "high",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|pwd)\b\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']"
        ),
    ),
    ("bearer_literal", "medium", re.compile(r"(?i)\bauthorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._\-]{16,}")),
)

# Values that look like secrets but are deliberate placeholders.
_PLACEHOLDER = re.compile(
    r"(?i)^(?:change[_-]?me|placeholder|example|dummy|test|sample|your[_-]?\w+|x{4,}|\*{4,}|<[^>]+>|\$\{[^}]+\}|redacted|none|null|todo)",
)
_ALLOW_COMMENT = re.compile(r"(?i)#\s*(?:nosec|secret-scanner:\s*ignore|pragma:\s*allowlist secret)")


@dataclass(slots=True, frozen=True)
class SecretFinding:
    rule: str
    severity: str
    path: str
    line: int
    excerpt: str
    entropy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - human output
        return f"{self.severity.upper():8s} {self.rule:20s} {self.path}:{self.line}  {self.excerpt}"


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _mask(value: str, keep: int = 4) -> str:
    stripped = value.strip()
    if len(stripped) <= keep * 2:
        return "*" * len(stripped)
    return f"{stripped[:keep]}{'*' * min(12, len(stripped) - keep * 2)}{stripped[-keep:]}"


def scan_text(text: str, *, path: str = "<memory>") -> list[SecretFinding]:
    """Scan a string, returning one finding per (rule, line)."""
    findings: list[SecretFinding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _ALLOW_COMMENT.search(line):
            continue
        for rule, severity, pattern in _RULES:
            match = pattern.search(line)
            if not match:
                continue
            captured = match.group(1) if match.groups() else match.group(0)
            if _PLACEHOLDER.match(captured.strip()):
                continue
            entropy = shannon_entropy(captured)
            # Assigned secrets need corroborating entropy; branded tokens do not.
            if rule in ("assigned_secret", "bearer_literal") and entropy < 3.0:
                continue
            findings.append(
                SecretFinding(
                    rule=rule,
                    severity=severity,
                    path=path,
                    line=lineno,
                    excerpt=_mask(captured),
                    entropy=round(entropy, 2),
                )
            )
    return findings


def scan_bytes(payload: bytes, *, path: str = "<memory>") -> list[SecretFinding]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return scan_text(text, path=path)


def scan_path(path: str | Path) -> list[SecretFinding]:
    target = Path(path)
    if not target.is_file():
        return []
    if target.suffix.lower() in _SKIP_SUFFIXES:
        return []
    try:
        if target.stat().st_size > _MAX_FILE_BYTES:
            return []
        payload = target.read_bytes()
    except OSError:
        return []
    return scan_bytes(payload, path=str(target))


def _iter_files(root: Path, extra_skip_dirs: Iterable[str] = ()) -> Iterator[Path]:
    skip = _SKIP_DIRS | set(extra_skip_dirs)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        yield path


def scan_repository(
    root: str | Path = ".", *, extra_skip_dirs: Iterable[str] = ()
) -> list[SecretFinding]:
    """Scan an entire tree, skipping VCS metadata, virtualenvs, data and binaries."""
    findings: list[SecretFinding] = []
    for path in _iter_files(Path(root), extra_skip_dirs):
        findings.extend(scan_path(path))
    return findings


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m security.secret_scanner", description="Scan for committed secrets"
    )
    parser.add_argument("paths", nargs="*", default=["."], help="files or directories to scan")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--fail-on",
        choices=("critical", "high", "medium"),
        default="high",
        help="minimum severity that makes the scan fail (default: high)",
    )
    args = parser.parse_args(argv)

    order = {"medium": 0, "high": 1, "critical": 2}
    findings: list[SecretFinding] = []
    for raw in args.paths:
        path = Path(raw)
        findings.extend(scan_repository(path) if path.is_dir() else scan_path(path))

    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
    else:
        for finding in findings:
            print(finding)
        print(f"\n{len(findings)} finding(s)")

    threshold = order[args.fail_on]
    return 1 if any(order[f.severity] >= threshold for f in findings) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
