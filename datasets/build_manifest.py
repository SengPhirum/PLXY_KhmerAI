#!/usr/bin/env python3
"""Rebuild dataset manifests and enforce the provenance rules (Phase 2).

    python datasets/build_manifest.py --root data/raw
    python datasets/build_manifest.py --root data/raw --check

Every data file under ``--root`` must have a manifest.  ``--check`` exits
non-zero when any file lacks provenance, when a checksum no longer matches, or
when an evaluation-only source has leaked into a training directory - the three
failure modes that let unreviewed or contaminated data reach a training run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hashing import sha256_file  # noqa: E402
from common.io import count_lines, read_json, write_json  # noqa: E402
from common.logging import get_logger  # noqa: E402
from common.paths import MANIFEST_DIR, ensure_dir  # noqa: E402

log = get_logger("datasets.manifest")

DATA_SUFFIXES = frozenset({".jsonl", ".json", ".txt", ".parquet", ".csv"})
EVALUATION_MARKERS = ("belebele", "flores", "/evaluation/", "eval_")


def scan(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in DATA_SUFFIXES and not p.name.startswith(".")
    )


def manifest_for(path: Path, *, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build (or refresh) the manifest for one data file."""
    stat = path.stat()
    manifest: dict[str, Any] = {
        "dataset": path.stem,
        "source": "",
        "revision": "",
        "download_date": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "license": "unknown",
        "language": "km",
        "subset": "",
        "raw_records": count_lines(path) if path.suffix == ".jsonl" else 0,
        "raw_bytes": stat.st_size,
        "sha256": sha256_file(str(path)),
        "intended_use": "unspecified",
        "commercial_review_status": "review_required",
        "path": str(path),
    }
    if existing:
        # Preserve human-entered provenance; refresh only the derived fields.
        for key in (
            "source",
            "revision",
            "license",
            "subset",
            "intended_use",
            "commercial_review_status",
            "download_date",
            "evaluation_only",
            "notes",
        ):
            if key in existing and existing[key] not in ("", None):
                manifest[key] = existing[key]
    manifest.setdefault("evaluation_only", any(m in str(path).lower() for m in EVALUATION_MARKERS))
    return manifest


def build(root: Path, *, check_only: bool = False) -> dict[str, Any]:
    files = scan(root)
    ensure_dir(MANIFEST_DIR)

    problems: list[str] = []
    manifests: list[dict[str, Any]] = []

    for path in files:
        manifest_path = MANIFEST_DIR / f"{path.stem}.json"
        existing = read_json(manifest_path) if manifest_path.is_file() else None

        if existing is None:
            problems.append(f"no manifest for {path} (run without --check to create one)")
        elif existing.get("sha256") and existing["sha256"] != sha256_file(str(path)):
            problems.append(
                f"checksum mismatch for {path}: the file changed since its manifest was written"
            )

        manifest = manifest_for(path, existing=existing)

        # Leakage rule: an evaluation-only source must never sit in a training path.
        in_training_path = "/raw/public/" in str(path).replace("\\", "/") or "/cleaned/" in str(
            path
        ).replace("\\", "/")
        if manifest.get("evaluation_only") and in_training_path:
            problems.append(
                f"EVALUATION LEAKAGE: {path} is marked evaluation_only but lives in a training path"
            )
        if (
            manifest.get("commercial_review_status") == "prohibited_for_commercial"
            and in_training_path
        ):
            problems.append(f"PROHIBITED SOURCE in a training path: {path}")

        manifests.append(manifest)
        if not check_only:
            write_json(manifest_path, manifest)

    summary = {
        "root": str(root),
        "files": len(files),
        "manifests": len(manifests),
        "total_records": sum(m["raw_records"] for m in manifests),
        "total_bytes": sum(m["raw_bytes"] for m in manifests),
        "by_review_status": _count(manifests, "commercial_review_status"),
        "evaluation_only_files": sum(1 for m in manifests if m.get("evaluation_only")),
        "problems": problems,
        "ok": not problems,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if not check_only:
        write_json(MANIFEST_DIR / "_index.json", {"summary": summary, "manifests": manifests})
    return summary


def _count(manifests: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for manifest in manifests:
        value = str(manifest.get(key, "unknown"))
        out[value] = out.get(value, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python datasets/build_manifest.py")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--check", action="store_true", help="verify only; do not write manifests")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    summary = build(root, check_only=args.check)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["problems"]:
        print("\nPROVENANCE PROBLEMS:", file=sys.stderr)
        for problem in summary["problems"]:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
