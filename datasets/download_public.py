#!/usr/bin/env python3
"""Download the reviewed public Khmer datasets with reproducible manifests (Phase 2).

    python datasets/download_public.py --list
    python datasets/download_public.py --all --limit 50000
    python datasets/download_public.py --name khmer_wikipedia
    python datasets/download_public.py --all --dry-run     # no network

**Not executed in this environment** (network downloads of multi-GB corpora are
out of scope here).  ``--dry-run`` prints exactly what each source would fetch
and writes the manifest skeleton, which is what CI validates.

Every download writes ``data/manifests/<name>.json`` recording the dataset id,
revision, download date, licence, language, subset, record count, byte count,
SHA-256 and the commercial-review status.  Nothing enters training without one.

**Evaluation sources are downloaded to a separate directory** and are marked
``evaluation_only``; ``datasets/build_manifest.py`` fails the build if an
evaluation source ever appears under a training path (§Phase 2 leakage rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# `datasets` (Hugging Face) shares its name with this directory.  Running this
# file as a script puts THIS directory on sys.path[0], so the repository root is
# added explicitly and the HF package still resolves from site-packages.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.config import load_config  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.io import append_jsonl, write_json  # noqa: E402
from common.logging import get_logger  # noqa: E402
from common.paths import MANIFEST_DIR, RAW_DIR, ensure_dir  # noqa: E402

log = get_logger("datasets.download")

TRAINING_DIR = RAW_DIR / "public"
EVALUATION_DIR = _REPO_ROOT / "data" / "evaluation" / "public"

# Text field per dataset - HF schemas are not uniform.
TEXT_FIELDS = ("text", "content", "raw_content", "document", "sentence", "answer", "targets")


def _record_text(row: dict[str, Any], subset: str = "") -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    # Translation pairs (OPUS-100) carry a dict under `translation`.
    translation = row.get("translation")
    if isinstance(translation, dict):
        khmer = translation.get("km") or translation.get("khm")
        english = translation.get("en")
        if khmer and english:
            return f"{khmer}\t{english}"
        if khmer:
            return str(khmer)
    # Instruction datasets (Aya, khmer_question_answer).
    inputs = row.get("inputs") or row.get("question") or row.get("instruction")
    outputs = row.get("targets") or row.get("answer") or row.get("output")
    if inputs and outputs:
        return f"{inputs}\n{outputs}"
    return ""


def _sources(config_path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_config(config_path)
    section = config.get("datasets", {}) or {}
    return list(section.get("public_sources", [])), list(section.get("evaluation_sources", []))


def _manifest_skeleton(source: dict[str, Any], *, output: Path, evaluation: bool) -> dict[str, Any]:
    return {
        "dataset": source["name"],
        "source": source["hf_id"],
        "revision": source.get("revision", "unresolved"),
        "download_date": datetime.now(UTC).isoformat(),
        "license": source.get("license", "see the dataset card"),
        "language": "km",
        "subset": source.get("subset", ""),
        "split": source.get("split", "train"),
        "raw_records": 0,
        "raw_bytes": 0,
        "sha256": "",
        "intended_use": source.get("intended_use", "unspecified"),
        "commercial_review_status": source.get("commercial_review_status", "review_required"),
        "output_path": str(output),
        "evaluation_only": evaluation,
        "downloaded": False,
        "notes": "",
    }


def download_source(
    source: dict[str, Any],
    *,
    evaluation: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    streaming: bool = True,
) -> dict[str, Any]:
    """Download one source and write its manifest.  Returns the manifest."""
    directory = ensure_dir(EVALUATION_DIR if evaluation else TRAINING_DIR)
    output = directory / f"{source['name']}.jsonl"
    manifest = _manifest_skeleton(source, output=output, evaluation=evaluation)

    if dry_run:
        manifest["notes"] = (
            f"DRY RUN - would stream {source['hf_id']} "
            f"(subset={source.get('subset', '-')}, split={source.get('split', 'train')}"
            + (f", limit={limit}" if limit else "")
            + f") to {output}"
        )
        write_json(MANIFEST_DIR / f"{source['name']}.json", manifest)
        log.info("datasets.dry_run", extra={"dataset": source["name"], "output": str(output)})
        return manifest

    try:
        from datasets import load_dataset
    except ImportError:
        manifest["notes"] = (
            "the `datasets` package is not installed: pip install -r requirements/training.txt"
        )
        write_json(MANIFEST_DIR / f"{source['name']}.json", manifest)
        log.error("datasets.missing_dependency", extra={"dataset": source["name"]})
        return manifest

    log.info(
        "datasets.download.start",
        extra={"dataset": source["name"], "hf_id": source["hf_id"], "limit": limit},
    )
    try:
        stream = load_dataset(
            source["hf_id"],
            source.get("subset") or None,
            split=source.get("split", "train"),
            streaming=streaming,
            revision=source.get("revision") or None,
        )
    except Exception as exc:
        manifest["notes"] = f"download failed: {type(exc).__name__}: {exc}"
        write_json(MANIFEST_DIR / f"{source['name']}.json", manifest)
        log.error("datasets.download.failed", extra={"dataset": source["name"], "error": str(exc)})
        return manifest

    output.unlink(missing_ok=True)
    written = 0
    total_bytes = 0
    batch: list[dict[str, Any]] = []
    for row in stream:
        text = _record_text(dict(row), source.get("subset", ""))
        if not text.strip():
            continue
        batch.append(
            {
                "id": f"{source['name']}-{written:08d}",
                "source": source["name"],
                "text": text,
                "metadata": {"hf_id": source["hf_id"], "subset": source.get("subset", "")},
            }
        )
        total_bytes += len(text.encode("utf-8"))
        written += 1
        if len(batch) >= 1000:
            append_jsonl(output, batch)
            batch.clear()
        if limit is not None and written >= limit:
            break
    if batch:
        append_jsonl(output, batch)

    manifest.update(
        raw_records=written,
        raw_bytes=total_bytes,
        sha256=sha256_file(str(output)) if output.is_file() else "",
        downloaded=written > 0,
        notes=f"{written} records written" if written else "no usable records found",
    )
    write_json(MANIFEST_DIR / f"{source['name']}.json", manifest)
    log.info(
        "datasets.download.done",
        extra={"dataset": source["name"], "records": written, "mb": round(total_bytes / 1e6, 1)},
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python datasets/download_public.py")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--name", default=None, help="download a single source by name")
    parser.add_argument("--all", action="store_true", help="download every training source")
    parser.add_argument("--evaluation", action="store_true", help="download the evaluation sources")
    parser.add_argument("--limit", type=int, default=None, help="max records per source")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_sources")
    args = parser.parse_args(argv)

    training, evaluation = _sources(args.config)

    if args.list_sources:
        print(f"{'name':28s} {'hf_id':50s} {'subset':14s} {'review status'}")
        print("-" * 118)
        for source in training + evaluation:
            print(
                f"{source['name']:28s} {source['hf_id']:50s} "
                f"{source.get('subset', '-')!s:14s} {source.get('commercial_review_status', '?')}"
            )
        print(
            "\nEvaluation sources are written to data/evaluation/public/ and are NEVER "
            "used for training (§Phase 2 leakage rule)."
        )
        return 0

    selected: list[tuple[dict[str, Any], bool]] = []
    if args.name:
        for source in training:
            if source["name"] == args.name:
                selected.append((source, False))
        for source in evaluation:
            if source["name"] == args.name:
                selected.append((source, True))
        if not selected:
            print(f"unknown source {args.name!r}; use --list", file=sys.stderr)
            return 1
    else:
        if args.all:
            selected += [(s, False) for s in training]
        if args.evaluation or args.all:
            selected += [(s, True) for s in evaluation]
        if not selected:
            parser.error("choose --all, --evaluation, --name NAME or --list")

    manifests = [
        download_source(source, evaluation=is_eval, limit=args.limit, dry_run=args.dry_run)
        for source, is_eval in selected
    ]

    print(json.dumps(manifests, indent=2, ensure_ascii=False))
    unreviewed = [
        m["dataset"]
        for m in manifests
        if not m["evaluation_only"] and m["commercial_review_status"] != "approved"
    ]
    if unreviewed:
        print(
            "\nLICENSING: the following sources are NOT approved for commercial training:\n"
            + "\n".join(f"  - {name}" for name in unreviewed)
            + "\nRecord a legal decision in docs/dataset_provenance.md before using them.",
            file=sys.stderr,
        )
    print(
        "\nNext:\n    python datasets/build_manifest.py --root data/raw\n    python datasets/license_report.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
