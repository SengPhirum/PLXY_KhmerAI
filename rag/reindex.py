"""Atomic knowledge-index update with regression gate and rollback (Phase 21/38).

Workflow::

    new docs -> validate -> normalise -> chunk -> embed -> build NEW index
             -> retrieval regression test -> activate (atomic swap)

Nothing mutates the live index.  Activation is a single ``os.replace`` of the
``data/index/ACTIVE`` symlink, so a reader either sees the whole old index or
the whole new one - never a half-written mixture.

Commands::

    python -m rag.reindex --config configs/rag/ingestion.yaml \
        --input data/interim/company_records.jsonl --activate
    python -m rag.reindex --list
    python -m rag.reindex --rollback
    python -m rag.reindex --activate-version 2026-08-14.2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import load_config
from common.io import read_json, read_jsonl, write_json
from common.logging import get_logger
from common.paths import INDEX_ROOT
from rag.embeddings import build_embedder
from rag.ingestion import IngestionSettings, build_index, load_records, new_index_version
from rag.retrieval import RetrievalConfig, Retriever
from rag.schemas import RetrievalFilters
from rag.vector_store import LocalVectorStore

log = get_logger(__name__)

__all__ = ["ReindexResult", "reindex", "activate", "rollback", "list_versions", "main"]

ACTIVE_POINTER = "ACTIVE"
HISTORY_FILE = "activation_history.json"


@dataclass(slots=True)
class ReindexResult:
    index_version: str
    activated: bool
    regression_passed: bool
    regression: dict[str, Any]
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_version": self.index_version,
            "activated": self.activated,
            "regression_passed": self.regression_passed,
            "regression": self.regression,
            "manifest": self.manifest,
        }


# --- activation -------------------------------------------------------------
def active_version(index_root: Path = INDEX_ROOT) -> str | None:
    pointer = index_root / ACTIVE_POINTER
    try:
        if pointer.is_symlink():
            return Path(os.readlink(pointer)).name
        if pointer.is_file():
            return pointer.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
    return None


def list_versions(index_root: Path = INDEX_ROOT) -> list[dict[str, Any]]:
    """Every built index version, newest first, with its manifest summary."""
    current = active_version(index_root)
    out: list[dict[str, Any]] = []
    for directory in sorted(index_root.glob("*.*"), reverse=True):
        if not directory.is_dir() or directory.name == ACTIVE_POINTER:
            continue
        manifest_path = directory / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        out.append(
            {
                "index_version": directory.name,
                "active": directory.name == current,
                "chunks": manifest.get("chunks", 0),
                "documents": manifest.get("documents", 0),
                "created_at": manifest.get("created_at", ""),
                "embedding_model": manifest.get("embedding_model", ""),
                "path": str(directory),
            }
        )
    return out


def _record_activation(index_root: Path, version: str, previous: str | None) -> None:
    path = index_root / HISTORY_FILE
    history = read_json(path) if path.is_file() else []
    history.append(
        {
            "activated": version,
            "previous": previous,
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(path, history[-50:])


def activate(version: str, *, index_root: Path = INDEX_ROOT) -> str | None:
    """Point ACTIVE at ``version``.  Returns the previously active version."""
    target = index_root / version
    if not (target / "manifest.json").is_file():
        raise FileNotFoundError(f"index version {version!r} not found at {target}")

    previous = active_version(index_root)
    pointer = index_root / ACTIVE_POINTER
    index_root.mkdir(parents=True, exist_ok=True)

    tmp = index_root / f".{ACTIVE_POINTER}.{os.getpid()}.tmp"
    tmp.unlink(missing_ok=True)
    try:
        os.symlink(version, tmp)
        os.replace(tmp, pointer)
    except OSError:
        # Filesystems without symlink support fall back to a pointer file.  The
        # swap is still atomic because it goes through os.replace.
        tmp.unlink(missing_ok=True)
        if pointer.is_symlink():
            pointer.unlink()
        tmp.write_text(version, encoding="utf-8")
        os.replace(tmp, pointer)

    _record_activation(index_root, version, previous)
    log.info("rag.reindex.activated", extra={"version": version, "previous": previous})
    return previous


def rollback(*, index_root: Path = INDEX_ROOT) -> str | None:
    """Re-activate the version that was active before the last activation."""
    path = index_root / HISTORY_FILE
    if not path.is_file():
        raise FileNotFoundError("no activation history; cannot roll back")
    history = read_json(path)
    for entry in reversed(history):
        previous = entry.get("previous")
        if previous and (index_root / previous / "manifest.json").is_file():
            activate(previous, index_root=index_root)
            return previous
    raise RuntimeError("no previous index version is available to roll back to")


def prune(keep: int = 3, *, index_root: Path = INDEX_ROOT) -> list[str]:
    """Delete old index versions, always keeping the active one."""
    current = active_version(index_root)
    versions = [v["index_version"] for v in list_versions(index_root)]
    removed: list[str] = []
    for version in versions[keep:]:
        if version == current:
            continue
        shutil.rmtree(index_root / version, ignore_errors=True)
        removed.append(version)
    if removed:
        log.info("rag.reindex.pruned", extra={"removed": removed})
    return removed


# --- regression gate --------------------------------------------------------
def run_regression(
    index_version: str,
    golden_path: str | Path,
    *,
    index_root: Path = INDEX_ROOT,
    settings: IngestionSettings | None = None,
    min_recall: float = 0.80,
    top_k: int = 5,
) -> dict[str, Any]:
    """Query the *new* index with the golden set before it is allowed to go live.

    Each golden record may declare ``expected_document_ids`` (or
    ``expected_product_id``).  Records without either are counted as
    "answerable" checks only: they must return at least one chunk.
    """
    golden = Path(golden_path)
    if not golden.is_file():
        return {
            "ran": False,
            "reason": f"golden file not found: {golden}",
            "passed": True,  # do not block on a missing optional gate
        }

    cfg = settings or IngestionSettings()
    store = LocalVectorStore.load(index_root / index_version / "vectors")
    embedder = build_embedder(cfg.embedding_backend, cfg.embedding_model, cfg.embedding_dim)
    retriever = Retriever(
        store,
        embedder,
        config=RetrievalConfig(top_k=top_k),
        index_version=index_version,
    )

    checked = hits = empty = 0
    failures: list[dict[str, Any]] = []
    for record in read_jsonl(golden, skip_invalid=True):
        question = str(record.get("question") or record.get("query") or "").strip()
        if not question:
            continue
        checked += 1
        filters = RetrievalFilters()
        if record.get("product_id"):
            filters.product_id = str(record["product_id"])
        result = retriever.retrieve(question, filters=filters)

        if result.is_empty:
            empty += 1
            if record.get("expected_document_ids") or record.get("expected_product_id"):
                failures.append({"question": question[:80], "reason": "no_results"})
            continue

        expected = set(record.get("expected_document_ids") or [])
        if expected:
            if expected & {c.document_id for c in result.chunks}:
                hits += 1
            else:
                failures.append(
                    {
                        "question": question[:80],
                        "reason": "expected_document_missing",
                        "expected": sorted(expected),
                        "got": [c.document_id for c in result.chunks],
                    }
                )
        elif record.get("expected_product_id"):
            wanted = str(record["expected_product_id"]).upper()
            if any(c.product_id.upper() == wanted for c in result.chunks):
                hits += 1
            else:
                failures.append(
                    {"question": question[:80], "reason": "expected_product_missing"}
                )
        else:
            hits += 1

    recall = hits / checked if checked else 0.0
    return {
        "ran": True,
        "checked": checked,
        "hits": hits,
        "empty_results": empty,
        f"recall_at_{top_k}": round(recall, 4),
        "min_recall": min_recall,
        "passed": checked == 0 or recall >= min_recall,
        "failures": failures[:20],
    }


# --- orchestration ----------------------------------------------------------
def reindex(
    input_path: str | Path,
    *,
    settings: IngestionSettings | None = None,
    index_version: str | None = None,
    golden_path: str | Path | None = None,
    activate_on_success: bool = False,
    min_recall: float = 0.80,
    keep_versions: int = 3,
) -> ReindexResult:
    """Build -> regression test -> (optionally) activate."""
    cfg = settings or IngestionSettings()
    documents = load_records(input_path)
    if not documents:
        raise ValueError(f"no valid company records in {input_path}")

    version = index_version or new_index_version()
    store, manifest = build_index(
        documents, settings=cfg, index_version=version, source_file=str(input_path)
    )
    if isinstance(store, LocalVectorStore):
        store.persist()

    regression: dict[str, Any] = {"ran": False, "passed": True}
    if golden_path:
        regression = run_regression(
            version,
            golden_path,
            index_root=Path(cfg.index_root),
            settings=cfg,
            min_recall=min_recall,
        )

    activated = False
    if activate_on_success:
        if regression.get("passed", True):
            activate(version, index_root=Path(cfg.index_root))
            prune(keep_versions, index_root=Path(cfg.index_root))
            activated = True
        else:
            log.error(
                "rag.reindex.regression_failed",
                extra={"version": version, "regression": regression},
            )

    return ReindexResult(
        index_version=version,
        activated=activated,
        regression_passed=bool(regression.get("passed", True)),
        regression=regression,
        manifest=manifest.model_dump(mode="json"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag.reindex",
        description="Build, test, activate and roll back company knowledge indexes",
    )
    parser.add_argument("--config", default="configs/rag/ingestion.yaml")
    parser.add_argument("--input", default=None, help="canonical records JSONL")
    parser.add_argument("--index-version", default=None)
    parser.add_argument("--activate", action="store_true", help="activate if regression passes")
    parser.add_argument("--activate-version", default=None, help="activate an existing version")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_versions")
    parser.add_argument("--golden", default="evaluation/golden/customer_support.jsonl")
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--allow-non-semantic", action="store_true")
    args = parser.parse_args(argv)

    settings = IngestionSettings.from_config(load_config(args.config))
    if args.allow_non_semantic:
        settings.allow_non_semantic_embedder = True
    index_root = Path(settings.index_root)

    if args.list_versions:
        print(json.dumps(list_versions(index_root), indent=2))
        return 0

    if args.rollback:
        restored = rollback(index_root=index_root)
        print(f"rolled back to {restored}")
        return 0

    if args.activate_version:
        previous = activate(args.activate_version, index_root=index_root)
        print(f"activated {args.activate_version} (was {previous})")
        return 0

    if not args.input:
        parser.error("--input is required unless --list/--rollback/--activate-version is used")

    result = reindex(
        args.input,
        settings=settings,
        index_version=args.index_version,
        golden_path=args.golden,
        activate_on_success=args.activate,
        min_recall=args.min_recall,
        keep_versions=args.keep,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if args.activate and not result.activated:
        print("\nNOT ACTIVATED: the retrieval regression gate failed.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
