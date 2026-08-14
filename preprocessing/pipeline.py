"""End-to-end Khmer corpus pipeline: raw -> cleaned -> deduplicated.

Stages, in order::

    read -> html/boilerplate strip -> unicode normalise -> PII -> language
         -> quality score -> exact dedup -> near dedup -> write + audit report

Run it::

    python -m preprocessing.pipeline \
        --input data/raw/public \
        --output data/cleaned \
        --report data/manifests/preprocessing_report.json

Every stage is individually testable and the whole run is deterministic: given
the same inputs, the same config and the same seed, the output files are
byte-identical, which is what makes the corpus reproducible for a training run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.hashing import sha256_text, stable_id
from common.io import read_jsonl, write_json, write_jsonl
from common.logging import get_logger
from preprocessing.exact_dedup import ExactDeduplicator, content_hash
from preprocessing.html_cleanup import clean_web_text
from preprocessing.khmer_detection import detect_language
from preprocessing.near_dedup import NearDedupConfig, NearDeduplicator
from preprocessing.pii_filter import PiiFilter, PiiPolicy
from preprocessing.quality_filter import QualityThresholds, assess_quality
from preprocessing.schemas import CleanRecord, PipelineStats
from preprocessing.unicode_normalization import NormalizationConfig, normalize_khmer

log = get_logger(__name__)

__all__ = ["KhmerPipeline", "PipelineConfig", "main", "run_pipeline"]


@dataclass(slots=True)
class PipelineConfig:
    """Everything that influences the output, so it can be fingerprinted."""

    strip_html: bool = True
    normalization: NormalizationConfig | None = None
    quality: QualityThresholds | None = None
    pii: PiiPolicy | None = None
    near_dedup: NearDedupConfig | None = None
    exact_dedup: bool = True
    enable_near_dedup: bool = True
    text_key: str = "text"
    source_key: str = "source"
    keep_rejected: bool = False

    def resolved(self) -> PipelineConfig:
        return PipelineConfig(
            strip_html=self.strip_html,
            normalization=self.normalization or NormalizationConfig.for_training_corpus(),
            quality=self.quality or QualityThresholds(),
            pii=self.pii or PiiPolicy.for_public_corpus(),
            near_dedup=self.near_dedup or NearDedupConfig(),
            exact_dedup=self.exact_dedup,
            enable_near_dedup=self.enable_near_dedup,
            text_key=self.text_key,
            source_key=self.source_key,
            keep_rejected=self.keep_rejected,
        )

    def fingerprint(self) -> str:
        """Hash of the effective settings - recorded in the run manifest."""
        resolved = self.resolved()
        # `resolved()` always fills these in; bind locally so the types are narrow.
        pii = resolved.pii or PiiPolicy.for_public_corpus()
        payload = {
            "strip_html": resolved.strip_html,
            "normalization": asdict(resolved.normalization),  # type: ignore[arg-type]
            "quality": asdict(resolved.quality),  # type: ignore[arg-type]
            "pii": {
                "mode": pii.mode,
                "keep_kinds": sorted(pii.keep_kinds),
                "drop_if_categories": sorted(pii.drop_if_categories),
            },
            "near_dedup": asdict(resolved.near_dedup),  # type: ignore[arg-type]
            "exact_dedup": resolved.exact_dedup,
            "enable_near_dedup": resolved.enable_near_dedup,
        }
        return sha256_text(json.dumps(payload, sort_keys=True, default=str))[:16]


class KhmerPipeline:
    """Stateful pipeline - the deduplicators accumulate across all inputs."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = (config or PipelineConfig()).resolved()
        self.stats = PipelineStats(config_fingerprint=(config or PipelineConfig()).fingerprint())
        self._exact = ExactDeduplicator() if self.config.exact_dedup else None
        self._near = (
            NearDeduplicator(self.config.near_dedup) if self.config.enable_near_dedup else None
        )
        self._pii = PiiFilter(self.config.pii)
        self.rejected: list[dict[str, Any]] = []

    # -- single record -------------------------------------------------------
    def process_record(self, record: dict[str, Any]) -> CleanRecord | None:
        self.stats.input_records += 1
        raw_text = str(record.get(self.config.text_key, "") or "")
        self.stats.bytes_in += len(raw_text.encode("utf-8"))
        source = str(record.get(self.config.source_key, "") or "unknown")

        text = clean_web_text(raw_text) if self.config.strip_html else raw_text

        norm = normalize_khmer(text, self.config.normalization)
        for rule, count in norm.changes.items():
            self.stats.normalisation_changes[rule] = (
                self.stats.normalisation_changes.get(rule, 0) + count
            )
        text = norm.text

        cleaned, _ = self._pii.process_text(text)
        if cleaned is None:
            self.stats.rejected_pii += 1
            self._reject(record, "pii_blocked")
            return None
        text = cleaned

        language, _profile = detect_language(text)
        assessment = assess_quality(text, self.config.quality)
        if not assessment.accepted:
            reason = assessment.rejection_reason or "unknown"
            if reason.startswith("language_"):
                self.stats.rejected_language += 1
            else:
                self.stats.rejected_quality += 1
            self.stats.rejection_reasons[reason] = self.stats.rejection_reasons.get(reason, 0) + 1
            self._reject(record, reason)
            return None

        if self._exact is not None and not self._exact.is_new(text, source=source):
            self.stats.exact_duplicates += 1
            self.stats.rejection_reasons["exact_duplicate"] = (
                self.stats.rejection_reasons.get("exact_duplicate", 0) + 1
            )
            return None

        record_id = str(record.get("id") or stable_id(source, content_hash(text)))

        if self._near is not None and self._near.add(record_id, text).is_duplicate:
            self.stats.near_duplicates += 1
            self.stats.rejection_reasons["near_duplicate"] = (
                self.stats.rejection_reasons.get("near_duplicate", 0) + 1
            )
            return None

        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("original_length", len(raw_text))
        metadata["normalisation"] = norm.changes
        metadata["quality_signals"] = {
            k: round(v, 4) for k, v in assessment.signals.items() if k != "chars"
        }

        clean = CleanRecord(
            id=record_id,
            source=source,
            text=text,
            language="km" if str(language).startswith("khmer") else str(language),
            khmer_ratio=assessment.profile.khmer_ratio if assessment.profile else 0.0,
            quality_score=assessment.score,
            hash=content_hash(text),
            metadata=metadata,
        )
        self.stats.output_records += 1
        self.stats.bytes_out += len(text.encode("utf-8"))
        return clean

    def _reject(self, record: dict[str, Any], reason: str) -> None:
        if self.config.keep_rejected:
            self.rejected.append(
                {
                    "id": record.get("id"),
                    "source": record.get(self.config.source_key),
                    "reason": reason,
                    "preview": str(record.get(self.config.text_key, ""))[:200],
                }
            )

    # -- streams -------------------------------------------------------------
    def process(self, records: Iterable[dict[str, Any]]) -> Iterator[CleanRecord]:
        for record in records:
            clean = self.process_record(record)
            if clean is not None:
                yield clean

    def audit(self) -> dict[str, Any]:
        """Full audit payload for ``data/manifests/preprocessing_report.json``."""
        payload: dict[str, Any] = self.stats.model_dump()
        payload["retention_rate"] = round(self.stats.retention_rate, 4)
        payload["pii"] = self._pii.report.to_dict()
        if self._exact is not None:
            payload["exact_dedup"] = self._exact.stats.to_dict()
        if self._near is not None:
            payload["near_dedup"] = self._near.stats.to_dict()
        if self.config.keep_rejected:
            payload["rejected_samples"] = self.rejected[:100]
        return payload


def _iter_input_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in {".jsonl", ".json", ".txt"}
    )


def _read_any(path: Path, text_key: str) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        yield from read_jsonl(path, skip_invalid=True)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [data]
        yield from (row for row in rows if isinstance(row, dict))
    else:  # .txt - one document per file
        yield {text_key: path.read_text(encoding="utf-8", errors="replace"), "source": path.stem}


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: PipelineConfig | None = None,
    report_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the pipeline over a file or directory and write cleaned JSONL."""
    started = time.perf_counter()
    pipeline = KhmerPipeline(config)
    root = Path(input_path)
    files = _iter_input_files(root)
    if not files:
        raise FileNotFoundError(f"no .jsonl/.json/.txt inputs found under {root}")

    out_path = Path(output_path)
    if out_path.suffix == "":
        out_path = out_path / "cleaned.jsonl"

    produced = 0
    rows: list[dict[str, Any]] = []
    for file in files:
        log.info("preprocessing.file.start", extra={"file": str(file)})
        for record in _read_any(file, pipeline.config.text_key):
            if limit is not None and produced >= limit:
                break
            record.setdefault("source", file.stem)
            clean = pipeline.process_record(record)
            if clean is not None:
                rows.append(clean.model_dump())
                produced += 1
        if limit is not None and produced >= limit:
            break

    write_jsonl(out_path, rows)
    pipeline.stats.seconds = round(time.perf_counter() - started, 3)
    audit = pipeline.audit()
    audit["input_files"] = [str(f) for f in files]
    audit["output_file"] = str(out_path)

    if report_path:
        write_json(report_path, audit)
        log.info("preprocessing.report.written", extra={"report": str(report_path)})

    log.info(
        "preprocessing.done",
        extra={
            "input_records": pipeline.stats.input_records,
            "output_records": pipeline.stats.output_records,
            "retention_rate": round(pipeline.stats.retention_rate, 4),
            "seconds": pipeline.stats.seconds,
        },
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m preprocessing.pipeline",
        description="Khmer corpus cleaning and deduplication pipeline",
    )
    parser.add_argument("--input", required=True, help="file or directory of .jsonl/.json/.txt")
    parser.add_argument("--output", required=True, help="output .jsonl (or a directory)")
    parser.add_argument("--report", default=None, help="where to write the audit JSON")
    parser.add_argument("--limit", type=int, default=None, help="stop after N accepted records")
    parser.add_argument("--no-html-strip", action="store_true")
    parser.add_argument("--no-near-dedup", action="store_true")
    parser.add_argument("--keep-rejected", action="store_true", help="record rejection samples")
    parser.add_argument(
        "--profile",
        choices=("corpus", "sft"),
        default="corpus",
        help="threshold profile: web corpus (default) or support dialogue",
    )
    args = parser.parse_args(argv)

    config = PipelineConfig(
        strip_html=not args.no_html_strip,
        enable_near_dedup=not args.no_near_dedup,
        keep_rejected=args.keep_rejected,
        quality=QualityThresholds.for_sft() if args.profile == "sft" else QualityThresholds(),
        near_dedup=NearDedupConfig.for_sft() if args.profile == "sft" else NearDedupConfig(),
    )

    audit = run_pipeline(
        args.input, args.output, config=config, report_path=args.report, limit=args.limit
    )
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k != "rejected_samples"},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if audit["output_records"] > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
