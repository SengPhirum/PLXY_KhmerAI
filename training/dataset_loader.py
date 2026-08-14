"""SFT/DPO dataset loading, validation, splitting and leakage prevention.

The split logic is the important part.  §Phase 7 requires that near-duplicates
must not cross splits, so the pipeline is:

    validate -> exact dedupe -> near dedupe -> seal the test set
             -> reject any train/validation record that leaks into it
             -> split deterministically by a hash of the content

Splitting on a content hash (rather than shuffling with a seed) means a record
lands in the same split even when the dataset grows, which keeps evaluation
comparable across dataset versions.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.hashing import sha256_text
from common.io import read_jsonl, write_json, write_jsonl
from common.logging import get_logger
from preprocessing.exact_dedup import ExactDeduplicator
from preprocessing.near_dedup import LeakageChecker, NearDedupConfig, NearDeduplicator
from preprocessing.quality_filter import QualityThresholds, assess_quality
from preprocessing.schemas import INTENTS, SFTRecord
from preprocessing.unicode_normalization import NormalizationConfig, normalize_text

log = get_logger(__name__)

__all__ = [
    "DatasetStats",
    "load_sft_records",
    "validate_records",
    "split_records",
    "build_splits",
    "conversation_text",
    "load_preference_pairs",
]

_SFT_NORMALIZATION = NormalizationConfig(zwsp_policy="strip")


def conversation_text(record: SFTRecord) -> str:
    """Flattened text used for dedup, leakage checks and quality scoring."""
    return "\n".join(f"{m.role}: {m.content}" for m in record.messages)


@dataclass(slots=True)
class DatasetStats:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    low_quality: int = 0
    leaked: int = 0
    by_intent: dict[str, int] = field(default_factory=dict)
    by_review_status: dict[str, int] = field(default_factory=dict)
    by_source_type: dict[str, int] = field(default_factory=dict)
    synthetic: int = 0
    multi_turn: int = 0
    mean_turns: float = 0.0
    mean_answer_chars: float = 0.0
    errors: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    intent_coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "low_quality": self.low_quality,
            "leaked_into_test": self.leaked,
            "synthetic": self.synthetic,
            "multi_turn": self.multi_turn,
            "mean_turns": round(self.mean_turns, 2),
            "mean_answer_chars": round(self.mean_answer_chars, 1),
            "by_intent": dict(sorted(self.by_intent.items(), key=lambda kv: -kv[1])),
            "by_review_status": self.by_review_status,
            "by_source_type": self.by_source_type,
            "splits": self.splits,
            "intent_coverage": self.intent_coverage,
            "errors": self.errors[:20],
        }
        if self.total:
            data["retention_rate"] = round(self.valid / self.total, 4)
        return data


def load_sft_records(path: str | Path, *, strict: bool = False) -> Iterator[SFTRecord]:
    """Parse an SFT JSONL file, reporting (or raising on) malformed records."""
    for index, row in enumerate(read_jsonl(path, skip_invalid=not strict), start=1):
        try:
            yield SFTRecord.model_validate(row)
        except Exception as exc:  # noqa: BLE001
            message = f"{Path(path).name}:{index}: {exc}"
            if strict:
                raise ValueError(message) from exc
            log.warning("dataset.invalid_record", extra={"detail": message[:200]})


def validate_records(
    records: Iterable[SFTRecord],
    *,
    thresholds: QualityThresholds | None = None,
    dedupe: bool = True,
    near_dedupe: bool = True,
    leakage_checker: LeakageChecker | None = None,
    normalise: bool = True,
) -> tuple[list[SFTRecord], DatasetStats]:
    """Validate, normalise, deduplicate and quality-filter a record stream."""
    stats = DatasetStats()
    quality = thresholds or QualityThresholds.for_sft()
    exact = ExactDeduplicator() if dedupe else None
    near = NearDeduplicator(NearDedupConfig.for_sft()) if near_dedupe else None

    kept: list[SFTRecord] = []
    turn_counts: list[int] = []
    answer_lengths: list[int] = []

    for record in records:
        stats.total += 1

        if normalise:
            record = record.model_copy(
                update={
                    "messages": [
                        m.model_copy(update={"content": normalize_text(m.content, _SFT_NORMALIZATION)})
                        for m in record.messages
                    ]
                }
            )

        text = conversation_text(record)
        answer = record.answer

        assessment = assess_quality(answer, quality)
        if not assessment.accepted:
            stats.low_quality += 1
            stats.errors.append(f"{record.metadata.id}: quality:{assessment.rejection_reason}")
            continue

        if exact is not None and not exact.is_new(text, source=record.metadata.source_type):
            stats.exact_duplicates += 1
            continue

        if leakage_checker is not None and leakage_checker.leaks(text):
            stats.leaked += 1
            stats.errors.append(f"{record.metadata.id}: leaks into the sealed test set")
            continue

        if near is not None and near.add(record.metadata.id, text).is_duplicate:
            stats.near_duplicates += 1
            continue

        stats.valid += 1
        stats.by_intent[record.metadata.intent] = stats.by_intent.get(record.metadata.intent, 0) + 1
        stats.by_review_status[record.metadata.review_status] = (
            stats.by_review_status.get(record.metadata.review_status, 0) + 1
        )
        source = record.metadata.source_type or "unspecified"
        stats.by_source_type[source] = stats.by_source_type.get(source, 0) + 1
        if record.metadata.synthetic:
            stats.synthetic += 1
        if record.turns > 1:
            stats.multi_turn += 1
        turn_counts.append(record.turns)
        answer_lengths.append(len(answer))
        kept.append(record)

    stats.invalid = stats.total - stats.valid - stats.exact_duplicates - stats.near_duplicates - stats.low_quality - stats.leaked
    stats.mean_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0.0
    stats.mean_answer_chars = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0

    missing = [intent for intent in INTENTS if intent not in stats.by_intent]
    stats.intent_coverage = {
        "covered": len(INTENTS) - len(missing),
        "total": len(INTENTS),
        "missing": missing,
        "distribution": {
            k: round(v / stats.valid, 4) for k, v in stats.by_intent.items()
        }
        if stats.valid
        else {},
    }
    if missing:
        log.warning("dataset.intent_gap", extra={"missing_intents": missing})

    return kept, stats


def _split_of(record: SFTRecord, ratios: dict[str, float]) -> str:
    """Deterministic split assignment from a content hash.

    Hash-based rather than shuffle-based so a record keeps its split when the
    dataset grows - which is what makes evaluation comparable across versions.
    """
    digest = sha256_text(conversation_text(record))
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, ratio in ratios.items():
        cumulative += ratio
        if bucket < cumulative:
            return name
    return next(reversed(ratios))


def split_records(
    records: list[SFTRecord],
    *,
    ratios: dict[str, float] | None = None,
) -> dict[str, list[SFTRecord]]:
    """Split deterministically.  Adversarial records always go to their own split."""
    resolved = ratios or {"train": 0.90, "validation": 0.05, "test": 0.05}
    total = sum(resolved.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    splits: dict[str, list[SFTRecord]] = {name: [] for name in resolved}
    splits.setdefault("adversarial", [])
    for record in records:
        if record.metadata.source_type == "adversarial" or record.metadata.intent in (
            "unsupported",
            "safety",
        ):
            splits["adversarial"].append(record)
            continue
        splits[_split_of(record, resolved)].append(record)
    return splits


def build_splits(
    input_paths: list[str | Path],
    output_dir: str | Path,
    *,
    test_seed_paths: list[str | Path] | None = None,
    ratios: dict[str, float] | None = None,
    report_path: str | Path | None = None,
) -> DatasetStats:
    """Full Phase 7 pipeline: load -> validate -> dedupe -> seal -> split -> write."""
    checker: LeakageChecker | None = None
    if test_seed_paths:
        checker = LeakageChecker()
        sealed = 0
        for path in test_seed_paths:
            for record in load_sft_records(path):
                checker.protect(record.metadata.id, conversation_text(record))
                sealed += 1
        log.info("dataset.test_set_sealed", extra={"records": sealed})

    all_records: list[SFTRecord] = []
    for path in input_paths:
        all_records.extend(load_sft_records(path))

    kept, stats = validate_records(all_records, leakage_checker=checker)
    splits = split_records(kept, ratios=ratios)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        if not records:
            continue
        written = write_jsonl(
            target / f"{name}.jsonl", [json.loads(r.model_dump_json()) for r in records]
        )
        stats.splits[name] = written

    # Final cross-split leakage assertion - the gate that Phase 7 requires.
    stats.intent_coverage["cross_split_leakage"] = _cross_split_leakage(splits)

    if report_path:
        write_json(report_path, stats.to_dict())
    log.info("dataset.splits_written", extra={"splits": stats.splits, "output": str(target)})
    return stats


def _cross_split_leakage(splits: dict[str, list[SFTRecord]]) -> dict[str, Any]:
    """Verify that no train record near-duplicates a validation/test record."""
    protected = LeakageChecker(NearDedupConfig.for_sft())
    for name in ("validation", "test", "adversarial"):
        for record in splits.get(name, []):
            protected.protect(f"{name}:{record.metadata.id}", conversation_text(record))
    hits = [r for r in splits.get("train", []) if protected.leaks(conversation_text(r))]
    return {
        "checked_train_records": len(splits.get("train", [])),
        "leaks": len(hits),
        "clean": not hits,
        "examples": [r.metadata.id for r in hits[:10]],
    }


def load_preference_pairs(path: str | Path) -> list[dict[str, str]]:
    """Load DPO triples ``{prompt, chosen, rejected}``, validating each row."""
    pairs: list[dict[str, str]] = []
    for index, row in enumerate(read_jsonl(path, skip_invalid=True), start=1):
        prompt = row.get("prompt") or row.get("question")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not (prompt and chosen and rejected):
            log.warning("dataset.preference.incomplete", extra={"row": index})
            continue
        if normalize_text(str(chosen)) == normalize_text(str(rejected)):
            log.warning("dataset.preference.identical", extra={"row": index})
            continue
        pairs.append(
            {
                "prompt": normalize_text(str(prompt), _SFT_NORMALIZATION),
                "chosen": normalize_text(str(chosen), _SFT_NORMALIZATION),
                "rejected": normalize_text(str(rejected), _SFT_NORMALIZATION),
            }
        )
    return pairs


def describe_mixture(stats: DatasetStats, target: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare the achieved intent mixture with the Phase 7 target distribution."""
    target_mixture = target or {
        "company_support": 0.40,
        "general_khmer": 0.20,
        "code_switch": 0.10,
        "difficult": 0.10,
        "unanswerable": 0.10,
        "comparison": 0.05,
        "multi_turn": 0.05,
    }
    groups = {
        "company_support": ("product_info", "service_info", "pricing", "specification", "availability", "warranty", "policy", "how_to", "installation", "troubleshooting"),
        "general_khmer": ("general_inquiry", "greeting"),
        "code_switch": (),
        "difficult": ("complaint", "ambiguous"),
        "unanswerable": ("unknown", "unsupported"),
        "comparison": ("comparison", "recommendation"),
        "multi_turn": ("multi_turn", "follow_up", "escalation"),
    }
    counts = Counter(stats.by_intent)
    total = sum(counts.values()) or 1
    achieved = {
        group: sum(counts[i] for i in intents) / total for group, intents in groups.items()
    }
    return {
        "target": target_mixture,
        "achieved": {k: round(v, 4) for k, v in achieved.items()},
        "delta": {
            k: round(achieved.get(k, 0.0) - v, 4) for k, v in target_mixture.items()
        },
        "note": (
            "code_switch is not an intent - measure it with "
            "preprocessing.language_mixing.analyse_code_switching over the answers."
        ),
    }
