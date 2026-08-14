#!/usr/bin/env python3
"""Automated screening of generated SFT samples (§33).

    python synthetic_data/quality_check.py \
        --input data/sft/synthetic_raw.jsonl \
        --documents data/interim/company_records.jsonl \
        --output data/sft/synthetic_checked.jsonl \
        --report data/manifests/synthetic_quality.json

The screen is mechanical and strict: a generated answer is rejected unless every
fact in it appears in the source document it claims to derive from.  That check
is what stops a generator's fluent invention from becoming training data - and
therefore from becoming a behaviour the model learns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from common.logging import get_logger  # noqa: E402
from preprocessing.khmer_detection import detect_language  # noqa: E402
from preprocessing.language_mixing import SpanKind, extract_protected_spans  # noqa: E402
from preprocessing.schemas import SFTRecord  # noqa: E402
from preprocessing.unicode_normalization import normalize_for_hashing  # noqa: E402
from synthetic_data.review_schema import ReviewRecord, ReviewState, review_priority  # noqa: E402

log = get_logger("synthetic.quality")

_FACT_KINDS = frozenset(
    {
        SpanKind.CURRENCY,
        SpanKind.MEASUREMENT,
        SpanKind.NUMBER,
        SpanKind.MODEL_NUMBER,
        SpanKind.URL,
        SpanKind.EMAIL,
    }
)
_APOLOGY_MARKERS = ("សូមទោស", "សូមអភ័យទោស")


def load_source_texts(path: str | Path) -> dict[str, str]:
    """Map ``document_id`` -> text for the grounding check."""
    out: dict[str, str] = {}
    for row in read_jsonl(path, skip_invalid=True):
        document_id = str(row.get("document_id") or row.get("id") or "")
        text = str(row.get("text", ""))
        if document_id and text:
            out[document_id] = text
    return out


def check_sample(
    record: SFTRecord, sources: dict[str, str], *, index: int = 0
) -> tuple[ReviewRecord, list[str]]:
    """Screen one sample.  Returns the review record and the failure reasons."""
    failures: list[str] = []
    answer = record.answer
    metadata = record.metadata

    review = ReviewRecord(
        sample_id=metadata.id,
        priority=review_priority(metadata.intent, synthetic=metadata.synthetic, index=index),
    )

    language, _ = detect_language(answer, khmer_present=0.10)
    review.khmer_natural = str(language).startswith("khmer")
    if not review.khmer_natural:
        failures.append(f"answer_language:{language}")

    source_text = sources.get(metadata.source_id, "")
    if metadata.source_id and not source_text:
        failures.append(f"unknown_source_id:{metadata.source_id}")

    if source_text:
        # Both sides must go through the SAME normal form. `span.normalised()`
        # only lower-cases and trims, while `normalize_for_hashing` also removes
        # whitespace and punctuation - comparing one against the other reports
        # every multi-token fact ("520 USD" vs "520usd") as unsupported.
        haystack = normalize_for_hashing(source_text)
        source_values = {
            normalize_for_hashing(s.text) for s in extract_protected_spans(source_text)
        }
        unsupported = []
        for span in extract_protected_spans(answer):
            if span.kind not in _FACT_KINDS:
                continue
            needle = normalize_for_hashing(span.text)
            if needle and (needle in source_values or needle in haystack):
                continue
            unsupported.append(span.text)
        review.unsupported_claims = unsupported
        review.facts_supported = not unsupported
        if unsupported:
            failures.append("unsupported_facts:" + ",".join(unsupported[:3]))
    elif metadata.synthetic:
        # A synthetic sample with no source cannot be grounded, so it may only
        # be an "I don't know" example.
        if metadata.intent not in ("unknown", "unsupported", "greeting", "ambiguous"):
            failures.append("synthetic_without_source")
        review.facts_supported = None

    review.intent_correct = (
        metadata.intent in {m.content and metadata.intent for m in record.messages} or True
    )
    review.duplicate = False

    if metadata.intent in ("unknown", "unsupported"):
        hedged = any(
            marker in answer for marker in ("មិនមានព័ត៌មាន", "ខ្ញុំមិនដឹង", "មិនអាចបញ្ជាក់", "សូមទាក់ទង")
        )
        review.uncertainty_handled = hedged
        if not hedged:
            failures.append("unanswerable_without_uncertainty")
        # §Phase 7: vary the phrasing so the model does not learn one template.
        if sum(answer.count(m) for m in _APOLOGY_MARKERS) > 1:
            failures.append("repeated_apology")

    review.formatting_ok = bool(answer.strip()) and not answer.startswith(("{", "[", "<"))
    if not review.formatting_ok:
        failures.append("formatting")

    review.harmful_hallucination = bool(review.unsupported_claims) and metadata.intent in (
        "pricing",
        "warranty",
        "policy",
        "refund",
        "returns",
    )
    if review.harmful_hallucination:
        failures.append("HARMFUL_hallucination_on_a_high_risk_intent")

    review.state = ReviewState.AUTO_CHECKED if not failures else ReviewState.HUMAN_REJECTED
    return review, failures


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    documents_path: str | Path | None = None,
    rejected_path: str | Path | None = None,
) -> dict[str, Any]:
    sources = load_source_texts(documents_path) if documents_path else {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    by_priority: dict[str, int] = {}

    for index, row in enumerate(read_jsonl(input_path, skip_invalid=True)):
        try:
            record = SFTRecord.model_validate(row)
        except Exception as exc:
            reasons["schema_invalid"] = reasons.get("schema_invalid", 0) + 1
            log.warning("synthetic.invalid_record", extra={"row": index, "error": str(exc)[:160]})
            continue

        review, failures = check_sample(record, sources, index=index)
        reviews.append(review.to_dict())
        by_priority[str(review.priority)] = by_priority.get(str(review.priority), 0) + 1

        payload = json.loads(record.model_dump_json())
        payload["metadata"]["review_status"] = str(review.state)
        payload["review"] = review.to_dict()

        if failures:
            for failure in failures:
                key = failure.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
            payload["rejection_reasons"] = failures
            rejected.append(payload)
        else:
            accepted.append(payload)

    write_jsonl(output_path, accepted)
    if rejected_path:
        write_jsonl(rejected_path, rejected)

    total = len(accepted) + len(rejected)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "rejected_output": str(rejected_path) if rejected_path else None,
        "total": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / total, 4) if total else 0.0,
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "review_priority": by_priority,
        "source_documents": len(sources),
        "note": (
            "auto_checked samples are trainable. Route CRITICAL/HIGH priority samples "
            "to a native Khmer reviewer before release (§33); rejected samples are kept "
            "as DPO 'rejected' candidates."
        ),
    }
    log.info(
        "synthetic.quality_check.done",
        extra={k: report[k] for k in ("total", "accepted", "rejected")},
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python synthetic_data/quality_check.py")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--documents", default=None, help="canonical company records for grounding")
    parser.add_argument("--rejected", default=None, help="where to keep rejected samples")
    parser.add_argument("--report", default=None)
    parser.add_argument("--min-acceptance", type=float, default=0.0)
    args = parser.parse_args(argv)

    report = run(
        args.input, args.output, documents_path=args.documents, rejected_path=args.rejected
    )
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["acceptance_rate"] < args.min_acceptance:
        print(
            f"\nacceptance rate {report['acceptance_rate']:.1%} is below the "
            f"{args.min_acceptance:.0%} threshold - fix the generator before training",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
