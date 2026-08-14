"""Customer-support accuracy evaluation (Phase 17).

Scores whether the assistant actually resolved the customer's need: correct
facts, correct intent handling, correct escalation, and appropriate length.

    python -m evaluation.evaluate_support --backend api
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import load_config
from common.logging import get_logger
from evaluation.metrics import aggregate, chrf, contains_all, khmer_fluency, latency_percentiles
from evaluation.runner import build_runner, load_golden, stamp_report, write_report
from evaluation.schemas import EvalReport, GoldenItem, ItemResult, ModelAnswer

log = get_logger(__name__)

__all__ = ["evaluate_support", "main", "score_item"]

_UNCERTAINTY_MARKERS = (
    "មិនមានព័ត៌មាន",
    "ខ្ញុំមិនដឹង",
    "មិនអាចបញ្ជាក់",
    "មិនមានក្នុងឯកសារ",
    "សូមទាក់ទង",
    "i don't have",
    "cannot confirm",
    "please contact",
)
_ESCALATION_MARKERS = (
    "បញ្ជូន",
    "ផ្នែកបម្រើអតិថិជន",
    "បុគ្គលិក",
    "ទាក់ទងផ្នែក",
    "contact support",
    "transfer you",
    "a colleague",
)
# A support answer that runs past this is not concise (§32).
_MAX_REASONABLE_CHARS = 1200


def _behaviour_ok(item: GoldenItem, answer: ModelAnswer) -> tuple[bool, str]:
    text = answer.answer
    lowered = text.lower()
    if item.expected_behaviour == "answer":
        hedged = any(m in text or m in lowered for m in _UNCERTAINTY_MARKERS)
        return (not hedged), "hedged_on_an_answerable_question" if hedged else ""
    if item.expected_behaviour in ("refuse", "uncertainty"):
        stated = any(m in text or m in lowered for m in _UNCERTAINTY_MARKERS)
        return stated, "" if stated else "did_not_state_uncertainty"
    if item.expected_behaviour == "escalate":
        escalated = answer.escalation_required or any(
            m in text or m in lowered for m in _ESCALATION_MARKERS
        )
        return escalated, "" if escalated else "did_not_escalate"
    if item.expected_behaviour == "clarify":
        asks = "?" in text or "តើ" in text
        return asks, "" if asks else "did_not_ask_a_clarifying_question"
    return True, ""


def score_item(item: GoldenItem, answer: ModelAnswer) -> ItemResult:
    result = ItemResult(
        item_id=item.id,
        category=item.category,
        passed=False,
        answer=answer.answer,
        latency_ms=answer.latency_ms,
    )
    if answer.error:
        result.failures.append(f"error:{answer.error}")
        return result
    if not answer.answer.strip():
        result.failures.append("empty_answer")
        return result

    ok, reason = _behaviour_ok(item, answer)
    result.scores["behaviour"] = 1.0 if ok else 0.0
    if reason:
        result.failures.append(reason)

    contains, missing = contains_all(answer.answer, item.must_contain)
    result.scores["required_facts"] = 1.0 if contains else 0.0
    if not contains:
        result.failures.append("missing_facts:" + ",".join(missing[:3]))

    forbidden = [t for t in item.must_not_contain if t in answer.answer]
    result.scores["no_forbidden_content"] = 0.0 if forbidden else 1.0
    if forbidden:
        result.failures.append("forbidden:" + ",".join(forbidden[:3]))

    if item.reference_answer:
        result.scores["chrf"] = chrf(answer.answer, item.reference_answer)

    fluency = khmer_fluency(answer.answer, source=item.question, expect_language=item.language)
    result.scores["khmer_fluency"] = fluency["score"]
    result.failures.extend(fluency["issues"])

    concise = len(answer.answer) <= _MAX_REASONABLE_CHARS
    result.scores["concise"] = 1.0 if concise else 0.0
    if not concise:
        result.failures.append(f"too_long:{len(answer.answer)}chars")

    if item.expected_document_ids:
        cited = set(answer.sources) & set(item.expected_document_ids)
        result.scores["cited_expected_source"] = 1.0 if cited else 0.0
        if not cited:
            result.failures.append("wrong_or_missing_source")

    result.passed = not result.failures
    result.needs_human_review = result.passed and bool(fluency["needs_human_review"])
    return result


def evaluate_support(
    golden_path: str | Path,
    backend: str = "static",
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    answers: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> EvalReport:
    items = load_golden(golden_path)
    runner = build_runner(backend, model=model, base_url=base_url, api_key=api_key, answers=answers)
    try:
        results = [score_item(item, runner.answer(item)) for item in items]
    finally:
        runner.close()

    report = EvalReport(
        name="customer_support",
        model=model or runner.name,
        items=len(results),
        passed=sum(1 for r in results if r.passed),
        results=results,
        aggregate=aggregate([r.scores for r in results]),
    )

    by_intent: dict[str, dict[str, float]] = {}
    for item, result in zip(items, results, strict=True):
        bucket = by_intent.setdefault(item.intent, {"items": 0.0, "passed": 0.0})
        bucket["items"] += 1
        bucket["passed"] += 1 if result.passed else 0
    for bucket in by_intent.values():
        bucket["pass_rate"] = bucket["passed"] / bucket["items"] if bucket["items"] else 0.0
    report.by_category = by_intent

    latency = latency_percentiles([r.latency_ms for r in results if r.latency_ms])
    if latency:
        report.aggregate.update({f"latency_ms_{k}": v for k, v in latency.items()})

    gates = thresholds or {}
    report.add_gate("support_accuracy", report.pass_rate, gates.get("support_accuracy_min", 0.85))
    report.add_gate(
        "behaviour_correct",
        report.aggregate.get("behaviour", 0.0),
        gates.get("behaviour_min", 0.90),
    )
    return stamp_report(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.evaluate_support")
    parser.add_argument("--golden", default="evaluation/golden/customer_support.jsonl")
    parser.add_argument("--backend", choices=("api", "ollama", "static"), default="static")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--answers", default=None)
    parser.add_argument("--report-name", default="customer_support")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)

    gates = (load_config(args.config).get("slo", {}) or {}).get("quality_gates", {}) or {}
    report = evaluate_support(
        args.golden,
        args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        answers=args.answers,
        thresholds={"support_accuracy_min": float(gates.get("support_accuracy_min", 0.85))},
    )
    write_report(report, args.report_name)
    print(report.to_markdown())
    return 0 if report.gates_passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
