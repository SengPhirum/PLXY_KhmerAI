"""Khmer language quality evaluation (Phase 5 baseline, Phase 17 release gate).

Measures fluency, grammaticality proxies, naturalness screens, code-switch
handling and noise robustness.  Automated screening catches mechanical failures;
anything that passes the screen is flagged for native-speaker review, because no
heuristic can judge whether Khmer is *idiomatic*.

    python -m evaluation.evaluate_language --backend api --golden evaluation/golden/khmer_general.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import load_config
from common.logging import get_logger
from evaluation.metrics import (
    aggregate,
    chrf,
    contains_all,
    khmer_fluency,
    latency_percentiles,
    token_f1,
)
from evaluation.runner import build_runner, load_golden, stamp_report, write_report
from evaluation.schemas import EvalReport, GoldenItem, ItemResult, ModelAnswer

log = get_logger(__name__)

__all__ = ["evaluate_language", "main", "score_item"]


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

    fluency = khmer_fluency(answer.answer, source=item.question, expect_language=item.language)
    result.scores["khmer_fluency"] = fluency["score"]
    result.scores["khmer_ratio"] = fluency.get("khmer_ratio", 0.0)
    result.needs_human_review = bool(fluency["needs_human_review"])
    result.failures.extend(fluency["issues"])

    if item.reference_answer:
        result.scores["chrf"] = chrf(answer.answer, item.reference_answer)
        result.scores["token_f1"] = token_f1(answer.answer, item.reference_answer)

    ok, missing = contains_all(answer.answer, item.must_contain)
    if not ok:
        result.failures.append("missing:" + ",".join(missing[:3]))
    present = [t for t in item.must_not_contain if t in answer.answer]
    if present:
        result.failures.append("forbidden:" + ",".join(present[:3]))

    result.passed = not result.failures and fluency["score"] >= 0.7
    return result


def evaluate_language(
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
        name="khmer_language",
        model=model or runner.name,
        items=len(results),
        passed=sum(1 for r in results if r.passed),
        results=results,
        aggregate=aggregate([r.scores for r in results]),
    )

    by_category: dict[str, dict[str, float]] = {}
    for result in results:
        bucket = by_category.setdefault(str(result.category), {"items": 0.0, "passed": 0.0})
        bucket["items"] += 1
        bucket["passed"] += 1 if result.passed else 0
    for bucket in by_category.values():
        bucket["pass_rate"] = bucket["passed"] / bucket["items"] if bucket["items"] else 0.0
    report.by_category = by_category

    latency = latency_percentiles([r.latency_ms for r in results if r.latency_ms])
    if latency:
        report.aggregate.update({f"latency_ms_{k}": v for k, v in latency.items()})

    gates = thresholds or {}
    report.add_gate(
        "khmer_fluency_mean",
        report.aggregate.get("khmer_fluency", 0.0),
        gates.get("khmer_fluency_min", 0.80),
    )
    report.add_gate("pass_rate", report.pass_rate, gates.get("pass_rate_min", 0.85))
    review = sum(1 for r in results if r.needs_human_review)
    report.notes = (
        f"{review} of {len(results)} answers passed the automated screen and require "
        "native-speaker scoring with the rubric in docs/evaluation.md. "
        "Automated fluency is a screen for mechanical defects only - it does not "
        "measure whether the Khmer is idiomatic."
    )
    return stamp_report(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.evaluate_language")
    parser.add_argument("--golden", default="evaluation/golden/khmer_general.jsonl")
    parser.add_argument("--backend", choices=("api", "ollama", "static"), default="static")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--answers", default=None, help="recorded answers for --backend static")
    parser.add_argument("--report-name", default="khmer_language")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)

    gates = (load_config(args.config).get("slo", {}) or {}).get("quality_gates", {}) or {}
    report = evaluate_language(
        args.golden,
        args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        answers=args.answers,
        thresholds={
            "khmer_fluency_min": float(gates.get("khmer_naturalness_min", 4.0)) / 5.0,
            "pass_rate_min": 0.85,
        },
    )
    write_report(report, args.report_name)
    print(report.to_markdown())
    return 0 if report.gates_passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
