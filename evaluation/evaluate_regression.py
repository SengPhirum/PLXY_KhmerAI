"""Candidate vs production regression comparison (§Phase 17 release comparison).

No checkpoint may be released without this: it compares two evaluation reports
metric-by-metric and produces a verdict.  A metric that moved the wrong way by
more than ``tolerance`` is a regression, and any regression on a *blocking*
metric (hallucination, grounding, support accuracy) fails the release
regardless of improvements elsewhere.

    python -m evaluation.evaluate_regression \
        --candidate evaluation/reports/customer_support.json \
        --baseline  reports/baseline/customer_support.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common.io import atomic_write_text, read_json, write_json
from common.logging import get_logger
from common.paths import EVAL_REPORT_DIR, ensure_dir
from evaluation.schemas import ComparisonReport

log = get_logger(__name__)

__all__ = ["compare_reports", "main", "LOWER_IS_BETTER", "BLOCKING_METRICS"]

# Metrics where a *lower* value is an improvement.
LOWER_IS_BETTER = frozenset(
    {
        "hallucination_rate",
        "unsupported_claim_rate",
        "empty_result_rate",
    }
    | {f"latency_ms_{p}" for p in ("mean", "p50", "p90", "p95", "p99", "max")}
)

# A regression on any of these blocks the release outright.
BLOCKING_METRICS = frozenset(
    {
        "hallucination_rate",
        "grounding_precision",
        "unsupported_claim_rate",
        "support_accuracy",
        "behaviour",
        "pass_rate",
        "recall@5",
        "adversarial_block_rate",
    }
)


def _metrics_of(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in (report.get("aggregate") or {}).items():
        if isinstance(value, int | float):
            metrics[key] = float(value)
    for key, gate in (report.get("gates") or {}).items():
        if isinstance(gate, dict) and isinstance(gate.get("value"), int | float):
            metrics.setdefault(key, float(gate["value"]))
    items = report.get("items") or 0
    if items:
        metrics.setdefault("pass_rate", float(report.get("passed", 0)) / float(items))
    return metrics


def compare_reports(
    candidate_path: str | Path,
    baseline_path: str | Path,
    *,
    tolerance: float = 0.01,
) -> ComparisonReport:
    """Compare two evaluation report JSON files."""
    candidate_raw = read_json(candidate_path)
    baseline_raw = read_json(baseline_path)

    candidate_metrics = _metrics_of(candidate_raw)
    baseline_metrics = _metrics_of(baseline_raw)

    comparison = ComparisonReport(
        candidate=str(candidate_raw.get("model") or Path(candidate_path).stem),
        baseline=str(baseline_raw.get("model") or Path(baseline_path).stem),
    )

    for metric in sorted(set(candidate_metrics) | set(baseline_metrics)):
        # Latency and count metrics are informational, not pass/fail.
        if metric.startswith(("latency_ms_", "chunks", "claims", "unanswerable_items", "adversarial_items")):
            comparison.metrics[metric] = {
                "baseline": baseline_metrics.get(metric, 0.0),
                "candidate": candidate_metrics.get(metric, 0.0),
            }
            continue
        if metric not in candidate_metrics or metric not in baseline_metrics:
            continue

        baseline_value = baseline_metrics[metric]
        candidate_value = candidate_metrics[metric]
        comparison.metrics[metric] = {"baseline": baseline_value, "candidate": candidate_value}

        delta = candidate_value - baseline_value
        improved = delta < -tolerance if metric in LOWER_IS_BETTER else delta > tolerance
        regressed = delta > tolerance if metric in LOWER_IS_BETTER else delta < -tolerance

        if regressed:
            marker = " [BLOCKING]" if metric in BLOCKING_METRICS else ""
            comparison.regressions.append(
                f"{metric}: {baseline_value:.4f} -> {candidate_value:.4f} ({delta:+.4f}){marker}"
            )
        elif improved:
            comparison.improvements.append(
                f"{metric}: {baseline_value:.4f} -> {candidate_value:.4f} ({delta:+.4f})"
            )

    blocking = [r for r in comparison.regressions if "[BLOCKING]" in r]
    if blocking:
        comparison.verdict = "reject"
    elif comparison.regressions:
        comparison.verdict = "review"
    elif comparison.improvements:
        comparison.verdict = "accept"
    else:
        comparison.verdict = "neutral"

    log.info(
        "evaluation.regression.compared",
        extra={
            "verdict": comparison.verdict,
            "regressions": len(comparison.regressions),
            "improvements": len(comparison.improvements),
        },
    )
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.evaluate_regression")
    parser.add_argument("--candidate", required=True, help="candidate report JSON")
    parser.add_argument("--baseline", required=True, help="current production report JSON")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--report-name", default="regression_comparison")
    parser.add_argument(
        "--allow-review",
        action="store_true",
        help="exit 0 on a non-blocking regression (still requires sign-off)",
    )
    args = parser.parse_args(argv)

    comparison = compare_reports(args.candidate, args.baseline, tolerance=args.tolerance)
    directory = ensure_dir(EVAL_REPORT_DIR)
    write_json(directory / f"{args.report_name}.json", json.loads(comparison.model_dump_json()))
    atomic_write_text(directory / f"{args.report_name}.md", comparison.to_markdown())
    print(comparison.to_markdown())

    if comparison.verdict == "reject":
        return 2
    if comparison.verdict == "review" and not args.allow_review:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
