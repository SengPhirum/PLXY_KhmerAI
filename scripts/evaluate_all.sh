#!/usr/bin/env bash
# Run the full evaluation battery and enforce the release gates (Phase 17).
#
#   bash scripts/evaluate_all.sh                      # against a running API
#   bash scripts/evaluate_all.sh --backend ollama --model khmer-support-9b
#   bash scripts/evaluate_all.sh --baseline reports/baseline
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
BACKEND="api"
MODEL=""
BASE_URL="http://127.0.0.1:8000"
BASELINE_DIR=""
FAILED_GATES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --baseline) BASELINE_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

COMMON=(--backend "$BACKEND" --base-url "$BASE_URL")
[[ -n "$MODEL" ]] && COMMON+=(--model "$MODEL")

if [[ "$BACKEND" == "api" ]] && ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
  echo "The API is not responding at ${BASE_URL}. Start it with: make serve" >&2
  exit 1
fi

run_stage() {
  local label="$1"; shift
  echo ""
  echo "=== ${label} ==="
  if "$@"; then
    echo "    gates passed"
  else
    echo "    GATES FAILED"
    FAILED_GATES+=("$label")
  fi
}

run_stage "Khmer language" "$PY" -m evaluation.evaluate_language \
  --golden evaluation/golden/khmer_general.jsonl "${COMMON[@]}" --report-name khmer_language

run_stage "Code switching" "$PY" -m evaluation.evaluate_language \
  --golden evaluation/golden/code_switch.jsonl "${COMMON[@]}" --report-name code_switch

run_stage "Customer support" "$PY" -m evaluation.evaluate_support \
  --golden evaluation/golden/customer_support.jsonl "${COMMON[@]}" --report-name customer_support

run_stage "Multi-turn" "$PY" -m evaluation.evaluate_support \
  --golden evaluation/golden/multiturn.jsonl "${COMMON[@]}" --report-name multiturn

run_stage "Hallucination + adversarial" "$PY" -m evaluation.evaluate_hallucination \
  --golden evaluation/golden/hallucination.jsonl \
  --adversarial evaluation/golden/adversarial.jsonl "${COMMON[@]}" --report-name hallucination

run_stage "Grounding" "$PY" -m evaluation.evaluate_grounding \
  --golden evaluation/golden/customer_support.jsonl \
  --index-dir data/index/ACTIVE "${COMMON[@]}" --report-name grounding

if [[ -e data/index/ACTIVE ]]; then
  run_stage "Retrieval" "$PY" -m evaluation.evaluate_retrieval \
    --index-dir data/index/ACTIVE --golden evaluation/golden/customer_support.jsonl
else
  echo ""
  echo "=== Retrieval ==="
  echo "    skipped: no active index"
fi

if [[ -n "$BASELINE_DIR" ]]; then
  echo ""
  echo "=== Regression vs baseline ==="
  for report in customer_support hallucination grounding; do
    if [[ -f "${BASELINE_DIR}/${report}.json" ]]; then
      "$PY" -m evaluation.evaluate_regression \
        --candidate "evaluation/reports/${report}.json" \
        --baseline "${BASELINE_DIR}/${report}.json" \
        --report-name "regression_${report}" || FAILED_GATES+=("regression:${report}")
    else
      echo "    no baseline for ${report}"
    fi
  done
fi

echo ""
echo "Reports written to evaluation/reports/"
ls -1 evaluation/reports/*.md 2>/dev/null | sed 's/^/  /' || true

echo ""
if [[ ${#FAILED_GATES[@]} -eq 0 ]]; then
  echo "ALL RELEASE GATES PASSED."
  exit 0
fi
echo "RELEASE BLOCKED. Failed gates:" >&2
printf '  - %s\n' "${FAILED_GATES[@]}" >&2
exit 1
