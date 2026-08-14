#!/usr/bin/env bash
# Phase 13 - Ollama performance tuning matrix on the Mac Studio.
#
#   bash ollama/benchmark.sh --model khmer-support-9b
#   bash ollama/benchmark.sh --model khmer-support-4b --parallel 1,2,4,6,8,10
#   bash ollama/benchmark.sh --compare khmer-support-4b,khmer-support-9b
#
# Sweeps OLLAMA_NUM_PARALLEL x context x connected clients, restarting the daemon
# for each parallelism setting (it is read at startup, not per request), and
# records TTFT, tokens/sec, p50/p95/p99, queue behaviour, failures and memory.
#
# Every number is MEASURED. Nothing in the output is estimated (§1.2).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="khmer-support-9b"
COMPARE=""
PARALLEL_LEVELS="1,2,4,6,8,10"
CONTEXT_LEVELS="4096,8192,16384"
CLIENT_LEVELS="1,5,10,20"
OUTDIR="reports/ollama_benchmark"
DURATION=45
PYTHON="${PYTHON:-.venv/bin/python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --compare) COMPARE="$2"; shift 2 ;;
    --parallel) PARALLEL_LEVELS="$2"; shift 2 ;;
    --context) CONTEXT_LEVELS="$2"; shift 2 ;;
    --clients) CLIENT_LEVELS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v ollama >/dev/null 2>&1 || { echo "ollama is not installed" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "python not found at $PYTHON (run: make setup)" >&2; exit 1; }

mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="${OUTDIR}/summary-${STAMP}.jsonl"

host_snapshot() {
  if [[ "$(uname)" == "Darwin" ]]; then
    local pages_free page_size total_gb
    page_size=$(sysctl -n hw.pagesize)
    pages_free=$(vm_stat | awk '/Pages free/ {gsub("\\.","",$3); print $3}')
    total_gb=$(( $(sysctl -n hw.memsize) / 1000000000 ))
    echo "{\"total_memory_gb\": ${total_gb}, \"free_pages\": ${pages_free}, \"page_size\": ${page_size}}"
  else
    echo "{\"note\": \"host memory snapshot is macOS-only\"}"
  fi
}

restart_daemon() {
  local parallel="$1" context="$2"
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 2
  OLLAMA_NUM_PARALLEL="$parallel" \
  OLLAMA_CONTEXT_LENGTH="$context" \
  OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE:-64}" \
  OLLAMA_KEEP_ALIVE=-1 \
  OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}" \
  OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}" \
  OLLAMA_MAX_LOADED_MODELS=1 \
    nohup ollama serve >"${OUTDIR}/ollama-${parallel}p-${context}c.log" 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf "http://${OLLAMA_HOST:-127.0.0.1:11434}/api/tags" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "the daemon did not become ready (parallel=${parallel} context=${context})" >&2
  return 1
}

warm_model() {
  ollama run "$1" "សួស្តី" >/dev/null 2>&1 || true
}

MODELS="$MODEL"
[[ -n "$COMPARE" ]] && MODELS="${COMPARE//,/ }"

echo "Benchmark matrix"
echo "  models:   ${MODELS}"
echo "  parallel: ${PARALLEL_LEVELS}"
echo "  context:  ${CONTEXT_LEVELS}"
echo "  clients:  ${CLIENT_LEVELS}"
echo "  output:   ${SUMMARY}"
echo ""

for model in $MODELS; do
  for context in ${CONTEXT_LEVELS//,/ }; do
    for parallel in ${PARALLEL_LEVELS//,/ }; do
      echo "=== ${model} | OLLAMA_NUM_PARALLEL=${parallel} | context=${context} ==="
      restart_daemon "$parallel" "$context" || continue
      warm_model "$model"
      before="$(host_snapshot)"

      "$PYTHON" -m evaluation.benchmark_model \
        --model "$model" \
        --concurrency "$CLIENT_LEVELS" \
        --num-ctx "$context" \
        --repeats 2 \
        --output "${OUTDIR}/${model}-p${parallel}-c${context}.json" >/dev/null || {
          echo "  benchmark failed for ${model} p=${parallel} c=${context}" >&2
          continue
        }

      after="$(host_snapshot)"
      "$PYTHON" - "$model" "$parallel" "$context" "$before" "$after" \
        "${OUTDIR}/${model}-p${parallel}-c${context}.json" "$SUMMARY" <<'PYEOF'
import json, sys
model, parallel, context, before, after, report_path, summary_path = sys.argv[1:8]
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)
for level in report["reports"][0]["levels"]:
    row = {
        "model": model,
        "ollama_num_parallel": int(parallel),
        "context": int(context),
        "clients": level["concurrency"],
        "requests": level["requests"],
        "success_rate": level["success_rate"],
        "requests_per_second": level["requests_per_second"],
        "ttft_p95_ms": (level.get("time_to_first_token_ms") or {}).get("p95"),
        "latency_p50_ms": (level.get("latency_ms") or {}).get("p50"),
        "latency_p95_ms": (level.get("latency_ms") or {}).get("p95"),
        "latency_p99_ms": (level.get("latency_ms") or {}).get("p99"),
        "tokens_per_second": level["tokens_per_second_mean"],
        "khmer_quality": level["khmer_quality_mean"],
        "failures": level["failed"],
        "errors": level["errors"][:3],
        "host_before": json.loads(before),
        "host_after": json.loads(after),
    }
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"  clients={row['clients']:>2}  ttft_p95={row['ttft_p95_ms']}ms  "
        f"p95={row['latency_p95_ms']}ms  tok/s={row['tokens_per_second']}  "
        f"ok={row['success_rate']:.0%}"
    )
PYEOF
    done
  done
done

echo ""
echo "Raw rows: ${SUMMARY}"
"$PYTHON" load_test/analyze_results.py --input "$SUMMARY" --output "${OUTDIR}/recommendation-${STAMP}.md" || true
echo ""
echo "Record the recommended configuration in:"
echo "  configs/production/ollama_9b.yaml  (runtime.environment, measured:)"
echo "  reports/final_load_test.md"
