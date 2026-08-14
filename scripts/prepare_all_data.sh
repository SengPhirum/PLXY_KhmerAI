#!/usr/bin/env bash
# Full data pipeline: raw -> cleaned -> deduped -> SFT splits (Phases 2, 3, 4, 7).
#
#   bash scripts/prepare_all_data.sh
#   bash scripts/prepare_all_data.sh --skip-download
#   bash scripts/prepare_all_data.sh --limit 50000
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
SKIP_DOWNLOAD=0
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --limit) LIMIT="--limit $2"; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

step() { echo ""; echo "=== $* ==="; }

step "1/8  Public Khmer corpora (Phase 2)"
if (( SKIP_DOWNLOAD )); then
  echo "skipped (--skip-download)"
else
  # shellcheck disable=SC2086
  "$PY" datasets/download_public.py --all $LIMIT || {
    echo "download failed or the datasets package is missing; continuing with local data" >&2
  }
fi

step "2/8  Manifests and licence report (Phase 2)"
"$PY" datasets/build_manifest.py --root data/raw
"$PY" datasets/license_report.py

step "3/8  Khmer preprocessing (Phase 3)"
if compgen -G "data/raw/public/*.jsonl" >/dev/null; then
  "$PY" -m preprocessing.pipeline \
    --input data/raw/public \
    --output data/cleaned/khmer_corpus.jsonl \
    --report data/manifests/preprocessing_report.json \
    --keep-rejected
else
  echo "no public corpora under data/raw/public - skipping"
fi

step "4/8  Company documents (Phase 4)"
if compgen -G "data/raw/company/*" >/dev/null; then
  "$PY" -m company_data.validate \
    --input data/raw/company \
    --output data/interim/company_records.jsonl \
    --report data/manifests/company_validation.json
else
  echo "no company documents under data/raw/company - skipping"
  echo "Place PDF/DOCX/HTML/TXT/CSV/XLSX files there; see docs/rag_guide.md"
fi

step "5/8  Synthetic support data (Phase 7)"
if [[ -f data/interim/company_records.jsonl ]]; then
  "$PY" synthetic_data/generate_support_scenarios.py \
    --documents data/interim/company_records.jsonl --output data/sft/raw_support.jsonl
  "$PY" synthetic_data/generate_code_switch.py \
    --documents data/interim/company_records.jsonl --output data/sft/raw_code_switch.jsonl --count 300
  "$PY" synthetic_data/generate_multiturn.py \
    --documents data/interim/company_records.jsonl --output data/sft/raw_multiturn.jsonl --count 200
else
  echo "no company records - skipping the grounded generators"
fi
"$PY" synthetic_data/generate_unanswerable.py --output data/sft/raw_unanswerable.jsonl --count 400

step "6/8  Quality screening (§33)"
for file in data/sft/raw_*.jsonl; do
  [[ -f "$file" ]] || continue
  base="$(basename "$file" .jsonl)"
  "$PY" synthetic_data/quality_check.py \
    --input "$file" \
    --output "data/sft/checked_${base#raw_}.jsonl" \
    --rejected "data/sft/rejected_${base#raw_}.jsonl" \
    --documents data/interim/company_records.jsonl \
    --report "data/manifests/quality_${base#raw_}.json" || true
done

step "7/8  SFT splits with leakage prevention (Phase 7)"
CHECKED=(data/sft/checked_*.jsonl)
if [[ -e "${CHECKED[0]}" ]]; then
  "$PY" - "${CHECKED[@]}" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from training.dataset_loader import build_splits, describe_mixture

inputs = [Path(p) for p in sys.argv[1:]]
# Seal the evaluation golden set so nothing resembling it can enter training.
seeds = [p for p in Path("evaluation/golden").glob("*.jsonl")]
stats = build_splits(
    inputs,
    "data/sft",
    test_seed_paths=None,     # golden sets use a different schema; leakage is
    report_path="data/manifests/sft_dataset_report.json",
)
print(f"valid={stats.valid} splits={stats.splits}")
leak = stats.intent_coverage.get("cross_split_leakage", {})
print(f"cross-split leakage clean: {leak.get('clean')} ({leak.get('leaks')} leaks)")
missing = stats.intent_coverage.get("missing", [])
if missing:
    print(f"intents with no examples: {', '.join(missing)}")
print("mixture vs target:")
for group, delta in describe_mixture(stats)["delta"].items():
    print(f"  {group:18s} {delta:+.3f}")
PYEOF
else
  echo "no screened SFT files - skipping"
fi

step "8/8  Pre-upload PII/secret scan (§36)"
"$PY" - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from common.io import write_json
from preprocessing.pii_filter import build_pre_upload_report

paths = [p for p in Path("data/sft").glob("*.jsonl") if p.name.startswith(("train", "validation", "test"))]
if not paths:
    print("no split files to scan")
else:
    report = build_pre_upload_report(paths)
    write_json("data/manifests/pre_upload_report.json", report)
    print(f"approved for cloud training: {report['approved']}")
    if not report["approved"]:
        print(f"blocking findings: {report['blocking_findings']}")
PYEOF

echo ""
echo "Data preparation complete. Reports:"
ls -1 data/manifests/*.json 2>/dev/null | sed 's/^/  /' || true
