#!/usr/bin/env bash
# Format check, lint, type check and secret scan (CI stage 1).
#
#   bash scripts/lint.sh
#   bash scripts/lint.sh --fix
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
FIX=0
[[ "${1:-}" == "--fix" ]] && FIX=1

STATUS=0
run() {
  local label="$1"; shift
  echo "==> ${label}"
  if "$@"; then echo "    ok"; else echo "    FAILED"; STATUS=1; fi
}

if (( FIX )); then
  run "ruff format" "$PY" -m ruff format .
  run "ruff check --fix" "$PY" -m ruff check --fix .
else
  run "ruff format --check" "$PY" -m ruff format --check .
  run "ruff check" "$PY" -m ruff check .
fi

if "$PY" -c "import mypy" 2>/dev/null; then
  run "mypy" "$PY" -m mypy common preprocessing company_data rag server security evaluation training
else
  echo "==> mypy"; echo "    skipped (not installed)"
fi

run "secret scan" "$PY" -m security.secret_scanner . --fail-on high

echo ""
[[ $STATUS -eq 0 ]] && echo "All checks passed." || echo "Some checks failed." >&2
exit $STATUS
