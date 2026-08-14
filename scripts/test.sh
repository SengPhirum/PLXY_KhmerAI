#!/usr/bin/env bash
# Run the test suite (CI stage 2).
#
#   bash scripts/test.sh            # everything hermetic
#   bash scripts/test.sh --unit     # unit tests only
#   bash scripts/test.sh --coverage
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

ARGS=(-q)
case "${1:-}" in
  --unit)     ARGS+=(-m "not integration and not e2e and not slow") ;;
  --coverage) ARGS+=(--cov=. --cov-report=term-missing --cov-report=html) ;;
  --fast)     ARGS+=(-x -m "not slow") ;;
  "")         ;;
  *)          ARGS+=("$@") ;;
esac

# Tests must never reach a network service or a real model.
export KHMERAI_EMBEDDING_BACKEND=hashing
export KHMERAI_VECTOR_BACKEND=local
export KHMERAI_RATE_LIMIT_ENABLED=false
export KHMERAI_LOG_LEVEL=WARNING

echo "==> pytest ${ARGS[*]}"
"$PY" -m pytest "${ARGS[@]}"
