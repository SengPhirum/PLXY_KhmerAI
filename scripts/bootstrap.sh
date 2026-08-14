#!/usr/bin/env bash
# Create the development environment from a fresh checkout (Phase 1).
#
#   bash scripts/bootstrap.sh              # base + server + rag + dev
#   bash scripts/bootstrap.sh --training   # add the GPU training stack
#   bash scripts/bootstrap.sh --minimal    # base only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python3}"
VENV=".venv"
PROFILE="dev"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --training) PROFILE="training"; shift ;;
    --minimal)  PROFILE="minimal"; shift ;;
    --python)   PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)  sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "==> Python"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "$PYTHON_BIN not found" >&2; exit 1; }
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VERSION" in
  3.11|3.12) echo "    ${PY_VERSION} ok" ;;
  *) echo "    WARNING: python ${PY_VERSION}; 3.11-3.12 is the validated range." >&2 ;;
esac

echo "==> Virtual environment"
[[ -d "$VENV" ]] || "$PYTHON_BIN" -m venv "$VENV"
PIP="$VENV/bin/python -m pip"
$PIP install --quiet --upgrade pip setuptools wheel
echo "    ${VENV}"

echo "==> Dependencies (${PROFILE})"
case "$PROFILE" in
  minimal)  $PIP install --quiet -r requirements/base.txt ;;
  dev)      $PIP install --quiet -r requirements/dev.txt ;;
  training) $PIP install --quiet -r requirements/dev.txt
            echo "    Installing the training stack (large download)..."
            $PIP install -r requirements/training.txt ;;
esac
echo "    done"

echo "==> Directories"
mkdir -p data/{raw/{public,company},interim,cleaned,deduped,sft,preference,evaluation,manifests,index} \
         evaluation/reports reports logs models outputs
echo "    ok"

echo "==> Configuration"
if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  echo "    .env created from .env.example (mode 600) - fill in the values"
else
  echo "    .env already exists"
fi

echo "==> Verification"
"$VENV/bin/python" -c "
import sys
sys.path.insert(0, '.')
from preprocessing import normalize_text, detect_language
sample = 'ការធានារយៈពេល ២៤ ខែ ចាប់ពីថ្ងៃទិញ។'
assert normalize_text(sample) == sample, 'Khmer normalisation altered valid text'
assert str(detect_language(sample)[0]).startswith('khmer')
print('    Khmer pipeline imports and round-trips correctly')
"

echo ""
echo "Bootstrap complete."
echo ""
echo "  source .venv/bin/activate"
echo "  make doctor      # check the environment"
echo "  make test        # run the test suite"
echo "  make serve       # start the API"
