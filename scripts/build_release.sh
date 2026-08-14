#!/usr/bin/env bash
# Assemble a release bundle after the gates pass (Phase 23).
#
#   bash scripts/build_release.sh --version 1.0.0
#   bash scripts/build_release.sh --version 1.0.0 --skip-eval
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
VERSION=""
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$VERSION" ]] || { echo "--version is required" >&2; exit 2; }

echo "=== Release ${VERSION} ==="

echo ""
echo "1. Working tree"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "   WARNING: uncommitted changes - the release will not be reproducible" >&2
  git status --short | head -10
fi
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "   commit ${COMMIT}"

echo ""
echo "2. Quality gates"
bash scripts/lint.sh
bash scripts/test.sh

echo ""
echo "3. Evaluation gates"
if (( SKIP_EVAL )); then
  echo "   skipped (--skip-eval) - NOT permitted for a production release"
else
  bash scripts/evaluate_all.sh || { echo "   evaluation gates failed - release blocked" >&2; exit 1; }
fi

echo ""
echo "4. Licence review"
"$PY" datasets/license_report.py --fail-on-unreviewed || {
  echo "   WARNING: unreviewed dataset sources are present" >&2
}

echo ""
echo "5. Bundle"
OUTDIR="dist/khmerai-${VERSION}"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
for item in configs prompts ollama deployment monitoring scripts docs requirements \
            evaluation/golden Makefile pyproject.toml README.md LICENSE .env.example; do
  [[ -e "$item" ]] && cp -R "$item" "$OUTDIR/" 2>/dev/null || true
done
mkdir -p "$OUTDIR/reports"
cp -R evaluation/reports/*.md "$OUTDIR/reports/" 2>/dev/null || true
cp -R reports/*.md "$OUTDIR/reports/" 2>/dev/null || true

"$PY" - "$VERSION" "$COMMIT" "$OUTDIR" <<'PYEOF'
import json, sys, pathlib, subprocess
sys.path.insert(0, ".")
from common.config import load_config
version, commit, outdir = sys.argv[1:4]
versions = load_config("configs/base.yaml").get("versions", {})
manifest = {
    "release_version": version,
    "code_commit": commit,
    "versions": versions,
    "active_index": (
        str(pathlib.Path("data/index/ACTIVE").readlink())
        if pathlib.Path("data/index/ACTIVE").is_symlink() else None
    ),
    "python": sys.version.split()[0],
    "rollback": {
        "application": "git checkout <previous-tag> && bash deployment/macos/install.sh",
        "model": "ollama cp khmer-support-9b-previous khmer-support-9b",
        "prompt": "git checkout <previous-tag> -- prompts/ && POST /v1/admin/reload",
        "index": "python -m rag.reindex --rollback && POST /v1/admin/reload",
    },
}
pathlib.Path(outdir, "RELEASE.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PYEOF

tar -czf "dist/khmerai-${VERSION}.tar.gz" -C dist "khmerai-${VERSION}"
echo ""
echo "Bundle: dist/khmerai-${VERSION}.tar.gz"
echo ""
echo "Before deploying, complete docs/release_checklist.md."
