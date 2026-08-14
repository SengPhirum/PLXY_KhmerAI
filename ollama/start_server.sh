#!/usr/bin/env bash
# Start the Ollama daemon with the tuned production environment.
#
#   bash ollama/start_server.sh                     # foreground
#   bash ollama/start_server.sh --profile 4b        # 4B tuning
#   bash ollama/start_server.sh --print-env         # show the settings only
#
# Values come from .env when present, else from the documented starting points.
# Replace them with the winners from `bash ollama/benchmark.sh` (Phase 13).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="9b"
PRINT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --print-env) PRINT_ONLY=1; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE:-64}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

if [[ "$PROFILE" == "4b" ]]; then
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-8}"
else
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
fi

echo "Ollama runtime environment (profile: ${PROFILE})"
for var in OLLAMA_HOST OLLAMA_CONTEXT_LENGTH OLLAMA_NUM_PARALLEL OLLAMA_MAX_QUEUE \
           OLLAMA_KEEP_ALIVE OLLAMA_FLASH_ATTENTION OLLAMA_KV_CACHE_TYPE OLLAMA_MAX_LOADED_MODELS; do
  printf "  %-28s %s\n" "$var" "${!var}"
done
echo ""
echo "These are STARTING POINTS. Replace them with the measured winners from"
echo "  bash ollama/benchmark.sh --model khmer-support-${PROFILE}"
echo ""

[[ "$PRINT_ONLY" -eq 1 ]] && exit 0

command -v ollama >/dev/null 2>&1 || { echo "ollama is not installed" >&2; exit 1; }

if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already running on ${OLLAMA_HOST}."
  echo "Environment changes need a restart:  pkill ollama && bash ollama/start_server.sh"
  exit 0
fi

echo "Starting ollama serve ..."
exec ollama serve
