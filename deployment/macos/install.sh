#!/usr/bin/env bash
# Mac Studio production installer (Phase 20).
#
#   bash deployment/macos/install.sh
#   bash deployment/macos/install.sh --prefix /usr/local/opt/khmerai --user
#   bash deployment/macos/install.sh --check     # verify only, change nothing
#
# Steps: verify hardware -> install prerequisites -> configure Ollama ->
# create directories -> install the Python service -> configure the environment
# -> install launchd services -> start Ollama -> start the API -> health check.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PREFIX="${KHMERAI_PREFIX:-/usr/local/opt/khmerai}"
LOG_DIR="/usr/local/var/log/khmerai"
DATA_DIR="/usr/local/var/khmerai"
CHECK_ONLY=0
USER_MODE=0
SERVICE_USER="${SUDO_USER:-$(whoami)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    --user) USER_MODE=1; PREFIX="$HOME/.khmerai"; LOG_DIR="$HOME/Library/Logs/khmerai"; DATA_DIR="$HOME/.khmerai/var"; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ok()   { printf "  \033[32m[ok]\033[0m   %s\n" "$*"; }
warn() { printf "  \033[33m[warn]\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m[fail]\033[0m %s\n" "$*"; }

# --- 1. verify hardware ------------------------------------------------------
echo "1. Hardware"
FAILED=0
if [[ "$(uname)" != "Darwin" ]]; then
  fail "this installer targets macOS; found $(uname)"
  FAILED=1
else
  ok "macOS $(sw_vers -productVersion)"
fi

ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
  ok "Apple silicon ($(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo arm64))"
else
  warn "architecture is ${ARCH}; the target is Apple silicon (M4-class). Metal acceleration will not be available."
fi

MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1000000000 ))
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 0)
if (( MEM_GB >= 40 )); then
  ok "${MEM_GB} GB unified memory, ${CORES} CPU cores"
elif (( MEM_GB > 0 )); then
  warn "${MEM_GB} GB unified memory (the 9B profile assumes 48 GB). Use the 4B model, or reduce OLLAMA_NUM_PARALLEL and OLLAMA_CONTEXT_LENGTH."
fi

DISK_FREE_GB=$(df -g / | awk 'NR==2 {print $4}')
if (( DISK_FREE_GB >= 60 )); then
  ok "${DISK_FREE_GB} GB free disk"
else
  warn "${DISK_FREE_GB} GB free disk; model weights plus GGUF conversions need roughly 60 GB"
fi

# --- 2. prerequisites --------------------------------------------------------
echo ""
echo "2. Prerequisites"
if command -v python3 >/dev/null 2>&1; then
  PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  case "$PY_VERSION" in
    3.11|3.12) ok "python ${PY_VERSION}" ;;
    *) warn "python ${PY_VERSION}; 3.11 or 3.12 is validated. Install with: brew install python@3.11" ;;
  esac
else
  fail "python3 not found. Install with: brew install python@3.11"; FAILED=1
fi

if command -v ollama >/dev/null 2>&1; then
  ok "ollama $(ollama --version 2>/dev/null | head -1)"
else
  fail "ollama not found. Install with: brew install ollama"; FAILED=1
fi

command -v nginx >/dev/null 2>&1 && ok "nginx present (optional reverse proxy)" || warn "nginx not installed (only needed when exposing beyond the trusted LAN)"

if (( FAILED )); then
  echo ""
  echo "Resolve the failures above and re-run." >&2
  exit 1
fi

if (( CHECK_ONLY )); then
  echo ""
  echo "Check complete. Nothing was changed."
  exit 0
fi

# --- 3. directories ----------------------------------------------------------
echo ""
echo "3. Directories"
SUDO=""
[[ "$USER_MODE" -eq 0 && "$(id -u)" -ne 0 ]] && SUDO="sudo"
for dir in "$PREFIX" "$LOG_DIR" "$DATA_DIR" "$DATA_DIR/backups" "$DATA_DIR/index"; do
  $SUDO mkdir -p "$dir"
  [[ -n "$SUDO" ]] && $SUDO chown "$SERVICE_USER" "$dir"
  ok "$dir"
done

# --- 4. application ----------------------------------------------------------
echo ""
echo "4. Application"
if [[ "$(cd "$PREFIX" 2>/dev/null && pwd)" != "$REPO_ROOT" ]]; then
  $SUDO rsync -a --delete \
    --exclude '.git' --exclude '.venv' --exclude 'data/raw' --exclude 'outputs' \
    --exclude '__pycache__' --exclude '.pytest_cache' \
    "$REPO_ROOT/" "$PREFIX/"
  ok "application copied to $PREFIX"
else
  ok "running in place at $PREFIX"
fi

cd "$PREFIX"
python3 -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet -r requirements/server.txt -r requirements/rag.txt
ok "python environment installed"

# --- 5. environment ----------------------------------------------------------
echo ""
echo "5. Environment"
if [[ ! -f "$PREFIX/.env" ]]; then
  cp "$PREFIX/.env.example" "$PREFIX/.env"
  ADMIN_KEY="$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')"
  # Use a temp file rather than sed -i to avoid the BSD/GNU sed difference.
  python3 - "$PREFIX/.env" "$ADMIN_KEY" <<'PYEOF'
import sys, pathlib
path, key = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text(encoding="utf-8")
text = text.replace("KHMERAI_ADMIN_API_KEY=CHANGE_ME_ADMIN_KEY", f"KHMERAI_ADMIN_API_KEY={key}")
text = text.replace("KHMERAI_ENV=development", "KHMERAI_ENV=production")
path.write_text(text, encoding="utf-8")
PYEOF
  chmod 600 "$PREFIX/.env"
  ok ".env created with a generated admin key (mode 600)"
  echo "        Admin key: ${ADMIN_KEY}"
  echo "        Store it in the team password manager now - it is not shown again."
else
  ok ".env already exists (left untouched)"
fi

# --- 6. Ollama model ---------------------------------------------------------
echo ""
echo "6. Ollama"
if ! curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
  warn "the Ollama daemon is not running; launchd will start it below"
fi
if curl -sf "http://127.0.0.1:11434/api/tags" 2>/dev/null | grep -q "khmer-support"; then
  ok "a khmer-support model is already present"
else
  warn "no khmer-support model found. Build it with:  bash ollama/create_model.sh"
fi

# --- 7. launchd --------------------------------------------------------------
echo ""
echo "7. launchd services"
if [[ "$USER_MODE" -eq 1 ]]; then
  PLIST_DIR="$HOME/Library/LaunchAgents"; LOAD="launchctl load -w"
else
  PLIST_DIR="/Library/LaunchDaemons"; LOAD="$SUDO launchctl load -w"
fi
mkdir -p "$PLIST_DIR" 2>/dev/null || $SUDO mkdir -p "$PLIST_DIR"

for plist in com.company.khmerai.ollama.plist com.company.khmerai.api.plist; do
  python3 - "$REPO_ROOT/deployment/macos/$plist" "$PLIST_DIR/$plist" "$PREFIX" "$LOG_DIR" "$SERVICE_USER" <<'PYEOF'
import sys, pathlib
source, target, prefix, log_dir, user = sys.argv[1:6]
text = pathlib.Path(source).read_text(encoding="utf-8")
text = (text.replace("__PREFIX__", prefix)
            .replace("__LOG_DIR__", log_dir)
            .replace("__USER__", user))
pathlib.Path(target).write_text(text, encoding="utf-8")
PYEOF
  $SUDO chmod 644 "$PLIST_DIR/$plist" 2>/dev/null || chmod 644 "$PLIST_DIR/$plist"
  $LOAD "$PLIST_DIR/$plist" 2>/dev/null || warn "$plist was already loaded"
  ok "$plist"
done

# --- 8. health ---------------------------------------------------------------
echo ""
echo "8. Health check"
for _ in $(seq 1 30); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  sleep 2
done
if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  ok "API is live"
  curl -s http://127.0.0.1:8000/health | python3 -m json.tool | sed 's/^/        /'
  READY="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/ready)"
  [[ "$READY" == "200" ]] && ok "API is ready" || warn "/ready returned ${READY}; run: curl -s localhost:8000/ready | python3 -m json.tool"
else
  fail "the API did not become healthy"
  echo "        Logs: tail -f ${LOG_DIR}/api.err.log" >&2
  exit 1
fi

cat <<SUMMARY

Installation complete.

  Application   ${PREFIX}
  Logs          ${LOG_DIR}
  Data          ${DATA_DIR}

Next steps
  1. Build the knowledge index:
       cd ${PREFIX} && ./.venv/bin/python -m company_data.validate \\
           --input data/raw/company --output data/interim/company_records.jsonl \\
           --report data/manifests/company_validation.json
       ./.venv/bin/python -m rag.reindex --input data/interim/company_records.jsonl --activate
  2. Smoke test:      bash scripts/smoke_test.sh
  3. Tune Ollama:     bash ollama/benchmark.sh --model khmer-support-9b
  4. Firewall + TLS:  see docs/deployment_guide.md
SUMMARY
