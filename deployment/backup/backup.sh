#!/usr/bin/env bash
# Backup and restore for production assets (Phase 22).
#
#   bash deployment/backup/backup.sh                    # create a backup
#   bash deployment/backup/backup.sh --list
#   bash deployment/backup/backup.sh --restore <archive> [--dry-run]
#   bash deployment/backup/backup.sh --verify <archive>
#
# Backed up: the vector index, company-data manifests, configuration, prompts,
# model metadata, evaluation reports and deployment scripts.
#
# NOT backed up: model binaries (reproducible from the adapter + base model, and
# they would dominate the archive), .env (secrets belong in the password
# manager), or raw customer documents (re-ingested from the source of truth).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BACKUP_DIR="${KHMERAI_BACKUP_DIR:-/usr/local/var/khmerai/backups}"
RETENTION_DAYS="${KHMERAI_BACKUP_RETENTION_DAYS:-30}"
GPG_RECIPIENT="${KHMERAI_BACKUP_GPG_RECIPIENT:-}"
ACTION="backup"
ARCHIVE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restore) ACTION="restore"; ARCHIVE="$2"; shift 2 ;;
    --verify)  ACTION="verify";  ARCHIVE="$2"; shift 2 ;;
    --list)    ACTION="list"; shift ;;
    --dir)     BACKUP_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

TARGETS=(
  data/index
  data/manifests
  configs
  prompts
  evaluation/golden
  deployment
  reports
)

mkdir -p "$BACKUP_DIR"

case "$ACTION" in
  list)
    echo "Backups in ${BACKUP_DIR}:"
    ls -lh "$BACKUP_DIR"/khmerai-*.tar.gz* 2>/dev/null || echo "  (none)"
    exit 0
    ;;

  verify)
    [[ -f "$ARCHIVE" ]] || { echo "no such archive: $ARCHIVE" >&2; exit 1; }
    echo "Verifying ${ARCHIVE} ..."
    if [[ "$ARCHIVE" == *.gpg ]]; then
      gpg --decrypt "$ARCHIVE" 2>/dev/null | tar -tzf - >/dev/null
    else
      tar -tzf "$ARCHIVE" >/dev/null
    fi
    echo "Archive is readable."
    if [[ -f "${ARCHIVE}.sha256" ]]; then
      (cd "$(dirname "$ARCHIVE")" && shasum -a 256 -c "$(basename "$ARCHIVE").sha256")
    else
      echo "warning: no .sha256 sidecar found" >&2
    fi
    exit 0
    ;;

  restore)
    [[ -f "$ARCHIVE" ]] || { echo "no such archive: $ARCHIVE" >&2; exit 1; }
    echo "Restoring from ${ARCHIVE}"
    echo "This will OVERWRITE: ${TARGETS[*]}"
    if (( DRY_RUN )); then
      echo ""
      echo "--dry-run: contents that would be restored:"
      if [[ "$ARCHIVE" == *.gpg ]]; then
        gpg --decrypt "$ARCHIVE" 2>/dev/null | tar -tzf - | head -50
      else
        tar -tzf "$ARCHIVE" | head -50
      fi
      exit 0
    fi
    read -r -p "Type 'restore' to continue: " CONFIRM
    [[ "$CONFIRM" == "restore" ]] || { echo "aborted"; exit 1; }

    # Snapshot the current state first, so a bad restore is itself reversible.
    PRE="${BACKUP_DIR}/pre-restore-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
    tar -czf "$PRE" "${TARGETS[@]}" 2>/dev/null || true
    echo "Current state saved to ${PRE}"

    if [[ "$ARCHIVE" == *.gpg ]]; then
      gpg --decrypt "$ARCHIVE" 2>/dev/null | tar -xzf - -C "$REPO_ROOT"
    else
      tar -xzf "$ARCHIVE" -C "$REPO_ROOT"
    fi
    echo "Restored."
    echo ""
    echo "Now verify and reload:"
    echo "  ls -l data/index/ACTIVE"
    echo "  curl -X POST -H \"X-Admin-Key: \$KHMERAI_ADMIN_API_KEY\" localhost:8000/v1/admin/reload"
    echo "  bash scripts/smoke_test.sh"
    exit 0
    ;;
esac

# --- create ------------------------------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="${BACKUP_DIR}/khmerai-${STAMP}.tar.gz"

EXISTING=()
for target in "${TARGETS[@]}"; do
  [[ -e "$target" ]] && EXISTING+=("$target")
done
if [[ ${#EXISTING[@]} -eq 0 ]]; then
  echo "nothing to back up" >&2
  exit 1
fi

# Record what produced this backup, so a restore is traceable.
MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT
{
  echo "{"
  echo "  \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"host\": \"$(hostname)\","
  echo "  \"git_commit\": \"$(git rev-parse HEAD 2>/dev/null || echo unknown)\","
  echo "  \"active_index\": \"$(readlink data/index/ACTIVE 2>/dev/null || echo none)\","
  echo "  \"targets\": [$(printf '"%s",' "${EXISTING[@]}" | sed 's/,$//')]"
  echo "}"
} > "$MANIFEST"
cp "$MANIFEST" .backup_manifest.json

echo "Creating ${ARCHIVE} ..."
tar -czf "$ARCHIVE" "${EXISTING[@]}" .backup_manifest.json
rm -f .backup_manifest.json

if [[ -n "$GPG_RECIPIENT" ]]; then
  echo "Encrypting for ${GPG_RECIPIENT} ..."
  gpg --yes --batch --encrypt --recipient "$GPG_RECIPIENT" --output "${ARCHIVE}.gpg" "$ARCHIVE"
  rm -f "$ARCHIVE"
  ARCHIVE="${ARCHIVE}.gpg"
else
  echo "WARNING: KHMERAI_BACKUP_GPG_RECIPIENT is not set - this backup is NOT encrypted." >&2
  echo "         The index contains company documents. Set a recipient in .env." >&2
fi

(cd "$(dirname "$ARCHIVE")" && shasum -a 256 "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")

echo "Created: ${ARCHIVE} ($(du -h "$ARCHIVE" | cut -f1))"

DELETED=$(find "$BACKUP_DIR" -name 'khmerai-*.tar.gz*' -mtime "+${RETENTION_DAYS}" -print -delete | wc -l | tr -d ' ')
[[ "$DELETED" -gt 0 ]] && echo "Pruned ${DELETED} backup(s) older than ${RETENTION_DAYS} days"

echo ""
echo "Verify it now - an untested backup is not a backup:"
echo "  bash deployment/backup/backup.sh --verify ${ARCHIVE}"
