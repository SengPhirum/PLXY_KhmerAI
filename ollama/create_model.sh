#!/usr/bin/env bash
# Create (or rebuild) the Ollama models from the Modelfiles.
#
#   bash ollama/create_model.sh                       # both models
#   bash ollama/create_model.sh --model 9b            # one model
#   bash ollama/create_model.sh --gguf path.gguf --name khmer-support-9b-Q5_K_M
#   bash ollama/create_model.sh --verify              # run the Khmer smoke prompts
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_FILTER="all"
GGUF=""
NAME=""
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_FILTER="$2"; shift 2 ;;
    --gguf)  GGUF="$2"; shift 2 ;;
    --name)  NAME="$2"; shift 2 ;;
    --verify) VERIFY_ONLY=1; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v ollama >/dev/null 2>&1 || {
  echo "ollama is not installed. Install it with:  brew install ollama" >&2
  exit 1
}

if ! curl -sf "http://${OLLAMA_HOST:-127.0.0.1:11434}/api/tags" >/dev/null 2>&1; then
  echo "The Ollama daemon is not responding on ${OLLAMA_HOST:-127.0.0.1:11434}." >&2
  echo "Start it with:  bash ollama/start_server.sh" >&2
  exit 1
fi

# --- Khmer verification prompts (§Phase 12 verification) ---------------------
verify_model() {
  local model="$1"
  echo ""
  echo "=== Verifying ${model} ==="
  local -a prompts=(
    "សួស្តី តើអ្នកអាចជួយអ្វីបានខ្លះ?"
    "តើ model QN-4500A មាន warranty ប៉ុន្មានឆ្នាំ?"
    "សូមប្រាប់លក្ខណៈបច្ចេកទេសនៃម៉ូដែល RF-22B"
    "តម្លៃ ១២០ ដុល្លារ ស្មើនឹងប៉ុន្មានរៀល បើ ១ ដុល្លារ ស្មើ ៤០០០ រៀល?"
    "សូមពន្យល់លម្អិតអំពីរបៀបថែទាំទូរទឹកកកឱ្យបានយូរអង្វែង រួមទាំងការសម្អាត និងការកំណត់សីតុណ្ហភាព។"
  )
  local -a labels=(khmer_greeting code_switch model_number khmer_numerals long_khmer)
  local i=0
  for prompt in "${prompts[@]}"; do
    printf -- "--- %s ---\n" "${labels[$i]}"
    ollama run "$model" "$prompt" 2>&1 | head -12
    printf "\n"
    i=$((i + 1))
  done
  echo "Check each answer for: Khmer output, intact model numbers, no mojibake,"
  echo "no orphan diacritics, and no runaway repetition."
}

build_model() {
  local name="$1" modelfile="$2"
  if [[ ! -f "$modelfile" ]]; then
    echo "missing Modelfile: $modelfile" >&2
    return 1
  fi
  local gguf
  gguf="$(awk '/^FROM /{print $2; exit}' "$modelfile")"
  # Resolve relative to the Modelfile's directory, as Ollama does.
  local resolved="$(cd "$(dirname "$modelfile")" && cd "$(dirname "$gguf")" 2>/dev/null && pwd)/$(basename "$gguf")" || resolved="$gguf"
  if [[ ! -f "$resolved" ]]; then
    echo "" >&2
    echo "GGUF not found: $gguf (resolved: $resolved)" >&2
    echo "Produce it first:" >&2
    echo "    python training/merge_lora.py --base Qwen/Qwen3.5-9B --adapter outputs/sft_9b/adapter --output models/khmer-support-9b-merged" >&2
    echo "    python training/export_model.py --merged models/khmer-support-9b-merged --outdir models/gguf" >&2
    return 1
  fi
  echo "Creating ${name} from ${modelfile} ..."
  ollama create "$name" -f "$modelfile"
  ollama show "$name" --modelfile >/dev/null
  echo "OK: ${name}"
}

if [[ -n "$GGUF" ]]; then
  [[ -n "$NAME" ]] || { echo "--gguf requires --name" >&2; exit 2; }
  TMP_MODELFILE="$(mktemp)"
  trap 'rm -f "$TMP_MODELFILE"' EXIT
  # Reuse the 9B parameter block, swapping only the weights - this is how the
  # quantization comparison keeps everything except the GGUF identical.
  sed "s|^FROM .*|FROM ${GGUF}|" ollama/Modelfile.9b > "$TMP_MODELFILE"
  ollama create "$NAME" -f "$TMP_MODELFILE"
  echo "OK: ${NAME}"
  [[ "$VERIFY_ONLY" -eq 1 ]] && verify_model "$NAME"
  exit 0
fi

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "9b" ]] && verify_model khmer-support-9b || true
  [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "4b" ]] && verify_model khmer-support-4b || true
  exit 0
fi

STATUS=0
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "9b" ]]; then
  build_model khmer-support-9b ollama/Modelfile.9b || STATUS=1
fi
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "4b" ]]; then
  build_model khmer-support-4b ollama/Modelfile.4b || STATUS=1
fi

echo ""
ollama list
echo ""
echo "Verify Khmer generation with:  bash ollama/create_model.sh --verify"
exit "$STATUS"
