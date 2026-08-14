# Training guide

## The decision that comes first

Before any GPU time: **does this base model already speak Khmer well enough?**

```bash
bash ollama/create_model.sh --gguf <base-model.gguf> --name qwen35-9b-base
python -m evaluation.evaluate_language --backend ollama --model qwen35-9b-base \
    --golden evaluation/golden/khmer_general.jsonl --report-name baseline_9b
python -m evaluation.evaluate_support  --backend ollama --model qwen35-9b-base \
    --report-name baseline_support_9b
```

Then read `evaluation/reports/baseline_9b.md`:

| Baseline result | Decision |
|---|---|
| Khmer fluency ≥ 0.80, natural on native review | **Skip continued pretraining.** Go straight to SFT. |
| Reads Khmer but the register is wrong for support | **Skip CPT.** This is an SFT problem, not a language-modelling one. |
| Mojibake, orphan diacritics, or systematic grammar failure | CPT is justified — see Phase 6. |

CPT costs GPU-days and readily damages reasoning and instruction-following on a
strong base model. The default answer is no.

## Order of work

```
Phase 5  baseline evaluation      ── decides whether Phase 6 happens at all
Phase 6  continued pretraining    ── optional, gated on measured deficiency
Phase 7  SFT dataset              ── 30k-80k examples, quality over quantity
Phase 8  QLoRA SFT                ── both 4B and 9B candidates
Phase 9  DPO                      ── optional, kept only if it beats SFT
Phase 12 merge + export + quantize
```

## Phase 7 — the dataset

Target 30,000–80,000 examples with this mixture (§Phase 7):

| Share | Content | Generator |
|---:|---|---|
| 40% | company/customer-support behaviour | `generate_support_scenarios.py` |
| 20% | general Khmer instruction and QA | curated public data |
| 10% | Khmer-English code switching | `generate_code_switch.py` |
| 10% | complaints, ambiguity, difficult interactions | `generate_support_scenarios.py` |
| 10% | unanswerable / anti-hallucination | `generate_unanswerable.py` |
| 5% | comparison and sales assistance | curated |
| 5% | multi-turn and escalation | `generate_multiturn.py` |

Build it:

```bash
bash scripts/prepare_all_data.sh
```

Then read `data/manifests/sft_dataset_report.json` and check three things:

1. `cross_split_leakage.clean` is `true` — no training record near-duplicates a
   validation or test record.
2. `intent_coverage.missing` is empty — every intent in the closed vocabulary has
   examples.
3. The mixture delta is small — `describe_mixture` prints it.

**The 10% unanswerable slice is the highest-value part of the dataset.** It is
what teaches the model that "I don't know" is a correct answer. Without it, no
amount of prompting reliably stops confident invention.

Synthetic data must be grounded: `synthetic_data/quality_check.py` rejects any
generated answer containing a fact absent from its source document. Route
`critical`/`high` priority samples (warranty, refund, pricing, policy, safety,
troubleshooting) to a native Khmer reviewer before training.

## Phase 8 — QLoRA SFT

### On Colab

Notebooks in `training/colab/` are restart-safe and resume from Drive.
Alternatively, from a terminal:

```bash
pip install -r requirements/training.txt
python training/train_sft.py --config configs/training/sft_9b.yaml --dry-run   # validate first
python training/train_sft.py --config configs/training/sft_9b.yaml
python training/train_sft.py --config configs/training/sft_9b.yaml --resume    # after a disconnect
```

**Before any cloud upload**, §36 applies. The training script enforces it:

```bash
python -c "
from preprocessing.pii_filter import build_pre_upload_report
from common.io import write_json
write_json('data/manifests/pre_upload_report.json',
           build_pre_upload_report(['data/sft/train.jsonl','data/sft/validation.jsonl']))"
```

`train_sft.py` refuses to run if that report says `approved: false`. Override
only with `--skip-upload-check`, and only for on-premise training.

### Memory profiles

| Profile | VRAM | Batch x accum | Seq len | 4-bit | Notes |
|---|---:|---|---:|:---:|---|
| `16gb` | 16 GB | 1 x 16 | 2048 | yes | T4/V100; fp16 compute |
| `24gb` | 24 GB | 2 x 8 | 3072 | yes | L4/A10/3090; fine for 4B |
| `40gb` | 40 GB | 4 x 4 | 4096 | yes | A100 40GB; **recommended for the 9B** |
| `80gb` | 80 GB | 8 x 2 | 4096 | no | LoRA on bf16; no quantisation |

```bash
python training/train_sft.py --config configs/training/sft_4b.yaml --memory-profile 24gb
```

### Two things that silently ruin a fine-tune

**LoRA targets.** Hard-coding `["q_proj", "k_proj", ...]` produces an adapter
that trains *nothing* if the model uses different names. `find_target_modules`
inspects the loaded model instead, excludes multimodal towers (this is a
text-only project), and logs what it found. Check that log line.

**The chat template.** Training must use exactly the template the model is served
with. `training/chat_template.py` prefers the tokenizer's own
`apply_chat_template` and logs loudly when it has to fall back. The `TEMPLATE`
block in `ollama/Modelfile.9b` must match. A mismatch produces a fine-tune that
appears to do nothing.

The run also asserts that the completion mask is sane: if fewer than 5% of tokens
carry loss, it raises rather than burning GPU hours on a broken mask.

### Parameter search

`configs/training/sft_9b.yaml` declares the grid (§Phase 8):

```
lora_rank      32, 64          (alpha ≈ 2x rank)
lora_dropout   0.0, 0.05
seq length     2048, 4096
learning rate  5e-5, 1e-4, 2e-4
warmup         3-5%
scheduler      cosine
grad clipping  1.0
```

Compare runs with `python -m evaluation.evaluate_regression`.

## Phase 9 — DPO, and when to reject it

```bash
python training/train_dpo.py --config configs/training/dpo.yaml --audit-only
```

The audit fails the phase if there are fewer than 1000 usable pairs, and warns
if "chosen" is almost always the longer answer — that teaches verbosity, the
opposite of the §32 generation policy.

Keep the checkpoint only if it wins:

```bash
python training/train_dpo.py --decide \
    evaluation/reports/sft_customer_support.json \
    evaluation/reports/dpo_customer_support.json
```

Exit code 0 keeps it; 1 rejects it. Any regression in hallucination, grounding or
Khmer fluency rejects it regardless of accuracy gains.

## Phase 12 — merge, export, quantize

```bash
python training/merge_lora.py --base Qwen/Qwen3.5-9B \
    --adapter outputs/sft_9b/adapter --output models/khmer-support-9b-merged
python training/export_model.py --merged models/khmer-support-9b-merged \
    --outdir models/gguf --quantize Q8_0,Q5_K_M,Q4_K_M
```

The merge refuses to run if the adapter was trained against a different base
model — merging mismatched weights produces something that appears to work and
is subtly broken.

**Do not choose the smallest file.** Khmer is unusually sensitive to aggressive
quantization. Build every level and measure:

```bash
for level in Q8_0 Q5_K_M Q4_K_M; do
  bash ollama/create_model.sh --gguf models/gguf/khmer-support-9b-$level.gguf \
      --name khmer-support-9b-$level
done
python -m evaluation.benchmark_model \
  --compare khmer-support-9b-Q4_K_M,khmer-support-9b-Q5_K_M,khmer-support-9b-Q8_0
python -m evaluation.evaluate_language --backend ollama --model khmer-support-9b-Q4_K_M
```

Record quality, Khmer integrity, memory, tokens/sec and TTFT per level in
`reports/final_model_evaluation.md`, then pick.

## Reproducibility

Every run writes `run_manifest.json` next to the checkpoint, containing the model
id and revision, dataset paths with SHA-256, the preprocessing fingerprint, the
code commit, the seed, every hyper-parameter, the installed package versions, and
the hardware. A result without its manifest is not a result.

## Expected artefacts

| Path | Contents |
|---|---|
| `outputs/<run>/adapter/` | LoRA weights + tokenizer |
| `outputs/<run>/run_manifest.json` | full reproducibility record |
| `outputs/<run>/metrics.json` | train/eval loss |
| `outputs/<run>/checkpoint-*/` | resumable checkpoints |
| `models/<name>-merged/` | merged HF weights + `merge_provenance.json` |
| `models/gguf/*.gguf` | one file per quantization level |

## Not executed here

No training was run in this repository's build environment — it has no GPU and no
model weights. Every command above is the exact command to run, `--dry-run`
validates configuration and data without a GPU (and runs in CI), and the artefact
table states precisely what a real run produces.
