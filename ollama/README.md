# Ollama layer

Everything needed to turn a fine-tuned checkpoint into the local production
runtime, and to tune that runtime for the Mac Studio.

## Files

| File | Purpose |
|---|---|
| `Modelfile.9b` | Production Modelfile for the quality candidate. |
| `Modelfile.4b` | Production Modelfile for the high-concurrency candidate. |
| `create_model.sh` | Build the models; `--verify` runs the Khmer smoke prompts. |
| `start_server.sh` | Start the daemon with the tuned environment. |
| `benchmark.sh` | Phase 13 tuning matrix (parallelism x context x clients). |

## Order of operations

```bash
# 1. Merge the adapter and export GGUF (on the training host)
python training/merge_lora.py --base Qwen/Qwen3.5-9B \
    --adapter outputs/sft_9b/adapter --output models/khmer-support-9b-merged
python training/export_model.py --merged models/khmer-support-9b-merged \
    --outdir models/gguf --quantize Q8_0,Q5_K_M,Q4_K_M

# 2. Build the model (on the Mac Studio)
bash ollama/start_server.sh &
bash ollama/create_model.sh

# 3. Verify Khmer generation before anything else
bash ollama/create_model.sh --verify

# 4. Choose the quantization from measurements, not from file size
for level in Q8_0 Q5_K_M Q4_K_M; do
  bash ollama/create_model.sh --gguf models/gguf/khmer-support-9b-$level.gguf \
      --name khmer-support-9b-$level
done
python -m evaluation.benchmark_model \
  --compare khmer-support-9b-Q4_K_M,khmer-support-9b-Q5_K_M,khmer-support-9b-Q8_0

# 5. Tune the runtime
bash ollama/benchmark.sh --model khmer-support-9b
```

## The two settings that matter most on 48 GB

`OLLAMA_NUM_PARALLEL` multiplies KV-cache memory: each parallel slot holds its
own cache for the full context. At 8192 context on a 9B model, going from 4 to 8
parallel roughly doubles KV memory, and on a 48 GB machine that is what pushes
the system into swap - where latency collapses rather than degrading gracefully.

`OLLAMA_KV_CACHE_TYPE=q8_0` halves KV memory against `f16` at negligible quality
cost, which is usually worth more than one extra parallel slot. Both claims are
measurable with `benchmark.sh`; measure them on the actual machine.

`OLLAMA_KEEP_ALIVE=-1` keeps the model resident. Without it the first request
after an idle period pays a multi-second model load, which shows up as a
catastrophic TTFT outlier in the p99.

## Verification prompts

`create_model.sh --verify` sends Khmer, code-switched, model-number, numeral and
long-form prompts. Check each answer for Khmer output, intact model numbers, no
mojibake, no orphan diacritics and no repetition loop. `evaluation/metrics.py`'s
`khmer_fluency` screens for the same defects automatically.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GGUF not found` | export not run | `python training/export_model.py ...` |
| Model answers in English | wrong system prompt or template mismatch | check `TEMPLATE` matches `training/chat_template.py` |
| Mojibake / dotted circles | quantization too aggressive, or a legacy-font source document | try Q5_K_M/Q8_0; check ingestion for legacy Khmer fonts |
| Repeats one phrase forever | `repeat_penalty` too low, temperature too low | raise `repeat_penalty` to 1.1; measure with `khmer_fluency` |
| Slow first token after idle | model unloaded | `OLLAMA_KEEP_ALIVE=-1` |
| 503s under load | `OLLAMA_NUM_PARALLEL` below API `max_active_generations` | keep API `max_active_generations <= OLLAMA_NUM_PARALLEL` |
