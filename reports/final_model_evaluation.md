# Final model evaluation

> **Status: NOT YET MEASURED.** No model has been trained or served from this
> repository's build environment (no GPU, no weights, no Ollama). Fill this in
> from a real run; do not estimate.

Model: `________`  Adapter: `________`  Quantization: `________`
Prompt version: `________`  Date: `________`  Engineer: `________`

## How to produce these numbers

```bash
bash scripts/evaluate_all.sh --backend api
python -m evaluation.benchmark_model --compare khmer-support-4b,khmer-support-9b --concurrency 1,5,10
```

## Automated gates

| Gate | Threshold | Measured | Pass | Report |
|---|---:|---:|:---:|---|
| Khmer fluency (screen) | ≥ 0.80 | | | `khmer_language.md` |
| Code-switch handling | ≥ 0.80 | | | `code_switch.md` |
| Support accuracy | ≥ 0.85 | | | `customer_support.md` |
| Multi-turn | ≥ 0.85 | | | `multiturn.md` |
| Grounding precision | ≥ 0.90 | | | `grounding.md` |
| Hallucination rate | ≤ 0.03 | | | `hallucination.md` |
| Unsupported claim rate | ≤ 0.02 | | | `grounding.md` |
| Injection block rate | ≥ 0.95 | | | `hallucination.md` |

## Native Khmer review

Reviewers: `________`  Sample size: `________` (minimum 100)

| Dimension | Mean (1–5) | Below 3 |
|---|---:|---:|
| Naturalness | | |
| Correctness | | |
| Helpfulness | | |
| Professional tone | | |
| Faithfulness | | |
| Clarity | | |

Faithfulness or Correctness below 3 on any item blocks the release.

## Quantization comparison (§Phase 12 — do not pick the smallest file)

| Level | Size | Khmer quality | Tokens/s | TTFT p95 | Peak memory | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Q8_0 | | | | | | |
| Q5_K_M | | | | | | |
| Q4_K_M | | | | | | |

## Model selection (§29)

| Category | 4B | 9B | Winner |
|---|---:|---:|---|
| Khmer quality | | | |
| Support accuracy | | | |
| Hallucination | | | |
| RAG grounding | | | |
| TTFT | | | |
| Tokens/sec | | | |
| p95 at 10 clients | | | |
| Peak memory | | | |
| Stability | | | |

**Selected model**: `________`  **Justification**: `________`

## Known limitations

## Decision

- [ ] APPROVED  - [ ] CONDITIONAL  - [ ] BLOCKED
