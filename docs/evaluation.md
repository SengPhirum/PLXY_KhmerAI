# Evaluation

## What is measured, and why each metric exists

| Dimension | Metric | Gate | Why this metric |
|---|---|---|---|
| Khmer language | fluency screen + native review | ≥ 4.0/5 | Automated screens catch mechanical defects only |
| Support accuracy | pass rate on the golden set | ≥ 0.85 | Did it resolve the customer's need? |
| Grounding | grounding precision | ≥ 0.90 | Is every business claim traceable? |
| Hallucination | invented-fact rate on unanswerable items | ≤ 0.03 | Does it say "I don't know"? |
| Retrieval | Recall@5, MRR, nDCG@10 | Recall@5 ≥ 0.85 | Did the right document surface? |
| Injection | block rate | ≥ 0.95 | Can a document redirect the assistant? |
| Latency | TTFT p95, e2e p95 | 2.5 s / 12 s | Customer-perceived speed |

## Running it

```bash
bash scripts/evaluate_all.sh                          # against a running API
bash scripts/evaluate_all.sh --backend ollama --model khmer-support-9b
bash scripts/evaluate_all.sh --baseline reports/baseline   # with regression check
```

Reports land in `evaluation/reports/` as JSON (machine) and Markdown (human).
Exit code 0 means every gate passed; the script names the gates that failed.

## Golden sets

Sealed test data in `evaluation/golden/`:

| File | Items | Tests |
|---|---:|---|
| `khmer_general.jsonl` | 10 | comprehension, summarisation, rewriting, grammar |
| `customer_support.jsonl` | 15 | product, pricing, warranty, policy, escalation |
| `code_switch.jsonl` | 8 | Khmer + English identifiers, URLs, currency |
| `multiturn.jsonl` | 5 | context retention, clarification, escalation |
| `hallucination.jsonl` | 12 | fake products, absent info, live data, competitors |
| `adversarial.jsonl` | 20 | injection, extraction, business-fact manipulation |

These are **sealed**: `preprocessing/near_dedup.py::LeakageChecker` rejects any
training record that near-duplicates one of them, and the check runs before the
train/validation/test split so a paraphrase cannot cross splits either.

Grow them from production failures. A real bad answer, added here with the
correct behaviour recorded, is worth more than ten synthetic cases.

## The limits of automated scoring

`khmer_fluency` reliably catches: wrong language, orphan diacritics, repetition
loops, Latin leakage, corrupted model numbers. It says **nothing** about whether
the Khmer is idiomatic, polite in the right register, or natural rather than
translation-like. Items that pass the screen are flagged `needs_human_review`,
and the report says how many.

Automated screening is a filter, not a verdict.

## Human review rubric

Native Khmer reviewers score 1–5 on six dimensions:

| Dimension | Question |
|---|---|
| Naturalness | Does it read like a Khmer speaker wrote it, not a translation? |
| Correctness | Are the facts right, given the source documents? |
| Helpfulness | Does it actually resolve the customer's need? |
| Professional tone | Polite, calm, appropriate for a support channel? |
| Faithfulness | Every claim traceable to a source; nothing invented? |
| Clarity | Unambiguous, well structured, appropriately concise? |

**Faithfulness or Correctness below 3 blocks a release regardless of the mean.**
An answer that is fluent, polite and wrong is worse than an awkward correct one,
because it is more likely to be believed.

Sample at least 100 answers per release, weighted toward pricing, warranty,
refunds and policy — the categories where an error creates a liability.

## Regression: candidate versus production

No checkpoint ships without this comparison.

```bash
python -m evaluation.evaluate_regression \
    --candidate evaluation/reports/customer_support.json \
    --baseline  reports/baseline/customer_support.json
```

Verdicts: `accept` (improvements, no regressions), `review` (a non-blocking
regression — needs sign-off), `reject` (a regression on a blocking metric:
hallucination, grounding, support accuracy, recall, injection block rate).

Exit codes 0 / 1 / 2 respectively, so it is usable directly as a CI gate.

## Model selection: 4B versus 9B

Decided by measured data (§29), never by parameter count.

```bash
python -m evaluation.benchmark_model --compare khmer-support-4b,khmer-support-9b \
    --concurrency 1,5,10
```

| Category | Source |
|---|---|
| Khmer quality | `evaluate_language.py` + native review |
| Support accuracy | `evaluate_support.py` |
| Hallucination | `evaluate_hallucination.py` |
| RAG grounding | `evaluate_grounding.py` |
| TTFT, tokens/sec | `benchmark_model.py` |
| p95 at 10 clients | `load_test/async_benchmark.py` |
| Peak memory | Activity Monitor / `ollama ps` |
| Stability | sustained load test |

The 4B is the model to beat on p95 at 10 clients, because smaller weights leave
far more unified memory for KV cache — and KV cache, not compute, is what limits
concurrency on 48 GB. If the 4B meets the quality gates, it is very likely the
better production choice.

## Quantization

Khmer is unusually sensitive to aggressive quantization. Build every level and
measure quality, Khmer integrity, memory, tokens/sec and TTFT before choosing.
Do not default to the smallest file (§Phase 12).

## What has not been measured here

This repository's build environment has no GPU, no model weights and no Ollama,
so **no model quality number in this project has been measured**. Every gate,
threshold and command is defined and executable; the report templates in
`reports/` are empty of results by design. Filling them in with anything other
than a real run would violate §1.2.

To produce real numbers: build a model (`docs/training_guide.md`), deploy it
(`docs/deployment_guide.md`), then run `bash scripts/evaluate_all.sh`.
