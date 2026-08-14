# Khmer Customer-Support LLM

A Khmer-first customer-support assistant that runs entirely on premise, on a Mac
Studio, through Ollama. It answers customer questions about products, services,
pricing, warranty, policies and troubleshooting — in natural Khmer, grounded in
approved company documents, and it says "I don't know" rather than inventing a
price.

```
Customer (Khmer)  →  FastAPI  →  guardrails  →  RAG over company docs
                                             →  fine-tuned Qwen via Ollama
                                             →  grounding check  →  Khmer answer + sources
```

---

## 1. What it does

- Understands natural Khmer, including informal phrasing, common misspellings and
  Khmer-English code switching.
- Answers in fluent, professional Khmer by default.
- Uses **only** verified company documents for business facts — prices, warranty,
  stock, policies, specifications.
- Preserves model numbers, SKUs, URLs, currencies and dates verbatim.
- Refuses to invent. Unknown product, absent policy, expired promotion → it says
  so and offers a route to a human.
- Updates company knowledge in minutes, without retraining.
- Serves 10+ connected clients with bounded, measured latency.

## 2. Architecture

The decision everything else follows from:

> **Model weights hold *how to talk*. The retrieval index holds *what is true*.**

Prices and policies change weekly. Encoding them in weights would mean retraining
for every price change, and would make a wrong answer unfixable without a
training run. So they live in an index that rebuilds in minutes and rolls back in
seconds.

```
                       ┌──────────────────────────────────────────────┐
Company documents ──►  │ loaders → normalise → validate → chunk       │
(PDF/DOCX/HTML/        │        → embed → index → regression gate     │
 TXT/CSV/XLSX)         │        → atomic swap → ACTIVE                │
                       └───────────────────┬──────────────────────────┘
                                           │
Public Khmer corpora ──► clean → dedup ──► SFT ──► QLoRA ──► merge ──► GGUF ──► Ollama
                                           │                                     │
                                           └──────────────► FastAPI ◄────────────┘
                                                              │
                                                          Customer
```

Full detail: [`docs/architecture.md`](docs/architecture.md).

## 3. Hardware

| | Production | Training |
|---|---|---|
| Machine | Mac Studio, Apple M4-class | Google Colab or a Linux CUDA host |
| CPU / GPU | 16 cores / 40 GPU cores | T4, L4, A100 40GB or A100 80GB |
| Memory | 48 GB unified | 16–80 GB VRAM |
| Runtime | Ollama (Metal) | PyTorch + PEFT + TRL |
| Models | Qwen3.5-9B (quality), Qwen3.5-4B (concurrency) | QLoRA / LoRA |

Company documents never leave the premises. Cloud training requires sanitised
data and a recorded approval ([§35–36](docs/security.md)).

## 4. Quick start

```bash
git clone <repo> && cd PLXY_KhmerAI
bash scripts/bootstrap.sh            # venv + dependencies + directories + .env
source .venv/bin/activate
make doctor                          # what is missing, and the command that fixes it
make test                            # 513 hermetic tests, no model needed
```

To serve, you also need a model and an index — sections 6–8.

## 5. Dataset preparation

```bash
bash scripts/prepare_all_data.sh
```

Downloads the reviewed public Khmer corpora, writes provenance manifests and the
licence report, runs the Khmer preprocessing pipeline, ingests company documents,
generates and screens synthetic support data, and builds leakage-checked
train/validation/test/adversarial splits.

Check `data/manifests/sft_dataset_report.json` for cross-split leakage and intent
coverage. Details: [`datasets/README.md`](datasets/README.md).

## 6. Training

```bash
# Decide whether continued pretraining is needed at all (usually it is not)
python -m evaluation.evaluate_language --backend ollama --model qwen35-9b-base

# Fine-tune
python training/train_sft.py --config configs/training/sft_9b.yaml --dry-run
python training/train_sft.py --config configs/training/sft_9b.yaml

# Merge, export, quantize
python training/merge_lora.py --base Qwen/Qwen3.5-9B \
    --adapter outputs/sft_9b/adapter --output models/khmer-support-9b-merged
python training/export_model.py --merged models/khmer-support-9b-merged \
    --outdir models/gguf --quantize Q8_0,Q5_K_M,Q4_K_M
```

Details: [`docs/training_guide.md`](docs/training_guide.md).

## 7. RAG

```bash
# Put documents under data/raw/company/, with front-matter metadata
python -m company_data.validate --input data/raw/company \
    --output data/interim/company_records.jsonl \
    --report data/manifests/company_validation.json
python -m rag.reindex --input data/interim/company_records.jsonl --activate
```

Details: [`docs/rag_guide.md`](docs/rag_guide.md).

## 8. Ollama

```bash
bash ollama/start_server.sh &
bash ollama/create_model.sh
bash ollama/create_model.sh --verify     # Khmer smoke prompts
bash ollama/benchmark.sh --model khmer-support-9b
```

Details: [`ollama/README.md`](ollama/README.md).

## 9. API

```bash
make serve          # development
make serve-prod     # as launchd runs it
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness — never touches Ollama |
| `GET /ready` | readiness — Ollama, model, index, disk, capacity |
| `GET /metrics` | Prometheus |
| `POST /v1/chat` | answer |
| `POST /v1/chat/stream` | answer as server-sent events |
| `POST /v1/rag/search` | retrieval only |
| `GET /v1/models` | served models and all five versions |
| `POST /v1/admin/reindex` | build/activate an index *(admin)* |
| `POST /v1/admin/reload` | hot-swap the index and prompts *(admin)* |
| `GET /v1/admin/diagnostics` | versions, config, host stats *(admin)* |
| `DELETE /v1/conversations/{id}` | erase a conversation |

```bash
curl -X POST localhost:8000/v1/chat -H 'Content-Type: application/json' \
  -d '{"message":"តើម៉ូដែល QN-4500A មានការធានារយៈពេលប៉ុន្មាន?","stream":false}'
```

```json
{
  "conversation_id": "a1b2c3",
  "answer": "ម៉ូដែល QN-4500A មានការធានារយៈពេល ២៤ ខែ ចាប់ពីថ្ងៃទិញ។ [1]",
  "language": "km",
  "sources": [{"document_id": "a3f2", "title": "គោលការណ៍ធានា", "version": "2.0"}],
  "confidence": 0.81,
  "escalation_required": false,
  "grounded": true
}
```

## 10. Testing

```bash
make test               # everything hermetic (no model, no network)
make test-unit
make test-security
make coverage
```

513 tests: Khmer normalisation and script handling, ingestion and governance,
chunking, retrieval and grounding, guardrails, concurrency, the evaluation
framework, the training stack, and the 24 required production scenarios.

## 11. Evaluation

```bash
bash scripts/evaluate_all.sh
```

| Gate | Threshold |
|---|---:|
| Khmer fluency | ≥ 0.80 (+ native review ≥ 4.0/5) |
| Support accuracy | ≥ 0.85 |
| Grounding precision | ≥ 0.90 |
| Hallucination rate | ≤ 0.03 |
| Retrieval Recall@5 | ≥ 0.85 |
| Injection block rate | ≥ 0.95 |

Details: [`docs/evaluation.md`](docs/evaluation.md).

## 12. Deployment

```bash
bash deployment/macos/install.sh --check    # verify hardware, change nothing
bash deployment/macos/install.sh
bash scripts/smoke_test.sh
```

Details: [`docs/deployment_guide.md`](docs/deployment_guide.md).

## 13. Updating company documents

The point of the whole RAG layer — no retraining:

```bash
# edit a document under data/raw/company/
python -m company_data.validate --input data/raw/company \
    --output data/interim/company_records.jsonl \
    --report data/manifests/company_validation.json
python -m rag.reindex --input data/interim/company_records.jsonl --activate
curl -X POST -H "X-Admin-Key: $KHMERAI_ADMIN_API_KEY" localhost:8000/v1/admin/reload
```

Build → regression gate → atomic swap. The live index is never mutated, and
`python -m rag.reindex --rollback` undoes it.

## 14. Monitoring

```bash
prometheus --config.file=monitoring/prometheus.yml
```

Metrics: requests, active, queued, latency, TTFT, generated tokens, retrieval
latency, empty retrievals, Ollama errors, API errors, escalations, unknown
answers, rate limits, injection blocks, guardrail blocks. Alert rules trace back
to the SLOs in `configs/base.yaml`, each annotated with its runbook section.

## 15. Troubleshooting

[`docs/troubleshooting.md`](docs/troubleshooting.md) for development,
[`docs/operations_runbook.md`](docs/operations_runbook.md) for production
incidents.

## 16. Security

Layered, because no single control is sufficient: input guards, ingestion
quarantine, retrieval-time injection dropping, delimited untrusted context,
output leak detection, grounding verification, redaction in the log formatter,
rate limiting, admin authentication with constant-time comparison.

Conversations are in memory with a 1-hour TTL and do not survive a restart.

Details: [`docs/security.md`](docs/security.md).

## 17. Licensing

This software is proprietary (see `LICENSE`). Third-party components keep their
own licences:

- **Base models** (Qwen family) — review the model card before commercial use.
- **Datasets** — per-source licences tracked in
  [`docs/dataset_provenance.md`](docs/dataset_provenance.md) and
  `data/manifests/license_report.csv`. No source marked `review_required` may
  train a production model.
- **Python dependencies** — see `requirements/`.

---

## Project layout

```
common/          config layering, atomic IO, hashing, versioning, redacting logs
preprocessing/   Khmer script model, normalisation, quality, dedup, PII
company_data/    loaders, canonical schema, normalisation, validation
rag/             chunking, embeddings, vector stores, BM25, fusion, retrieval
prompts/         versioned Khmer/English system prompts and scaffolds
server/          FastAPI app, guardrails, conversation memory, metrics
security/        redaction, secret scanning, injection defence, access control
evaluation/      metrics, evaluators, golden sets, benchmarks
training/        SFT/CPT/DPO, merge, export, Colab notebooks
synthetic_data/  grounded generators + review workflow
datasets/        public corpus acquisition and provenance
ollama/          Modelfiles and runtime tuning
load_test/       load profiles and the tuning analyser
deployment/      macOS installer, launchd, nginx, backup
monitoring/      Prometheus config and alert rules
docs/            architecture, guides, runbook, checklists
```

## Status and honesty about what is measured

Every component is implemented and tested. **No model quality or performance
number in this repository has been measured**, because the environment it was
built in has no GPU, no model weights and no Ollama. The report templates in
`reports/` are deliberately empty of results, and every gate, threshold and
command needed to fill them in is defined and runnable.

What that means concretely:

| Verified here | Requires hardware to verify |
|---|---|
| 513 tests pass | Khmer answer quality from a real model |
| Khmer normalisation preserves valid text | Training convergence |
| Ingestion governance and conflict detection | Quantization quality trade-off |
| Retrieval, grounding, conflict exposure | Latency, TTFT, tokens/sec |
| Guardrails block injection and ungrounded claims | Concurrency at 10/20 clients |
| Atomic reindex, rollback, hot-swap | Peak unified memory |
| API contract, streaming, concurrency, rate limits | macOS launchd deployment |

Run `bash scripts/evaluate_all.sh` and `bash ollama/benchmark.sh` on the target
hardware to fill in the rest.
