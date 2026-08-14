# Production Implementation Report

**Repository:** `SengPhirum/PLXY_KhmerAI` · **Branch:** `main` · **Date:** 2026-08-14
All verification commands below were run against the tree of the commit that introduces this file.
**Build environment:** Linux x86_64, Python 3.11.15, 4 CPU cores, 15 GB RAM, **no GPU, no Ollama, no
Apple silicon, no model weights.**

> **Measurement honesty (§1.2).** Every number in this report was produced by a command actually run
> in the build environment above. Sections 6, 7, 8 and 9 require a GPU host and a Mac Studio; those
> runs did **not** happen here, so those sections state that plainly and give the exact commands that
> produce the numbers. Nothing is estimated, extrapolated or copied from a model card.

---

## 1. Overall Status

**PARTIAL** — code-complete, measurement-incomplete.

| | |
|---|---|
| All 24 phases implemented as working code | ✅ |
| Test suite | ✅ 513 tests pass |
| Lint / format / type gates | ✅ `ruff check` clean, 128 files formatted, `mypy` clean over 106 source files |
| Secret scan of the tracked tree | ✅ 0 findings at `--fail-on high` |
| Model trained | ❌ requires a GPU host (Colab / A100 / H100) |
| Khmer quality, RAG quality, Mac Studio throughput measured | ❌ requires the target hardware |

`PARTIAL` is the only defensible status. Marking this `PASS` would assert quality and performance
results that were never measured, which §1.2 and §46 forbid.

---

## 2. Implemented Components

| Phase | Component | Where | State |
|---|---|---|---|
| 0 | Requirements, architecture, directory skeleton | `docs/architecture.md`, `configs/` | done |
| 1 | Bootstrap, pinned deps, `.env.example`, environment doctor | `scripts/bootstrap.sh`, `requirements/`, `scripts/doctor.py` | done |
| 2 | Public Khmer dataset acquisition with manifests + licence report | `datasets/` | code done, download not run (no network egress to HF) |
| 3 | Khmer preprocessing: syllable clustering, Unicode repair, detection, code-switching, quality, dedup, PII | `preprocessing/` | done, 142 tests |
| 4 | Company-document ingestion: 6 loaders, normalisation, conflict detection, validation | `company_data/` | done, 28 tests |
| 5 | Baseline Khmer evaluation harness | `evaluation/` | harness done, baseline not run |
| 6 | Continued pretraining (decision + trainer) | `training/train_cpt.py`, `docs/training_guide.md` | done, not run |
| 7 | SFT dataset construction + synthetic generation + review gate | `synthetic_data/`, `preprocessing/schemas.py` | done |
| 8 | QLoRA SFT, 4 memory profiles, completion-only loss masking | `training/train_sft.py` | done, not run |
| 9 | DPO preference optimisation | `training/train_dpo.py` | done, not run |
| 10 | RAG: chunking, embeddings, hybrid dense+BM25 with RRF, reranking, citations | `rag/` | done, 62 tests |
| 11 | Production prompting: 5 prompt files, budget validation, elastic context | `prompts/`, `server/chat_service.py` | done |
| 12 | Ollama export: merge → GGUF → quantise → Modelfile | `training/merge_lora.py`, `training/export_model.py`, `ollama/` | done, not run |
| 13 | Ollama parallelism / context benchmark matrix | `ollama/benchmark.sh` | script done, not run |
| 14 | FastAPI service: 9 endpoints, SSE streaming, admission control | `server/` | done, 80 tests |
| 15 | Conversation management + extractive summarisation | `server/models.py` | done |
| 16 | Security: injection defence, redaction, guardrails, RBAC, rate limiting | `security/`, `server/guardrails.py` | done, 91 tests |
| 17 | Evaluation framework + 70 sealed golden items | `evaluation/`, `evaluation/golden/` | done, 42 tests |
| 18 | Load testing (async + Locust, 5 scenarios) | `load_test/` | done, not run |
| 19 | Monitoring: Prometheus metrics, scrape config, alert rules | `server/metrics.py`, `monitoring/` | done |
| 20 | macOS deployment: installer, 2 launchd plists, nginx | `deployment/` | done, not run (no Mac) |
| 21 | Knowledge update: atomic build→gate→swap with rollback | `rag/reindex.py` | done |
| 22 | Backup / restore / verify | `deployment/backup/backup.sh` | done |
| 23 | CI and release gates | `.github/workflows/ci.yml`, `scripts/build_release.sh` | done |
| 24 | Final production validation | `tests/end_to_end/` (24 scenarios), `scripts/smoke_test.sh` | offline scenarios pass; live smoke test needs a running stack |

---

## 3. Repository Structure

223 tracked files · 128 Python modules · 27,021 lines of Python.

```
PLXY_KhmerAI/
├── common/            paths, atomic IO, hashing, config, versions, JSON logging
├── preprocessing/     Khmer script, Unicode repair, detection, mixing, quality, dedup, PII
├── company_data/      schema, 6 loaders, normalise, validate (conflict detection)
├── synthetic_data/    4 generators, review schema, quality check
├── datasets/          public download, manifest builder, licence report
├── rag/               chunking, embeddings, vector store, BM25, hybrid, rerank, citations, reindex
├── training/          common (memory profiles, manifests), chat template, SFT / CPT / DPO, merge, export
├── evaluation/        metrics, runner, 6 evaluators, benchmark, golden/ (70 sealed items)
├── server/            FastAPI app, chat + RAG services, guardrails, rate limit, metrics, health
├── security/          redaction, secret scanner, prompt injection, access control
├── load_test/         async benchmark, locustfile, analysis, 5 scenarios
├── monitoring/        prometheus.yml, alert rules
├── deployment/        macos/ (install + 2 plists), nginx/, backup/
├── ollama/            Modelfile.9b, Modelfile.4b, create/start/benchmark scripts
├── scripts/           bootstrap, doctor, lint, test, prepare_all_data, evaluate_all, release, smoke
├── prompts/           system_km, system_en, rag_context, escalation, refusal
├── configs/           base.yaml + models/ training/ rag/ production/
├── docs/              10 guides
├── reports/           5 result templates (empty by design) + this report
└── tests/             integration + end_to_end (24 production scenarios)
```

### Test distribution (513 tests, all passing)

| Area | Tests |
|---|---|
| `preprocessing/` | 142 |
| `security/` | 91 |
| `server/` | 80 |
| `rag/` | 62 |
| `evaluation/` (`tests/test_evaluation.py`) | 42 |
| `training/` | 39 |
| `company_data/` | 28 |
| `tests/end_to_end/` | 24 |
| `tests/integration/` | 5 |

---

## 4. Models

Nothing was downloaded, trained, merged, quantised or served in this environment. The following are
**configured targets**, not verified artefacts.

| Role | Configured identifier | Config | Status |
|---|---|---|---|
| Primary generator | `Qwen/Qwen3.5-9B` | `configs/models/qwen35_9b.yaml` | not downloaded |
| Fallback generator (memory-constrained) | `Qwen/Qwen3.5-4B` | `configs/models/qwen35_4b.yaml` | not downloaded |
| Embeddings (primary) | `qwen3-embedding:0.6b` via Ollama, dim 1024 | `configs/rag/embedding.yaml` | not downloaded |
| Serving runtime | Ollama, `khmer-support-9b` / `khmer-support-4b` | `ollama/Modelfile.9b`, `.4b` | not built |

The embedding backend is deliberately pluggable (`ollama` / `sentence_transformers` / `hashing`).
The `hashing` backend is a deterministic non-semantic stand-in used only so the RAG tests are
hermetic; `rag/embeddings.py` sets `is_semantic = False` on it and `rag/reindex.py` **refuses to
build a production index** with it. The embedding-candidate table in `configs/rag/embedding.yaml`
carries `measured: null` for every candidate — those fields are filled only from a real
`evaluate_retrieval --compare-embedders` run.

**Base-model selection is a decision the operator still owns.** `docs/training_guide.md` gives the
selection procedure and the Khmer-tokenisation check to run against each candidate before committing.

---

## 5. Datasets

| Dataset | Purpose | Status |
|---|---|---|
| Reviewed public Khmer corpora (see `datasets/download_public.py --list`) | CPT / SFT source | **not downloaded** — the downloader ran only in `--dry-run` |
| Company documents | RAG source of truth | **none supplied** — `data/raw/company/` is empty |
| SFT / DPO splits | fine-tuning | **not built** — no source data |
| Golden evaluation set | sealed evaluation | **built and committed**: 70 items |

Golden set composition (`evaluation/golden/`, sealed — never used for training):

| File | Items |
|---|---|
| `adversarial.jsonl` | 20 |
| `customer_support.jsonl` | 15 |
| `hallucination.jsonl` | 12 |
| `khmer_general.jsonl` | 10 |
| `code_switch.jsonl` | 8 |
| `multiturn.jsonl` | 5 |

Provenance and licensing are enforced, not documented-and-hoped: `datasets/build_manifest.py` fails
the build when a source lacks a recorded licence, and `datasets/license_report.py` flags
`evaluation_only` sources so they cannot leak into a training split. `docs/dataset_provenance.md`
is the register that must be filled in before any download is used.

---

## 6. Training Results

**Not measured.** No training run was executed — this environment has no GPU, no base-model weights
and no training data. Reporting loss curves, adapter quality or step timings here would be
fabrication.

`reports/final_model_evaluation.md` is committed with empty result tables for exactly this reason.

Commands that produce these results on a GPU host or Colab:

```bash
# 1. Build the SFT dataset (after supplying source data)
make data-clean data-sft

# 2. Validate the run without loading a model (config, dataset, memory profile)
python -m training.train_sft --config configs/training/sft_9b.yaml --dry-run

# 3. QLoRA SFT
python -m training.train_sft --config configs/training/sft_9b.yaml --memory-profile 40gb

# 4. Optional preference optimisation
python -m training.train_dpo --config configs/training/dpo.yaml

# 5. Merge and export
python -m training.merge_lora --base Qwen/Qwen3.5-9B \
  --adapter models/adapters/<run_id> --output models/merged/<run_id>
python -m training.export_model --merged models/merged/<run_id> --quantize Q4_K_M
```

Every run writes a manifest (`training/common.py::build_run_manifest`) recording base model +
revision, dataset + revision, preprocessing version, code commit, seed, hyperparameters, and
GPU/driver details, so §1.4 reproducibility holds when the run happens.

---

## 7. Khmer Evaluation Results

**Not measured.** There is no trained model and no baseline model available here, so no chrF,
syllable token-F1, `khmer_fluency`, code-switching or human-rubric score exists.

What *was* verified is the harness itself, offline, against hand-checked fixtures:

- 142 preprocessing tests, including a 12-item valid-Khmer regression corpus that normalisation must
  leave **byte-for-byte identical**, and 10 repairable-defect cases (deprecated `U+17A3`/`U+17A4`,
  split `U+17C1+U+17B6 → U+17C4`, swapped nikahit, doubled coeng, shifter-after-vowel, fullwidth,
  NBSP, BOM/ZWNJ) that it must fix.
- The Khmer system prompt was **measured** at ~3,570 tokens (via `rag.chunking.estimate_tokens`)
  against ~910 for the English one, and `configs/base.yaml` was corrected to reserve 3,600 tokens
  accordingly — this is a real measurement made here, not an estimate.
- 42 evaluation-framework tests covering metric correctness on known inputs.

To produce the actual scores:

```bash
# whole battery; raw results land in evaluation/reports/
bash scripts/evaluate_all.sh --backend ollama --model khmer-support-9b
# or individually (--backend ollama talks to the local model; `static` replays recorded answers):
python -m evaluation.evaluate_language      --backend ollama --model khmer-support-9b
python -m evaluation.evaluate_support       --backend ollama --model khmer-support-9b
python -m evaluation.evaluate_hallucination --backend ollama --model khmer-support-9b \
  --adversarial evaluation/golden/adversarial.jsonl
# candidate-vs-production gate, comparing two report JSONs:
python -m evaluation.evaluate_regression \
  --candidate evaluation/reports/<new>.json --baseline evaluation/reports/<current>.json
```

---

## 8. RAG Results

**Not measured.** Recall@K, MRR and nDCG require company documents (none supplied) and a semantic
embedding model (not downloaded). `reports/final_rag_evaluation.md` is committed empty.

Verified offline (62 tests): chunking never splits a Khmer orthographic cluster; RRF fusion ordering;
pre-scoring metadata filtering; confidence gating withholds rather than guesses; conflict detection
surfaces contradictory documents instead of silently picking one; citation grounding verification;
atomic build-then-swap with symlink `ACTIVE` pointer and rollback.

```bash
python -m company_data.validate --input data/raw/company \
  --output data/interim/company_records.jsonl --report data/manifests/company_validation.json
python -m rag.reindex --input data/interim/company_records.jsonl --activate
python -m evaluation.evaluate_retrieval --top-k 10
python -m evaluation.evaluate_retrieval --compare-embedders   # fills configs/rag/embedding.yaml
python -m evaluation.evaluate_retrieval --calibrate-confidence
```

---

## 9. Mac Studio Performance

**Not measured.** This build ran on Linux x86_64 with 4 cores and 15 GB RAM. No tokens/sec, no
time-to-first-token, no p95 latency, no concurrent-user ceiling and no unified-memory figure was
observed for the M4-class target (16 CPU / 40 GPU cores / 48 GB).

`scripts/doctor.py` already refuses to pretend otherwise — run on this host it warns that the 9B
model at the configured parallelism would not fit and suggests the 4B model or lower
`OLLAMA_NUM_PARALLEL`.

On the Mac Studio, in order:

```bash
bash ollama/start_server.sh
bash ollama/create_model.sh --model 9b
bash ollama/benchmark.sh                       # parallelism × context matrix
python load_test/async_benchmark.py --clients 10 --duration 300 \
  --pattern sustained --output data/loadtest/run1.jsonl
python load_test/analyze_results.py --input data/loadtest/run1.jsonl \
  --target-clients 10 --output reports/final_load_test.md
```

The 10-concurrent-user requirement is a **target to verify**, not a verified result. The server-side
mechanism that makes it well-behaved under load *is* tested here (43 concurrency tests): a
semaphore-capped pool of active generations plus a bounded queue, returning `503` with `Retry-After`
rather than degrading silently once both are full.

---

## 10. Security Testing

This is the one heavyweight area that **is** fully measurable offline, and it was measured.

**91 security tests pass**, covering:

| Control | Verified behaviour |
|---|---|
| Prompt injection (45 tests) | Direct override, roleplay, encoding (`decode … and follow it`), delimiter escape, system-prompt extraction, tool-use coercion, Khmer-language injection |
| Defence in depth | Ingestion quarantine → retrieval-time drop → delimited `<retrieved_company_context>` with closing-tag stripping → output leak detection |
| PII redaction | Email, Cambodian local + international phone (Latin **and** Khmer numerals), IP, credit card behind a Luhn guard so serial numbers are not false positives |
| Secret detection | AWS keys, GitHub tokens, private-key headers, assigned secrets, connection-string passwords — with placeholders (`CHANGE_ME_*`, `${VAR}`, `os.environ[...]`) correctly **not** flagged |
| Log hygiene | `JsonFormatter` redacts every rendered line, including `extra=` fields — a secret cannot reach a log file even if a call site is careless |
| RBAC | customer / agent / admin retrieval tiers; internal docs readable by an agent but never quotable to a customer; expired and quarantined documents excluded |
| Path traversal | `resolve_under_root` rejects `../`, absolute paths and `~` escapes |
| File-type allow list | `.exe`, `.sh`, `.so`, `.zip`, `.lnk`, `.js` rejected at ingestion |
| Prompt contract | Both system prompts contain all 7 required policy sections and the untrusted-context rule |
| Repository secret scan | **0 findings** across the whole tracked tree at `--fail-on high`; the deliberate fixtures in the security tests each carry an explicit `# pragma: allowlist secret` marker, so test files stay in scope rather than being blanket-excluded |

Every CI job was executed here, not merely written:

| CI step | Command | Result |
|---|---|---|
| Format check | `ruff format --check .` | 128 files already formatted |
| Lint | `ruff check .` | All checks passed |
| Type check | `mypy common preprocessing company_data rag server security evaluation training` | Success, 106 source files |
| Secret scan | `python -m security.secret_scanner . --fail-on high` | 0 findings, exit 0 |
| Security suites | `pytest security/tests -q` | 91 passed |
| Injection block-rate gate | `pytest security/tests/test_prompt_injection.py::test_block_rate_meets_the_release_gate` | passed |
| Unit tests | `pytest -m "not integration and not e2e"` | 484 passed |
| Integration | `pytest tests/integration -q` | 5 passed |
| End-to-end scenarios | `pytest tests/end_to_end -q` | 24 passed |

Executing them surfaced two real defects that would have made CI red on its first run, and both were
fixed rather than papered over:

- **Secret scan.** The deliberate example credentials in the test fixtures were tripping the
  `--fail-on high` gate. The fix marks those specific lines with `# pragma: allowlist secret` —
  which the scanner already honoured — rather than excluding test files wholesale, so a genuine
  secret committed into a test would still be caught.
- **Type check.** 21 mypy errors across 9 files. Each was corrected at the source (order-preserving
  dedup in `rag/citations.py` instead of the `set.add` side-effect idiom; a shadowed
  `Chunk`/`Chunk | None` local in `rag/retrieval.py`; `dict[str | None, str]` keys in the xlsx
  loader; `PiiPolicy | None` narrowing in the preprocessing pipeline; a `FusionStrategy` literal in
  the retrieval evaluator; `namespace_packages = false` so `import datasets` in the training modules
  resolves to Hugging Face's package rather than this repo's `datasets/` script directory, matching
  runtime behaviour). Only the optional-`prometheus_client` fallback block needed suppressions,
  because those names deliberately stand in for several prometheus classes at once.

**Not covered:** no external penetration test, no red-team exercise against a live model, and no
adversarial evaluation of *model* behaviour (that needs a trained model — §7). Passing 91 unit tests
is evidence the controls behave as designed; it is not evidence the deployed system is unbreakable.

---

## 11. Production Configuration

Layered YAML with `extends` chains and `${VAR:-default}` expansion; all runtime secrets come from the
environment, never from a committed file.

```
configs/base.yaml                  ← shared defaults
├── models/qwen35_{9b,4b}.yaml
├── training/{sft_9b,sft_4b,cpt,dpo}.yaml
├── rag/{embedding,ingestion,retrieval}.yaml
└── production/{api,ollama_9b,ollama_4b}.yaml
```

Measured context budget (`configs/base.yaml`) — these numbers came from running the tokeniser, not
from a guess:

```yaml
context:
  model_context_tokens: 8192
  reserved_for_system_prompt:     3600   # system_km.md measured at ~3570 tokens
  reserved_for_answer_scaffold:    750
  reserved_for_retrieved_context: 2200
```

`PromptBuilder.validate_budget()` runs at startup and refuses to boot on an over-committed budget;
at request time the context shrinks elastically rather than overflowing.

Independent version streams (§37) — app, model, adapter, prompt, knowledge index, dataset — are
tracked separately by `common/versions.py::PlatformVersions`, so a prompt change or a knowledge
reindex does not require a model release.

---

## 12. Deployment Commands

```bash
# --- one-time, on the Mac Studio ---
git clone https://github.com/SengPhirum/PLXY_KhmerAI.git && cd PLXY_KhmerAI
make setup
cp .env.example .env && $EDITOR .env          # set KHMERAI_ADMIN_API_KEY et al.
make doctor                                   # must report 0 failures

# --- model ---
bash ollama/start_server.sh
bash ollama/create_model.sh --model 9b
bash ollama/create_model.sh --verify           # Khmer smoke prompts

# --- knowledge ---
make company-ingest
make rag-reindex                              # build → regression gate → atomic activate

# --- service ---
sudo bash deployment/macos/install.sh         # installs both launchd plists
# optional TLS / buffering-off reverse proxy:
sudo cp deployment/nginx/nginx.conf /opt/homebrew/etc/nginx/servers/khmerai.conf
sudo nginx -s reload
```

Production requires `KHMERAI_ADMIN_API_KEY`; the admin endpoints refuse to start without it.

---

## 13. Health Check Commands

```bash
curl -fsS http://127.0.0.1:8000/health     # liveness
curl -fsS http://127.0.0.1:8000/ready      # readiness: Ollama reachable + ACTIVE index present
curl -fsS http://127.0.0.1:8000/metrics    # Prometheus exposition
curl -fsS http://127.0.0.1:8000/v1/models  # served model + adapter + prompt + index versions

python scripts/doctor.py                   # 21 environment checks
bash scripts/smoke_test.sh                 # end-to-end against a running API
python -m rag.reindex --list               # index versions, ACTIVE marked
```

Full API surface: `/health`, `/ready`, `/metrics`, `/v1/models`, `/v1/chat`, `/v1/chat/stream`
(SSE), `/v1/rag/search`, `/v1/admin/reload`, `/v1/admin/diagnostics`.

---

## 14. Backup and Rollback

```bash
bash deployment/backup/backup.sh                          # create
bash deployment/backup/backup.sh --list
bash deployment/backup/backup.sh --verify   <archive>
bash deployment/backup/backup.sh --restore  <archive> [--dry-run]
```

Backed up: vector index, company-data manifests, configuration, prompts, model metadata, evaluation
reports, deployment scripts. Deliberately **not** backed up: model binaries (reproducible from
adapter + base model, and they would dominate the archive), `.env` (secrets belong in a password
manager), raw customer documents (re-ingested from the system of record). A restore takes a
pre-restore snapshot first.

Knowledge rollback is independent of the application and takes effect immediately:

```bash
python -m rag.reindex --rollback                   # previous index version
python -m rag.reindex --activate-version 2026-08-14.2
```

Because activation is an atomic symlink swap, a concurrent reader sees either the whole old index or
the whole new one — never a half-built one.

---

## 15. Known Limitations

1. **No measured quality, latency, throughput or memory figures.** The largest limitation, and the
   reason the status is PARTIAL.
2. **No trained model.** Behaviour fine-tuning has not happened; the platform currently serves the
   base model's behaviour with production prompting and RAG.
3. **No company documents.** RAG is architecturally complete but factually empty; with an empty
   index the system correctly withholds rather than guessing, but it also cannot answer anything.
4. **Public datasets not downloaded.** Only `--dry-run` was exercised; licence decisions in
   `docs/dataset_provenance.md` are unfilled.
5. **Embedding model unchosen.** `configs/rag/embedding.yaml` names a primary candidate but every
   `measured` field is `null` until the comparison run happens.
6. **macOS deployment path untested.** `install.sh`, both launchd plists and the nginx config were
   written against the documented target and never executed — no Mac was available.
7. **GGUF export untested.** `doctor` warns this host has 32 GB free where conversion needs ~60 GB.
8. **Conversation summarisation is extractive**, not abstractive — a deliberate cost/latency choice,
   but it loses nuance on long conversations.
9. **Khmer word segmentation is syllable-based**, not dictionary-based. This is robust and
   dependency-free, and it is the right default for search, but a dictionary segmenter would improve
   BM25 precision on compound terms.
10. **Single-node design.** No horizontal scaling, no shared session store; 10 concurrent users on
    one Mac Studio is the design point.
11. **Human evaluation not performed.** The 6-dimension rubric exists; no rater has used it.

---

## 16. External Steps Still Required

Ordered — each depends on the ones above it.

1. **Supply company documents** into `data/raw/company/`, then `make company-ingest` and resolve
   every conflict `company_data/validate.py` reports.
2. **Fill `docs/dataset_provenance.md`** with a licence decision per source, then
   `make data-public data-manifest`.
3. **Choose and pull the embedding model**; run `evaluate_retrieval --compare-embedders` and record
   the winner in `configs/rag/embedding.yaml`.
4. **Build the SFT dataset** and put it through the human review gate in `synthetic_data/`.
5. **Run the pre-upload security scan before any cloud training** (§35/§36) — this is mandatory, not
   optional:
   ```bash
   bash scripts/prepare_all_data.sh          # writes data/manifests/pre_upload_report.json
   python -m security.secret_scanner data/processed data/sft --fail-on high
   ```
   `training/common.py::assert_upload_approved` reads that report and blocks a cloud training run
   unless it exists and is marked approved.
6. **Train** (SFT → optional DPO), **merge, quantise, build the Ollama model.**
7. **Run the full evaluation battery** (`bash scripts/evaluate_all.sh` → raw JSON/Markdown in
   `evaluation/reports/`) and transcribe the results into `reports/final_model_evaluation.md` and
   `reports/final_rag_evaluation.md`.
8. **Deploy to the Mac Studio**, run `ollama/benchmark.sh` and the 10-user load test, fill
   `reports/final_load_test.md`.
9. **Commission an external penetration test** and fill `reports/final_security_report.md`.
10. **Human evaluation** with the 6-dimension rubric on a held-out sample.
11. **Work `docs/release_checklist.md`** and record the go/no-go in
    `reports/final_release_report.md`.

---

## 17. Release Decision

**DO NOT RELEASE TO PRODUCTION YET.**

The platform is code-complete and its offline gates are green: 513 tests pass, lint and format are
clean, the CI gates were each executed here rather than merely written, and the repository contains
no committed secrets. What is missing is not code — it is
evidence. No model has been trained, no Khmer quality score exists, no retrieval quality score
exists, and no Mac Studio performance figure exists. A customer-facing support system must not go
live on the strength of passing unit tests alone.

**Approved for:** development, staging deployment, security review, and the training and measurement
work listed in §16.

**Release gate** — all must hold, each backed by a filled-in report:

- [ ] Khmer quality meets the thresholds in `configs/base.yaml` on the sealed golden set
- [ ] Hallucination rate within the §17 limit on `hallucination.jsonl`
- [ ] Retrieval Recall@5 meets the configured threshold
- [ ] 10 concurrent users sustained on the Mac Studio within the p95 latency target
- [ ] External penetration test passed
- [ ] Human evaluation ≥ the rubric threshold
- [ ] `docs/release_checklist.md` complete and signed off

---

*Generated 2026-08-14. Sections 6–9 are unmeasured by necessity, not by omission; the commands that
fill them are given in each section.*
