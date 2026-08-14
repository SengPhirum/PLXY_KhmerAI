# Release checklist

A release is blocked until every **required** box is ticked with evidence. Copy
this file into the release ticket and fill in the evidence column — a tick with
no artefact is not a tick.

Release: `________`  Date: `________`  Engineer: `________`  Approver: `________`

## Versions being released

| Artefact | Version | Changed? |
|---|---|---|
| Application | | |
| Model | | |
| Adapter | | |
| Prompt | | |
| Knowledge index | | |
| Dataset | | |

`curl -s localhost:8000/health \| python3 -c "import sys,json; print(json.load(sys.stdin)['versions'])"`

## 1. Repository (required)

- [ ] All required directories exist — `make doctor`
- [ ] Dependencies pinned — `requirements/*.txt` uses `==`
- [ ] `.env.example` complete; no real secret in it
- [ ] Bootstrap works from a clean checkout — `bash scripts/bootstrap.sh`
- [ ] README current
- [ ] No secrets committed — `python -m security.secret_scanner . --fail-on high`

## 2. Dataset (required if the dataset changed)

- [ ] Provenance exists for every source — `python datasets/build_manifest.py --root data/raw --check`
- [ ] Licences tracked and reviewed — `docs/dataset_provenance.md`
- [ ] No source is `review_required` in production training
- [ ] Evaluation data isolated — no `evaluation_only` file in a training path
- [ ] Khmer normalisation tests pass — `pytest preprocessing/tests -q`
- [ ] Exact and near dedup verified
- [ ] Cross-split leakage clean — `sft_dataset_report.json -> cross_split_leakage.clean == true`
- [ ] Intent coverage complete — `intent_coverage.missing == []`
- [ ] Pre-upload PII/secret scan approved (if cloud training was used)

## 3. Training (required if the model changed)

- [ ] 4B and 9B configurations exist and load
- [ ] QLoRA run completed; `run_manifest.json` present
- [ ] Checkpoint resume verified
- [ ] Merge succeeded with base-model verification
- [ ] Export produced every quantization level
- [ ] Experiment metadata stored (commit, seed, dataset hashes, hyper-parameters)

## 4. Evaluation (required, always)

Run `bash scripts/evaluate_all.sh` and attach the reports.

| Gate | Threshold | Measured | Pass |
|---|---:|---:|:---:|
| Khmer fluency | ≥ 0.80 | | |
| Support accuracy | ≥ 0.85 | | |
| Grounding precision | ≥ 0.90 | | |
| Hallucination rate | ≤ 0.03 | | |
| Unsupported claim rate | ≤ 0.02 | | |
| Retrieval Recall@5 | ≥ 0.85 | | |
| Injection block rate | ≥ 0.95 | | |

- [ ] All automated gates pass
- [ ] Native Khmer review completed — ≥ 100 answers, mean ≥ 4.0/5
- [ ] No reviewer scored Faithfulness or Correctness below 3
- [ ] Regression versus current production run — verdict `accept` or a signed-off `review`

## 5. RAG (required if the index changed)

- [ ] Every loader verified — PDF, DOCX, HTML, TXT, CSV, XLSX
- [ ] Version filtering works; expired documents excluded by default
- [ ] Conflicting active versions resolved (none outstanding)
- [ ] Retrieval quality measured
- [ ] No-result behaviour safe — confidence gate withholds rather than guesses
- [ ] Reindex → regression gate → activate → rollback all exercised

## 6. Ollama (required if the model changed)

- [ ] Model builds from the Modelfile
- [ ] Khmer generation verified — `bash ollama/create_model.sh --verify`
- [ ] Quantization levels benchmarked; the choice is justified by measurement
- [ ] Context length matches `KHMERAI_NUM_CTX`
- [ ] Parallelism benchmarked — `bash ollama/benchmark.sh`

## 7. API (required)

- [ ] Health, ready, metrics respond
- [ ] Chat and streaming work
- [ ] Sources returned for grounded answers
- [ ] Error handling verified — 503/504/429/422
- [ ] Rate limiting verified
- [ ] Admin routes protected
- [ ] `bash scripts/smoke_test.sh` passes

## 8. Security (required)

- [ ] Prompt-injection suite passes — `pytest security/tests -q`
- [ ] Retrieved instructions cannot override system policy
- [ ] No secrets in logs — verified by test
- [ ] Request-size limits enforced
- [ ] Document ingestion validates file types and rejects traversal
- [ ] Admin key rotated if this release changes access
- [ ] Ollama (11434) not reachable off-host

## 9. Performance (required)

| Clients | p50 | p95 | p99 | TTFT p95 | tok/s | Peak mem | OOM |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | | | | | | | |
| 5 | | | | | | | |
| 10 | | | | | | | |
| 20 | | | | | | | |

- [ ] All four levels measured
- [ ] p95 within SLO at the target client count
- [ ] No uncontrolled OOM
- [ ] Recommended configuration recorded in `configs/production/ollama_*.yaml`

## 10. Production readiness (required)

- [ ] Install script verified on the target Mac
- [ ] launchd services load and auto-restart
- [ ] Monitoring scraping; alert rules loaded
- [ ] Backup runs and **verifies** — `backup.sh --verify`
- [ ] Restore tested from a real archive
- [ ] Rollback commands documented and tested for each artefact
- [ ] Final smoke test passes

## 11. Rollback plan (required)

| Artefact | Rollback command | Tested |
|---|---|:---:|
| Application | `git checkout <prev> && bash deployment/macos/install.sh` | |
| Model | `ollama cp khmer-support-9b-previous khmer-support-9b` | |
| Prompt | `git checkout <prev> -- prompts/ && POST /v1/admin/reload` | |
| Index | `python -m rag.reindex --rollback && POST /v1/admin/reload` | |

- [ ] Previous model tag retained in Ollama
- [ ] Previous index version retained on disk
- [ ] Someone who is not the release engineer can execute the rollback

## 12. Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Engineering | | | |
| Khmer language reviewer | | | |
| Security | | | |
| Legal (dataset/model licensing) | | | |
| Business owner | | | |

## Decision

- [ ] **APPROVED** — every required gate passed with evidence
- [ ] **CONDITIONAL** — approved with these exceptions and mitigations: `______`
- [ ] **BLOCKED** — reason: `______`
