# Final release report

> **Status: NOT COMPLETE.** Assembled from the four reports in this directory
> once they contain real measurements.

Release: `________`  Date: `________`  Engineer: `________`

## 1. Overall status

`PASS / PARTIAL / BLOCKED`: `________`

## 2. Versions

| Artefact | Version |
|---|---|
| Application | |
| Model | |
| Adapter | |
| Prompt | |
| Knowledge index | |
| Dataset | |
| Code commit | |

## 3. Gate summary

| Area | Report | Status |
|---|---|---|
| Model quality | `final_model_evaluation.md` | |
| RAG | `final_rag_evaluation.md` | |
| Security | `final_security_report.md` | |
| Performance | `final_load_test.md` | |
| Licensing | `docs/dataset_provenance.md` | |
| Checklist | `docs/release_checklist.md` | |

## 4. Regression versus current production

| Metric | Production | Candidate | Delta | Verdict |
|---|---:|---:|---:|---|

## 5. Deployment commands

```bash
bash deployment/macos/install.sh
bash ollama/create_model.sh
python -m rag.reindex --input data/interim/company_records.jsonl --activate
bash scripts/smoke_test.sh
```

## 6. Health checks

```bash
curl -s localhost:8000/health | python3 -m json.tool
curl -s localhost:8000/ready  | python3 -m json.tool
bash scripts/smoke_test.sh
```

## 7. Rollback

| Artefact | Command | Tested |
|---|---|:---:|
| Application | `git checkout <prev> && bash deployment/macos/install.sh` | |
| Model | `ollama cp khmer-support-9b-previous khmer-support-9b` | |
| Prompt | `git checkout <prev> -- prompts/ && POST /v1/admin/reload` | |
| Index | `python -m rag.reindex --rollback && POST /v1/admin/reload` | |

## 8. Known limitations

## 9. External steps still required

## 10. Release decision

- [ ] APPROVED  - [ ] CONDITIONAL  - [ ] BLOCKED

Signatures: Engineering `____` Khmer reviewer `____` Security `____` Legal `____` Business `____`
