# Reports

Templates for the final production-validation reports (§Phase 24). They are
**empty of results by design**: filling them in with anything other than a real
measurement would violate §1.2 of the specification.

| File | Produced by |
|---|---|
| `final_model_evaluation.md` | `bash scripts/evaluate_all.sh` + native review |
| `final_rag_evaluation.md` | `python -m evaluation.evaluate_retrieval` |
| `final_security_report.md` | `pytest security/tests` + the behavioural injection suite |
| `final_load_test.md` | `python load_test/async_benchmark.py` + `bash ollama/benchmark.sh` |
| `final_release_report.md` | assembled from the four above |

Generated evaluation output lands in `evaluation/reports/` and is gitignored;
these five files are the curated, human-signed summaries that go into the release
record.

`implementation_report.md` is different: it is the §46 build report and it *is*
filled in. It states what was implemented, which gates were actually executed in
the build environment, and — explicitly — which measurements were **not** taken
because they need a GPU host or the Mac Studio. Read it first.
