# Final RAG evaluation

> **Status: NOT YET MEASURED.** Requires a built index and an embedding model.

Index version: `________`  Embedding model: `________`  Date: `________`

```bash
python -m evaluation.evaluate_retrieval --index-dir data/index/ACTIVE
python -m evaluation.evaluate_retrieval --compare-embedders
python -m evaluation.evaluate_retrieval --sweep-chunking 400,600,800 --overlap 0,50,100
python -m evaluation.evaluate_retrieval --calibrate-confidence
```

## Corpus

| | Value |
|---|---:|
| Documents ingested | |
| Documents retrievable | |
| Chunks indexed | |
| Quarantined (injection/secrets) | |
| Excluded (expired/draft/internal) | |
| Conflicts outstanding | |

## Retrieval quality

| Metric | Threshold | Measured | Pass |
|---|---:|---:|:---:|
| Recall@1 | | | |
| Recall@3 | | | |
| Recall@5 | ≥ 0.85 | | |
| MRR | | | |
| nDCG@10 | | | |
| Empty-result rate | | | |
| Latency p95 | ≤ 800 ms | | |

## Embedding benchmark

| Candidate | Dim | Recall@5 | MRR | Index build | Verdict |
|---|---:|---:|---:|---:|---|
| qwen3-embedding-0.6b (ollama) | 1024 | | | | |
| Qwen3-Embedding-0.6B (ST) | 1024 | | | | |
| BAAI/bge-m3 | 1024 | | | | |
| intfloat/multilingual-e5-large | 1024 | | | | |

## Chunking sweep

| Chunk size | Overlap | Chunks | Recall@5 | MRR | Latency p95 |
|---:|---:|---:|---:|---:|---:|
| 400 | 0 / 50 / 100 | | | | |
| 600 | 0 / 50 / 100 | | | | |
| 800 | 0 / 50 / 100 | | | | |

**Selected**: chunk_size `____`, overlap `____`. **Justification**: `________`

## Fusion strategy

| Strategy | Recall@5 | MRR | Verdict |
|---|---:|---:|---|
| dense_only | | | |
| lexical_only | | | |
| rrf | | | |
| weighted | | | |

## Confidence calibration

| | Answerable | Unanswerable |
|---|---:|---:|
| Top-1 similarity p50 | | |
| Top-1 similarity p10 | | |

Applied thresholds: min_to_answer `____`, medium `____`, high `____`

## Behaviour checks

- [ ] Expired documents excluded by default
- [ ] Internal documents never reach a customer answer
- [ ] Conflicting active versions surfaced, not silently resolved
- [ ] No-result behaviour produces uncertainty, not a guess
- [ ] Injected chunks dropped at query time
