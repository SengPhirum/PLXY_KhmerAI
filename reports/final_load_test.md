# Final load test and Ollama tuning

> **Status: NOT YET MEASURED.** Requires the target Mac Studio with a built
> model. Nothing here may be estimated (§1.2).

Machine: `________`  Model: `________`  Quantization: `________`  Date: `________`

```bash
bash ollama/benchmark.sh --model khmer-support-9b
python load_test/async_benchmark.py --clients 1  --duration 60
python load_test/async_benchmark.py --clients 5  --duration 60
python load_test/async_benchmark.py --clients 10 --duration 120
python load_test/async_benchmark.py --clients 20 --duration 120 --pattern ramp
python load_test/async_benchmark.py --clients 20 --pattern burst --duration 60
python load_test/analyze_results.py --input reports/ollama_benchmark/summary-*.jsonl
```

## Tuning matrix

| Model | NUM_PARALLEL | Context | Clients | TTFT p95 | p50 | p95 | p99 | tok/s | Success | Peak mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | | | | |

## Traffic-mixture load test

Mixture: 50% short, 25% RAG-heavy, 15% follow-up, 5% long, 5% adversarial.

| Clients | Pattern | Requests | Success | p50 | p95 | p99 | TTFT p95 | tok/s | 503 | 429 | Incomplete streams |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | constant | | | | | | | | | | |
| 5 | constant | | | | | | | | | | |
| 10 | constant | | | | | | | | | | |
| 20 | ramp | | | | | | | | | | |
| 20 | burst | | | | | | | | | | |

## SLO compliance

| SLO | Target | Measured at 10 clients | Pass |
|---|---:|---:|:---:|
| TTFT p95 | ≤ 2.5 s | | |
| End-to-end p95 | ≤ 12 s | | |
| End-to-end p99 | ≤ 20 s | | |
| Retrieval p95 | ≤ 0.8 s | | |
| Success rate | ≥ 99.5% | | |
| Uncontrolled OOM | 0 | | |

## Memory

| | Value |
|---|---:|
| Total unified memory | 48 GB |
| Model resident | |
| KV cache at chosen settings | |
| API process | |
| Peak observed | |
| Headroom | |

## Recommended production configuration

```
OLLAMA_NUM_PARALLEL=____
OLLAMA_CONTEXT_LENGTH=____
OLLAMA_KV_CACHE_TYPE=____
OLLAMA_KEEP_ALIVE=-1
KHMERAI_MAX_ACTIVE_GENERATIONS=____   # must be <= OLLAMA_NUM_PARALLEL
KHMERAI_MAX_QUEUE_DEPTH=____
```

**Decision (§Phase 13)**: `____` connected clients served by `____` active
generations plus a queue. Justification from the measured customer experience:
`________`

## Failure behaviour

- [ ] Graceful 503 with `Retry-After` when the queue is full
- [ ] No OOM at 20 clients
- [ ] Streams complete or fail cleanly; no partial hangs
- [ ] Recovers after an Ollama restart
