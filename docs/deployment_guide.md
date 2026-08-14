# Deployment guide — Mac Studio

Target: Mac Studio, Apple M4-class, 16 CPU cores, 40 GPU cores, 48 GB unified
memory, macOS. On-premise, no cloud inference.

## 1. Prerequisites

```bash
brew install python@3.11 ollama
brew install nginx          # only if exposing beyond the trusted LAN
bash deployment/macos/install.sh --check    # verifies hardware, changes nothing
```

## 2. Install

```bash
git clone <repo> /usr/local/opt/khmerai && cd /usr/local/opt/khmerai
bash deployment/macos/install.sh
```

The installer verifies hardware, installs the Python environment, creates
directories, generates an admin key into `.env` (mode 600), installs both launchd
services, starts them and health-checks the result. **Copy the generated admin
key into the password manager when it is printed — it is not shown again.**

For a non-root install: `bash deployment/macos/install.sh --user`.

## 3. Build the model

```bash
# On the training host: merge and export (see docs/training_guide.md)
# Then copy models/gguf/*.gguf to the Mac Studio and:
bash ollama/create_model.sh
bash ollama/create_model.sh --verify        # Khmer smoke prompts
```

## 4. Build the knowledge index

```bash
python -m company_data.validate --input data/raw/company \
    --output data/interim/company_records.jsonl \
    --report data/manifests/company_validation.json
python -m rag.reindex --input data/interim/company_records.jsonl --activate
```

## 5. Verify

```bash
curl -s localhost:8000/ready | python3 -m json.tool
bash scripts/smoke_test.sh
```

The smoke test covers health, readiness, Khmer output, grounding, code switching,
an unknown product, prompt injection, streaming, retrieval, admin protection,
10 concurrent clients and metrics.

## 6. Tune Ollama — this is where the performance is

Everything shipped is a documented **starting point**. Measure on the real
machine:

```bash
bash ollama/benchmark.sh --model khmer-support-9b
```

It sweeps `OLLAMA_NUM_PARALLEL` x context x connected clients, restarting the
daemon each time (those variables are read at startup, not per request), and
writes a recommendation via `load_test/analyze_results.py`.

### What actually limits throughput

KV cache, not compute. It scales with
`OLLAMA_NUM_PARALLEL x OLLAMA_CONTEXT_LENGTH`, and on 48 GB the failure mode is
not gradual: once the machine swaps, latency collapses. So:

| Setting | Effect | Guidance |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | multiplies KV memory | start at 4 for the 9B, 8 for the 4B |
| `OLLAMA_CONTEXT_LENGTH` | multiplies KV memory | 8192; drop to 4096 under pressure |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | halves KV memory | usually worth more than one extra parallel slot |
| `OLLAMA_KEEP_ALIVE=-1` | keeps the model resident | without it, the first request after idle pays a multi-second reload that dominates p99 |
| `OLLAMA_MAX_LOADED_MODELS=1` | prevents two models resident | 48 GB cannot hold 4B + 9B + embeddings comfortably |

**Invariant**: `KHMERAI_MAX_ACTIVE_GENERATIONS ≤ OLLAMA_NUM_PARALLEL`. Exceeding
it produces 503s under load that look like an Ollama failure but are an API
misconfiguration.

### 10 connected clients does not mean 10 concurrent generations

§Phase 13 is explicit: it is acceptable — often better — to serve 10+ *connected*
clients with 4–6 *active* generations plus a queue, when the measured customer
experience is better. `analyze_results.py` applies exactly that rule and will
recommend queueing over parallelism when the SLOs say so.

## 7. Load test

```bash
python load_test/async_benchmark.py --clients 1  --duration 60
python load_test/async_benchmark.py --clients 5  --duration 60
python load_test/async_benchmark.py --clients 10 --duration 120
python load_test/async_benchmark.py --clients 20 --duration 120 --pattern ramp
python load_test/async_benchmark.py --clients 20 --pattern burst --duration 60
```

Record p50/p95/p99, TTFT, tokens/sec and peak memory per level in
`reports/final_load_test.md`. Peak unified memory is read from Activity Monitor
or `ollama ps` — the API cannot observe it.

## 8. Network exposure

Default is loopback-only, which needs no TLS and no reverse proxy. Beyond the
trusted LAN:

```bash
cp deployment/nginx/nginx.conf /opt/homebrew/etc/nginx/servers/khmerai.conf
# generate or install certificates, then:
nginx -t && brew services restart nginx
```

Two things in that config are load-bearing:

- `proxy_buffering off` on `/v1/chat/stream`. With buffering on, nginx holds the
  whole SSE response and the customer sees nothing until generation finishes —
  the entire benefit of streaming is lost, and it looks like the model is slow.
- `/v1/admin/` is restricted to internal networks, and `/metrics` to loopback.

### Firewall

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setblockall off
```

| Port | Bind | Exposure |
|---|---|---|
| 8000 | 127.0.0.1 | never directly exposed |
| 11434 | 127.0.0.1 | **never exposed** — Ollama has no authentication |
| 9090 | 127.0.0.1 | Prometheus, loopback only |
| 443 | 0.0.0.0 | nginx, only when needed |

Ollama on 11434 with no auth is the highest-risk misconfiguration in this stack:
anyone who reaches it can run arbitrary prompts against the model and read the
loaded model list. Keep it on loopback.

## 9. Monitoring

```bash
brew install prometheus
prometheus --config.file=monitoring/prometheus.yml
```

Alert rules are in `monitoring/alerts/khmerai_alerts.yml`, with thresholds traced
back to `configs/base.yaml -> slo` and each alert annotated with its runbook
section.

## 10. Backup

```bash
bash deployment/backup/backup.sh
bash deployment/backup/backup.sh --verify <archive>     # untested backup ≠ backup
```

Schedule daily at 02:00 via launchd. Set `KHMERAI_BACKUP_GPG_RECIPIENT` — the
index contains company documents, and an unencrypted backup of it is a data-loss
event waiting to happen.

## 11. Operating

```bash
sudo launchctl kickstart -k system/com.company.khmerai.api
sudo launchctl kickstart -k system/com.company.khmerai.ollama
tail -f /usr/local/var/log/khmerai/api.err.log
```

Both services restart automatically on crash with a 10-second throttle.

Full diagnostic procedures: `docs/operations_runbook.md`.

## 12. Not executed here

No deployment was performed from this repository's build environment — it is a
Linux container with no Ollama, no model weights and no Mac. Every command above
is the exact command to run. `bash deployment/macos/install.sh --check` verifies
prerequisites without changing anything, and `bash scripts/smoke_test.sh` is the
acceptance test for a real deployment.
