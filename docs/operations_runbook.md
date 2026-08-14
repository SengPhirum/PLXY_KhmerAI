# Operations runbook

Diagnostic first, action second. Every procedure starts with a command that
tells you what is actually wrong.

## Quick reference

```bash
curl -s localhost:8000/health  | python3 -m json.tool     # is it alive?
curl -s localhost:8000/ready   | python3 -m json.tool     # can it serve?
curl -s -H "X-Admin-Key: $KHMERAI_ADMIN_API_KEY" \
     localhost:8000/v1/admin/diagnostics | python3 -m json.tool
bash scripts/smoke_test.sh                                # end-to-end
tail -f /usr/local/var/log/khmerai/api.err.log            # API log
tail -f /usr/local/var/log/khmerai/ollama.err.log         # model log
ollama ps                                                 # loaded models + memory
```

Service control:

```bash
sudo launchctl kickstart -k system/com.company.khmerai.api      # restart API
sudo launchctl kickstart -k system/com.company.khmerai.ollama   # restart Ollama
sudo launchctl print system/com.company.khmerai.api             # status
```

---

## Ollama down

**Symptoms**: `/ready` reports `ollama: unhealthy`; chat returns 503
`model_unavailable`; `KhmerAIOllamaErrors` fires.

**Diagnose**

```bash
curl -sf localhost:11434/api/tags || echo "daemon not answering"
sudo launchctl print system/com.company.khmerai.ollama | grep -E "state|last exit"
tail -50 /usr/local/var/log/khmerai/ollama.err.log
ps aux | grep -c "[o]llama"
```

**Act**

```bash
sudo launchctl kickstart -k system/com.company.khmerai.ollama
sleep 10 && curl -sf localhost:11434/api/tags && ollama ps
```

If it will not stay up, the usual cause is memory: a `KEEP_ALIVE=-1` model plus
a second loaded model exceeds unified memory. Check `OLLAMA_MAX_LOADED_MODELS=1`
in the plist and confirm only one model is resident.

**Verify**: `bash scripts/smoke_test.sh`

---

## API down

**Symptoms**: connection refused on :8000; `KhmerAIApiDown` fires.

**Diagnose**

```bash
sudo launchctl print system/com.company.khmerai.api | grep -E "state|last exit"
tail -100 /usr/local/var/log/khmerai/api.err.log
lsof -i :8000
```

A startup crash is almost always configuration. The most common causes, all of
which fail loudly at startup by design:

| Log line | Cause | Fix |
|---|---|---|
| `KHMERAI_ADMIN_API_KEY must be set` | production without an admin key | set it in `.env` |
| `the hashing embedding backend is for tests only` | test backend in production | `KHMERAI_EMBEDDING_BACKEND=ollama` |
| `environment variable ... is not set` | a config references a missing var | add it to `.env` |
| `chat.prompt_budget` warning | a prompt edit no longer fits the window | shorten the prompt or raise `KHMERAI_NUM_CTX` |

**Act**

```bash
cd /usr/local/opt/khmerai
./.venv/bin/python -c "from server.config import Settings; print(Settings().redacted())"  # validate config
sudo launchctl kickstart -k system/com.company.khmerai.api
```

---

## Memory pressure

**Symptoms**: latency collapses rather than degrading; swap in Activity Monitor;
`KhmerAIMemoryPressure` fires.

**Diagnose**

```bash
vm_stat | head -8
ollama ps                                   # resident models and their size
curl -s localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['active_generations'], d['queued_requests'])"
```

**Understand before acting**: KV cache is the variable cost, and it scales with
`OLLAMA_NUM_PARALLEL x OLLAMA_CONTEXT_LENGTH`. Going from 4 to 8 parallel at
8192 context roughly doubles it.

**Act, in order of least disruption**

1. Reduce parallelism and let the API queue:
   ```bash
   sudo launchctl unload /Library/LaunchDaemons/com.company.khmerai.ollama.plist
   # edit OLLAMA_NUM_PARALLEL 4 -> 2 in the plist
   sudo launchctl load -w /Library/LaunchDaemons/com.company.khmerai.ollama.plist
   # keep KHMERAI_MAX_ACTIVE_GENERATIONS <= OLLAMA_NUM_PARALLEL
   ```
2. Reduce `OLLAMA_CONTEXT_LENGTH` to 4096 (also lower `KHMERAI_NUM_CTX`, and
   re-check the prompt budget warning at startup).
3. Confirm `OLLAMA_KV_CACHE_TYPE=q8_0` — it halves KV memory.
4. Switch to the 4B model:
   ```bash
   # .env: KHMERAI_OLLAMA_MODEL=khmer-support-4b
   sudo launchctl kickstart -k system/com.company.khmerai.api
   ```

---

## Slow responses

Work through the layers in this order; each command isolates one.

```bash
# 1. Retrieval
time curl -s -X POST localhost:8000/v1/rag/search \
  -H 'Content-Type: application/json' -d '{"query":"ការធានា","top_k":6}' >/dev/null

# 2. Queueing
curl -s localhost:8000/health | python3 -m json.tool | grep -E "active|queued"

# 3. Model, bypassing the API entirely
time curl -s localhost:11434/api/chat -d \
  '{"model":"khmer-support-9b","messages":[{"role":"user","content":"សួស្តី"}],"stream":false}' >/dev/null

# 4. Host
vm_stat | head -5; uptime
```

| Slow layer | Likely cause | Action |
|---|---|---|
| Retrieval > 0.8 s | index grew past a flat scan | switch to the Qdrant backend, or reduce `candidate_k` |
| Queue depth > 0 sustained | more traffic than capacity | see *Memory pressure*, step 1 |
| Model slow with an empty queue | model unloaded and reloading | confirm `OLLAMA_KEEP_ALIVE=-1` |
| Everything slow | host swapping | see *Memory pressure* |

A first-token time in the tens of seconds after an idle period is almost always
a model reload, not a slow model.

---

## Bad answers

**Do not retrain.** Retraining is the slowest, most expensive and least likely
fix. Diagnose which layer produced the bad answer first.

**Capture the evidence**

```bash
# 1. What did retrieval actually return for this question?
curl -s -X POST localhost:8000/v1/rag/search -H 'Content-Type: application/json' \
  -d '{"query":"<the customer question>","top_k":6}' | python3 -m json.tool

# 2. Which versions produced the answer?
curl -s localhost:8000/health | python3 -c "import sys,json; print(json.load(sys.stdin)['versions'])"

# 3. The answer itself, with its sources
curl -s -X POST localhost:8000/v1/chat -H 'Content-Type: application/json' \
  -d '{"message":"<the customer question>","stream":false}' | python3 -m json.tool
```

**Classify** — the decision tree, in order:

| Observation | Layer at fault | Action |
|---|---|---|
| The correct document is not in the search results | **retrieval** | check filters/status; re-run `evaluation/evaluate_retrieval.py`; consider chunk size |
| The document is returned but says the wrong thing | **source documentation** | fix the document and reindex; this is a business-data problem, not an AI one |
| The document is correct and returned, but the answer ignores it | **prompt** | check `prompt_version`; re-run the grounding evaluation |
| The answer is grounded but reads badly in Khmer | **model behaviour** | native-speaker review; a candidate SFT improvement |
| The behaviour is wrong across many examples of one intent | **dataset** | that intent is under-represented; check `data/manifests/sft_dataset_report.json` |
| The answer contradicts the API contract (missing sources, wrong schema) | **software bug** | reproduce in a test, then fix |

Record every case with: user input, retrieved documents, prompt version, model
version, index version, the answer, and a reviewer label. That record is what
turns anecdote into a dataset improvement.

---

## Knowledge index problems

**Symptom**: the assistant says "I don't have that information" about everything.

```bash
ls -l data/index/ACTIVE                      # where does it point?
python -m rag.reindex --list                 # what versions exist?
curl -s -H "X-Admin-Key: $KHMERAI_ADMIN_API_KEY" \
  localhost:8000/v1/admin/diagnostics | python3 -c "import sys,json; print(json.load(sys.stdin)['rag'])"
```

If `chunks: 0`, the last build produced nothing. The most common cause is that
every document defaulted to `confidentiality: internal` — ingestion reports this
explicitly as `nothing_retrievable`. Check
`data/manifests/company_validation.json`.

**Roll back**

```bash
python -m rag.reindex --rollback
curl -X POST -H "X-Admin-Key: $KHMERAI_ADMIN_API_KEY" localhost:8000/v1/admin/reload
bash scripts/smoke_test.sh
```

---

## Restore

```bash
bash deployment/backup/backup.sh --list
bash deployment/backup/backup.sh --verify <archive>      # always verify first
bash deployment/backup/backup.sh --restore <archive> --dry-run
bash deployment/backup/backup.sh --restore <archive>
curl -X POST -H "X-Admin-Key: $KHMERAI_ADMIN_API_KEY" localhost:8000/v1/admin/reload
bash scripts/smoke_test.sh
```

A restore snapshots the current state first, so a bad restore is itself
reversible.

---

## Rollback

Each artefact rolls back independently.

| Artefact | Command | Effect |
|---|---|---|
| Application | `git checkout <tag> && bash deployment/macos/install.sh` | ~2 min downtime |
| Model | `ollama cp khmer-support-9b-previous khmer-support-9b` then restart the API | no downtime |
| Prompt | `git checkout <tag> -- prompts/` then `POST /v1/admin/reload` | no downtime |
| Index | `python -m rag.reindex --rollback` then `POST /v1/admin/reload` | no downtime |
| Config | edit `.env`, restart the API | ~10 s |

Always verify with `bash scripts/smoke_test.sh` afterwards.

---

## Escalation

| Situation | Who | When |
|---|---|---|
| Customer-facing outage > 15 min | on-call engineer | immediately |
| Wrong price or warranty served | on-call + the document owner | immediately — this is a commercial liability |
| Confidential document exposed | security lead | immediately |
| Sustained injection attempts | security lead | same day |
| Quality degradation without an outage | ML engineer | next working day |
