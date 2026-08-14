# Architecture

## 1. What this system is

A Khmer-first customer-support assistant that answers questions about the
company's products, services, pricing, warranty, policies and troubleshooting.
It runs entirely on premise, on a Mac Studio, through Ollama. Company facts come
from a retrieval index built from approved company documents; the model supplies
language and behaviour, never business facts.

The single most important architectural decision follows from that sentence:

> **Model weights hold *how to talk*. The retrieval index holds *what is true*.**

Prices, warranty periods, stock levels and policies change weekly. Encoding them
in weights would mean retraining for every price change and would make wrong
answers unfixable without a training run. Instead they live in an index that can
be rebuilt in minutes and rolled back in seconds (§Phase 21).

## 2. Request flow

```
Customer
   |
   v
HTTPS / nginx  ──────────────  TLS, per-IP rate limit, SSE buffering OFF
   |
   v
FastAPI (server/main.py)
   |
   +--> RequestContextMiddleware   request ID, timing, structured logging
   +--> RequestSizeLimitMiddleware 64 KB cap, before the body is parsed
   +--> RateLimitMiddleware        token bucket per API key / IP
   +--> SecurityHeadersMiddleware
   |
   +--> require_client / require_admin        (server/dependencies.py)
   |
   +--> InputGuard                            (server/guardrails.py)
   |      size, control chars, injection score, intent, PII detection
   |
   +--> ConversationStore                     (server/models.py)
   |      recent turns verbatim, older turns summarised, TTL, bounded
   |
   +--> RagService -> Retriever               (server/rag_service.py, rag/retrieval.py)
   |      |
   |      +--> metadata filter   status/expiry/confidentiality, BEFORE scoring
   |      +--> dense retrieval   Qwen3-Embedding via Ollama
   |      +--> BM25             Khmer syllable n-grams
   |      +--> RRF fusion
   |      +--> optional rerank
   |      +--> confidence gate  weak context is WITHHELD, not passed on
   |      +--> conflict detect  disagreeing active documents are exposed
   |      +--> injection drop   poisoned chunks are removed
   |
   +--> PromptBuilder                         (server/chat_service.py)
   |      [SYSTEM POLICY] [LANGUAGE] [CUSTOMER-SERVICE] [GROUNDING] [SECURITY]
   |      [CONVERSATION SUMMARY] [RETRIEVED COMPANY CONTEXT] [USER MESSAGE]
   |      [OUTPUT FORMAT]        + hard context budget
   |
   +--> OllamaClient                          (server/ollama_client.py)
   |      semaphore-capped active generations, bounded queue, 503 + Retry-After
   |
   +--> OutputGuard                           (server/guardrails.py)
   |      system-prompt leak, secret leak, identifier corruption, grounding
   |
   +--> SSE stream / JSON response
   |
   +--> Prometheus metrics + redacted structured logs
```

## 3. Measurable requirements

Everything below is declared in `configs/base.yaml` and enforced somewhere in
code or in a release gate. Nothing here is aspirational prose.

| Requirement | Value | Enforced by |
|---|---|---|
| Primary language | Khmer | `prompts/system_km.md`, `evaluation/evaluate_language.py` |
| Supported | Khmer, English, code-switched | `preprocessing/khmer_detection.py` |
| Connected clients | 10 target, 20 graceful | `load_test/`, `server/ollama_client.py` |
| Active generations | 4 (tunable) | `KHMERAI_MAX_ACTIVE_GENERATIONS` |
| TTFT p95 | 2.5 s | `monitoring/alerts/`, `load_test/analyze_results.py` |
| End-to-end p95 | 12 s | same |
| Retrieval p95 | 0.8 s | same |
| Success rate | 99.5% | same |
| Context window | 8192 tokens | `KHMERAI_NUM_CTX` + `OLLAMA_CONTEXT_LENGTH` |
| Max response | 768 tokens | `KHMERAI_MAX_OUTPUT_TOKENS` |
| Grounding precision | ≥ 0.90 | `server/guardrails.py`, `evaluation/evaluate_grounding.py` |
| Hallucination rate | ≤ 0.03 | `evaluation/evaluate_hallucination.py` |
| Injection block rate | ≥ 0.95 | `security/tests/test_prompt_injection.py` |
| Retrieval Recall@5 | ≥ 0.85 | `evaluation/evaluate_retrieval.py` |
| Conversation retention | 1 h, in memory | `server/models.py` |
| Backup | daily, encrypted, 30 days | `deployment/backup/backup.sh` |

## 4. The context budget

The 8192-token window is a hard constraint, and Khmer consumes it far faster
than English. Measured with `rag.chunking.estimate_tokens`:

| Section | Tokens | Note |
|---|---:|---|
| `prompts/system_km.md` | ~3570 | The English version of the same content is ~900. |
| `prompts/rag_answer.md` | ~730 | |
| Retrieved context | ≤ 2200 | `configs/rag/retrieval.yaml` |
| Conversation history | remainder | dropped oldest-first |
| Output reserve | 768 | |
| Template margin | 300 | |

That the Khmer system prompt costs four times its English equivalent is not a
detail — it is the reason `reserved_for_system_prompt` is 3600 rather than the
900 that seemed reasonable before measuring. `PromptBuilder.validate_budget()`
runs at startup and warns if a prompt edit breaks the arithmetic, and
`PromptBuilder.build_messages` enforces it at request time by shrinking the
elastic parts in order: history, then retrieved chunks, then (last resort) the
customer's own message. The system prompt is never the thing that gets dropped.

## 5. Component decisions

### Vector store: local flat index, Qdrant available

At a few thousand to a few hundred thousand chunks, an exact brute-force scan
over normalised vectors is sub-millisecond and has perfect recall. An ANN index
would add a daemon to supervise and would lose recall for no latency gain at
this size. The local store is a directory, which makes the atomic build-then-swap
and the rollback in Phase 21 trivial: activation is one `os.replace` of a
symlink.

`QdrantVectorStore` implements the same interface for when the corpus outgrows a
flat scan. It is preferred over Chroma or raw FAISS because it filters on payload
*inside* the search — post-filtering silently returns fewer than *k* results
whenever the top matches are expired or internal documents.

### Retrieval: hybrid, with the filter first

Dense retrieval alone fails exactly where support needs precision: an exact model
number, an SKU, a rarely-seen technical term. BM25 over Khmer syllable n-grams
matches those literally. RRF fuses the two because a cosine similarity and an
unbounded BM25 score are genuinely incomparable, and any weighted sum silently
changes meaning when the embedding model is swapped.

Metadata filtering runs *before* scoring. This is the difference between "the top
6 results, of which 2 happen to be servable" and "the top 6 servable results".

### Confidence gate: withholding beats guessing

When nothing clears the confidence floor, the retriever returns an **empty**
result rather than the best of a bad set. Stuffing weakly-related context is the
largest single cause of confident hallucination in a support RAG system, and it
is prevented in `rag/retrieval.py` rather than left to the prompt to resist.

### Conversation memory: extractive, not abstractive

Older turns are compressed by *extracting* the product under discussion, the
unresolved question and any commitment made. An abstractive summary would need a
second model call on every turn — doubling latency and cost — and would itself be
a hallucination surface. Extraction cannot introduce a claim that was not said.

### Admission control in the API, not just in Ollama

Connections are cheap; generations are not. The API caps active generations with
a semaphore, queues up to a bound, and returns 503 with `Retry-After` beyond it.
This is what makes "10+ connected clients on 4–6 active generations" a design
rather than an accident (§Phase 13).

## 6. Khmer-specific engineering

Khmer breaks the assumptions built into most NLP tooling, and each break is
handled explicitly:

| Assumption | Reality in Khmer | What we do |
|---|---|---|
| Words are whitespace-separated | No spaces between words | Orthographic syllable clustering (`preprocessing/khmer_script.py`) |
| A character is a token | One syllable ≈ 1.6 model tokens | Measured coefficients in `rag/chunking.py` |
| Truncate at a character offset | Cuts a cluster → mojibake | Chunk boundaries only between clusters |
| NFC fixes ordering | Khmer marks are ccc=0, so NFC reorders nothing | Explicit canonical cluster ordering |
| BLEU/word-F1 measure quality | Whitespace tokenisation yields one "word" per sentence | chrF and syllable-F1 |
| Latin regexes find PII | Phone numbers appear in Khmer numerals | `KHMER_DIGIT_TRANSLATION` before matching |
| A model number is just text | Khmer-tuned models over-transliterate them | Protected spans, verified in the output guard |

## 7. Data flow

```
Public Khmer corpora ──> manifest + licence review ──> preprocessing ──> dedup ──┐
                                                                                 ├──> SFT dataset ──> QLoRA ──> merge ──> GGUF ──> Ollama
Company documents ──> loaders ──> normalise ──> validate ──> canonical records ──┘                                                    │
                                                        │                                                                             │
                                                        └──> chunk ──> embed ──> index ──> regression gate ──> ACTIVE  ────────────────┤
                                                                                                                                      v
                                                                                                                          FastAPI ──> Customer
```

The two paths meet only at serving time. A company-document change goes round
the bottom loop in minutes and never touches the model.

## 8. Versioning

Five artefacts version independently (`common/versions.py`, §37):

```
app: 1.0.0
model: khmer-support-qwen35-9b-sft-v1
prompt: 1.0
knowledge_index: 2026-08-14.2
dataset: sft-2026-08-v1
```

All five are returned by `GET /health`, `GET /v1/models` and
`GET /v1/admin/diagnostics`, so a production incident can be tied to an exact
combination without reading logs. Each rolls back independently — see
`docs/operations_runbook.md`.

## 9. What is deliberately not here

- **No cloud inference.** Company documents never leave the premises.
- **No conversation persistence by default.** Memory only, 1-hour TTL.
- **No fine-tuned business facts.** See §1.
- **No ANN index at this corpus size.** See §5.
- **No abstractive summarisation in the request path.** See §5.
- **No automatic model selection by size.** 4B versus 9B is decided by measured
  data (§29), and the decision matrix is generated by
  `evaluation/benchmark_model.py`.
