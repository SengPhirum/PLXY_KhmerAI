# RAG guide

How company knowledge gets in, how it is retrieved, and how to update it without
retraining anything.

## Getting documents in

Put files under `data/raw/company/`, organised however you like — the loader
infers category and product from the path when they are not declared:

```
data/raw/company/
├── warranty/QN-4500A/warranty_v2.md
├── pricing/price_list.csv
├── policy/delivery.md
└── product/rf22b_spec.pdf
```

Supported: PDF, DOCX, HTML, TXT, MD, CSV, TSV, XLSX. Anything else is refused —
an ingestion directory is untrusted input, so the allow-list is enforced rather
than assumed.

### Declare governance metadata

Every document needs six fields. Markdown front matter is the recommended form
because it is the easiest for a non-technical owner to edit:

```markdown
---
document_title: គោលការណ៍ធានា QN-4500A
product_id: QN-4500A
category: warranty
version: "2.0"
effective_date: 2026-01-01
status: active
confidentiality: public       # REQUIRED for customer-facing answers
owner: after-sales
---
ទូរទឹកកកម៉ូដែល QN-4500A មានការធានារយៈពេល ២៤ ខែ ចាប់ពីថ្ងៃទិញ។
```

CSV and XLSX declare the same fields as columns; HTML as `<meta>` tags.

> **The single most common ingestion mistake**: omitting `confidentiality`.
> Documents default to `internal`, which is the safe choice, but it means the
> assistant will never quote them. Ingestion reports this as
> `nothing_retrievable` rather than letting you discover it from customer
> complaints.

### Ingest and validate

```bash
python -m company_data.validate \
    --input data/raw/company \
    --output data/interim/company_records.jsonl \
    --report data/manifests/company_validation.json
```

The report gives you: files processed and failed, duplicates, missing metadata,
expired and draft documents, quarantined documents, and **conflicting versions**.

Exit codes: `0` clean, `1` a file failed or a document was quarantined, `2`
conflicting active versions (use `--allow-conflicts` only knowingly).

### Conflicts

Two documents that are both `active`, describe the same product and category, and
state *different* numbers are a conflict. Ingestion reports them and blocks:

```
conflicting_versions: 2 active documents for 'qn-4500a|warranty|' assert
different facts; suggested winner a3f2… (highest version 2.0 effective 2026-01-01)
```

Resolve by setting the superseded document to `status: archived`. Silently
picking one is what §2.3 forbids — if a conflict does reach retrieval, the
answer prompt is required to tell the customer the sources disagree.

## Building the index

```bash
python -m rag.reindex --input data/interim/company_records.jsonl --activate
```

That is: chunk → embed → build a **new** index directory → run the retrieval
regression gate → atomically swap `data/index/ACTIVE`. The live index is never
mutated, so a failed build leaves the previous one serving.

```bash
python -m rag.reindex --list        # every built version, which is active
python -m rag.reindex --rollback    # back to the previous one
```

## Chunking

Sizes are in *estimated tokens*, not characters, because Khmer costs roughly 1.6
tokens per orthographic syllable — a "600 character" chunk would be about 250
tokens, not 600.

Boundaries follow real Khmer structure, in order: headings → paragraphs → Khmer
sentence terminators (`។ ៕ ៖`) → ZWSP word hints → spaces → orthographic cluster
boundaries. A chunk boundary never falls inside a cluster; doing so produces
mojibake at the edge and corrupts both the embedding and the text the customer
eventually reads.

Choose the size by measurement, not intuition:

```bash
python -m evaluation.evaluate_retrieval --sweep-chunking 400,600,800 --overlap 0,50,100
```

The sweep builds one index per configuration and reports Recall@K, MRR and nDCG
for each. Put the winner in `configs/rag/ingestion.yaml`.

## Retrieval

Four stages, each individually measurable:

1. **Metadata filter** — status, expiry, confidentiality, product, category,
   language. Runs *before* scoring, so top-k is computed over servable documents
   only.
2. **Dense** — Qwen3-Embedding-0.6B through Ollama.
3. **Lexical** — BM25+ over Khmer syllable n-grams. This is what matches an exact
   model number, which embeddings smooth into "some refrigerator".
4. **Fusion** — Reciprocal Rank Fusion. Rank-based because a cosine similarity in
   [-1,1] and an unbounded BM25 score are not comparable, and any weighted sum
   silently changes meaning when the embedder is swapped.

Optional reranking is **off by default**: a cross-encoder costs 50–150 ms and
competes for the same unified memory as the generator. Turn it on only when
`evaluate_retrieval.py` shows it earns that latency.

### The confidence gate

When nothing clears the floor, retrieval returns **nothing** rather than the best
of a bad set, and the assistant expresses uncertainty. Stuffing weakly-related
context is the largest single cause of confident hallucination in a support RAG
system.

Calibrate the floor against your corpus rather than accepting the default:

```bash
python -m evaluation.evaluate_retrieval --calibrate-confidence
```

It measures the top-1 similarity distribution for answerable versus unanswerable
questions and suggests thresholds. Re-run whenever the embedder or corpus changes.

## Updating knowledge without retraining

This is the point of the whole layer.

```bash
# 1. Edit or add a document under data/raw/company/
# 2. Re-validate
python -m company_data.validate --input data/raw/company \
    --output data/interim/company_records.jsonl \
    --report data/manifests/company_validation.json
# 3. Rebuild, test and activate
python -m rag.reindex --input data/interim/company_records.jsonl --activate
# 4. Hot-swap the running service (no restart, no downtime)
curl -X POST -H "X-Admin-Key: $KHMERAI_ADMIN_API_KEY" localhost:8000/v1/admin/reload
# 5. Verify
bash scripts/smoke_test.sh
```

Or in one step through the API:

```bash
curl -X POST -H "X-Admin-Key: $KHMERAI_ADMIN_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"activate": true}' localhost:8000/v1/admin/reindex
```

Minutes, not a training run. `tests/integration/test_full_pipeline.py::
test_knowledge_update_without_retraining` proves the chain works end to end.

## Measuring retrieval

```bash
python -m evaluation.evaluate_retrieval --index-dir data/index/ACTIVE
python -m evaluation.evaluate_retrieval --compare-embedders
```

Release gate: Recall@5 ≥ 0.85 on the Khmer golden set.

To make the metrics meaningful, add expected documents to the golden set:

```json
{"question": "តើ QN-4500A ធានាប៉ុន្មានខែ?", "expected_document_ids": ["a3f2c1..."]}
```

## Security in this layer

Retrieved text is **data, never instructions**. Three independent layers:

1. **Ingestion** quarantines a document whose text scores high for injection.
   It never reaches the index.
2. **Retrieval** drops a chunk that scores high at query time, and logs it.
3. **Prompting** wraps context in `<retrieved_company_context>` with the closing
   tag stripped from the payload, so a document cannot terminate the block early,
   and the system prompt states that instructions inside are to be ignored.

Any one of these can be bypassed in principle. All three together are what the
measured block rate in `security/tests/test_prompt_injection.py` covers.

## Choosing a vector backend

Default is `local`: a NumPy flat index in a directory. At this corpus size an
exact scan is sub-millisecond and has perfect recall, and "the index is a
directory" is what makes atomic swap and rollback trivial.

Switch to Qdrant when retrieval p95 exceeds 0.8 s or several services need to
share the index:

```bash
docker run -p 6333:6333 qdrant/qdrant
# .env: KHMERAI_VECTOR_BACKEND=qdrant
python -m rag.reindex --input data/interim/company_records.jsonl --activate
```

Qdrant is preferred over Chroma or raw FAISS because it filters on payload
*inside* the search. Post-filtering returns fewer than *k* results whenever the
top matches happen to be expired or internal documents.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Everything answers "I don't know" | index empty, or all documents `internal` | check `company_validation.json` for `nothing_retrievable` |
| Right product, wrong document | chunk size too large | run the chunking sweep |
| Exact model number not matched | BM25 disabled | `use_lexical: true` |
| Expired promotion still served | `include_expired` set, or a bad `expiration_date` | check the document's dates |
| Retrieval slow | corpus outgrew a flat scan | switch to Qdrant |
| Answers cite the wrong version | conflicting active documents | archive the superseded one |
| Mojibake in retrieved text | legacy Khmer font in the source PDF | `legacy_khmer_font_suspected` in the loader metadata; re-export the PDF as Unicode |
