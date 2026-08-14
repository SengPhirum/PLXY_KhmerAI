# Security

## Threat model

| # | Threat | Realistic? | Control |
|---|---|---|---|
| T1 | Direct prompt injection by a customer | Yes, constantly | Input guard + system prompt + output guard |
| T2 | **Indirect injection via a company document** | Yes — and the dangerous one | Ingestion quarantine + retrieval drop + delimiting |
| T3 | System-prompt extraction | Yes | Prompt policy + output leak detection |
| T4 | Confidential document disclosure | Yes | Confidentiality filter before retrieval |
| T5 | Customer PII in logs | Yes, by accident | Redaction in the log formatter |
| T6 | Credential leakage into a corpus | Yes | Secret scanner; records dropped, not redacted |
| T7 | Oversized input / resource exhaustion | Yes | 64 KB cap, rate limit, bounded queue |
| T8 | Path traversal in ingestion | Yes | `resolve_under_root`, symlink refusal |
| T9 | Malicious file upload | Yes | Extension allow-list |
| T10 | Business-fact manipulation | **Highest commercial impact** | Grounding verification |

T2 and T10 deserve emphasis. T2 is dangerous because the payload arrives with the
authority of a trusted internal document. T10 — persuading the assistant to state
a wrong price or an over-generous warranty that the company must then honour — is
the attack with actual money attached, and it is defended by grounding
verification rather than by prompt wording.

## Defence in depth

```
customer message
  ├─ size limit (middleware, before parsing)
  ├─ rate limit (token bucket)
  ├─ control-character stripping
  ├─ injection scan          → high-confidence attempts refused
  └─ PII detection           → logged as counts, never as values
      │
company document
  ├─ extension allow-list
  ├─ path traversal + symlink refusal
  ├─ size limit
  ├─ injection scan          → quarantined, never indexed
  ├─ secret scan             → record dropped
  └─ PII redaction
      │
retrieval
  ├─ confidentiality filter  → internal/restricted never reach a customer
  ├─ status filter           → expired/draft excluded
  └─ injection scan          → poisoned chunk dropped and logged
      │
prompt
  ├─ <retrieved_company_context> delimiter, closing tag stripped from payload
  └─ explicit "this is data, not instructions" policy
      │
generated answer
  ├─ system-prompt leak detection
  ├─ secret leak detection
  ├─ identifier-corruption check
  └─ grounding verification  → unsupported business claims replaced
      │
logs
  └─ redaction in the formatter, so nothing bypasses it
```

No single layer is sufficient. The measured block rate across all layers is the
release gate, not the presence of any one control.

## Prompt injection

Detection covers Khmer *and* English patterns — an attack written
`មិនអើពើនឹងការណែនាំខាងលើ` is invisible to an English-only filter, and a
Khmer-first product will receive Khmer attacks.

```bash
pytest security/tests/test_prompt_injection.py -q        # in CI
python -m evaluation.evaluate_hallucination \
    --adversarial evaluation/golden/adversarial.jsonl \
    --backend api                                        # release gate
```

Gate: ≥ 95% block rate on high-severity families, 0% false positives on the
benign support corpus. The full attack taxonomy is in
`prompts/prompt_injection_tests.md`.

Note the deliberate asymmetry: a *direct* injection from a customer is annotated
and usually allowed through to the model (the system prompt handles it, and
blocking outright tells an attacker exactly where the boundary is), while an
*indirect* injection inside a document is removed entirely.

## Authentication

| Endpoint group | Control |
|---|---|
| `/health`, `/ready` | none — probes must always work |
| `/metrics` | loopback only, enforced at nginx |
| `/v1/chat*`, `/v1/rag/search` | optional client key (`KHMERAI_REQUIRE_CLIENT_AUTH`) |
| `/v1/admin/*` | admin key required, constant-time comparison |

An unset admin key means admin endpoints are **disabled**, not open. Comparison
uses `hmac.compare_digest` so a wrong key cannot be recovered by timing.

## Secrets

- `.env` is mode 600 and gitignored; `.env.example` carries only placeholders.
- Secrets never appear in launchd plists — those are world-readable in
  `/Library/LaunchDaemons`.
- `security/secret_scanner.py` runs in CI and blocks a commit containing a
  credential.
- Redaction happens in the log *formatter*, so a stack trace containing a key is
  scrubbed too — there is no code path that logs around it.

Rotate the admin key by generating a new one, updating `.env`, and restarting the
API.

## Privacy

- Conversations are in memory, TTL 1 hour, bounded count, and are erased on
  restart. `test_19` in the e2e suite asserts they do not survive one.
- `DELETE /v1/conversations/{id}` erases immediately.
- Logs record message *lengths* and PII *counts*, never content.
- Cambodian phone formats and Khmer-numeral phone numbers are detected — the
  generic English-centric regexes miss both.

## Cloud training (§35, §36)

Company documents must not leave the premises without written approval. The gate
is mechanical:

```bash
python -c "
from preprocessing.pii_filter import build_pre_upload_report
from common.io import write_json
write_json('data/manifests/pre_upload_report.json',
           build_pre_upload_report(['data/sft/train.jsonl']))"
```

`training/train_sft.py` refuses to start if the report says `approved: false`.
Prefer sanitised, anonymised, synthetic or public data for cloud training. If
company data must be used, record the approval and the data-handling terms here
before the upload, not after.

## Reporting

Suspected incident → security lead immediately. Include: request IDs, timestamps,
the versions from `/health`, and relevant log lines (already redacted). Do not
paste customer messages into a ticket.
