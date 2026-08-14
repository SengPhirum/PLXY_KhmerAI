# Final security report

> **Status: PARTIALLY MEASURED.** The static suites run in CI and pass. The
> behavioural suite requires a running model and has not been executed.

Date: `________`  Reviewer: `________`  Model: `________`

## Static suites (measured in CI)

| Suite | Result |
|---|---|
| `security/tests/test_prompt_injection.py` | 45 tests pass |
| `security/tests/test_security_controls.py` | 46 tests pass |
| Detector block rate on the documented corpus | ≥ 95% (gate enforced by test) |
| False positives on the benign Khmer corpus | 0 |
| Repository secret scan | clean |

## Behavioural suite (requires a model — NOT YET RUN)

```bash
python -m evaluation.evaluate_hallucination \
  --adversarial evaluation/golden/adversarial.jsonl --backend api \
  --report-name prompt_injection
```

| Family | Attacks | Blocked | Rate |
|---|---:|---:|---:|
| A. Direct injection | 10 | | |
| B. Indirect (in-document) | 8 | | |
| C. Prompt/context extraction | 4 | | |
| D. Encoding and obfuscation | 5 | | |
| E. Business-fact manipulation | 5 | | |
| F. Safety and out-of-scope | 4 | | |
| **Total** | **36** | | |

Gate: ≥ 0.95 on high-severity families.

## Controls verified

| Control | Verified by | Status |
|---|---|:---:|
| Request size limit (64 KB) | test | ✅ |
| Rate limiting | test | ✅ |
| Admin authentication, constant-time | test | ✅ |
| Path traversal blocked | test | ✅ |
| File-type allow-list | test | ✅ |
| Secrets never logged | test | ✅ |
| PII redaction incl. Khmer numerals | test | ✅ |
| Internal documents not retrievable | test | ✅ |
| Injected documents quarantined | test | ✅ |
| Injected chunks dropped at query time | test | ✅ |
| Output leak detection | test | ✅ |
| Grounding enforcement | test | ✅ |
| Ollama not exposed off-host | deployment check | ☐ |
| TLS on any exposed endpoint | deployment check | ☐ |
| Backups encrypted | deployment check | ☐ |

## Penetration test

Performed by: `________`  Date: `________`  Findings: `________`

## Outstanding risks

| Risk | Severity | Mitigation | Accepted by |
|---|---|---|---|
