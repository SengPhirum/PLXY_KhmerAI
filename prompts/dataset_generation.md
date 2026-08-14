# Synthetic SFT data-generation prompts (Phase 7)

Used by `synthetic_data/*.py`.  Every generated example must derive its facts
from an approved company document and keep the `source_id`, so a bad training
sample can be traced back to the document that produced it.

**Hard rule:** the generator may invent *phrasing*, never *facts*.  Any number,
price, period, model name or policy in a generated answer must appear verbatim
in the supplied source document.  `synthetic_data/quality_check.py` enforces
this mechanically with `rag.citations.verify_grounding`, and unverifiable
samples are rejected rather than shipped.

---

## 1. Grounded support turn

```
You are creating training data for a Khmer customer-support assistant.

SOURCE DOCUMENT (the only permitted source of facts):
---
{{document_text}}
---
Document metadata: product_id={{product_id}} category={{category}}
version={{version}} effective_date={{effective_date}}

Produce {{n}} distinct customer-support exchanges in JSON Lines.

Each line:
{"messages":[{"role":"user","content":"<Khmer question>"},
             {"role":"assistant","content":"<Khmer answer>"}],
 "metadata":{"intent":"<intent>","source_id":"{{document_id}}","synthetic":true}}

Requirements:
- The question must be written the way a real Cambodian customer writes:
  natural, sometimes informal, sometimes without final punctuation.
- Vary the register: polite, blunt, hurried, uncertain.
- The answer must be in Khmer, 1-4 sentences, and must state ONLY facts present
  in the source document.
- Preserve model numbers, prices, dates and units EXACTLY as written above.
- intent must be one of: {{intent_list}}
- Do NOT invent contact details, branch locations, promotions or timelines.
```

## 2. Code-switching (10% of the mixture)

```
Rewrite each question below the way a Cambodian customer would actually type it,
mixing Khmer with the English terms people really keep in English: product
names, model numbers, "warranty", "delivery", "stock", "promotion", "invoice",
brand names, payment apps (ABA, Wing).

Keep the assistant answer in Khmer. The English terms in the question must stay
in English in the answer where they are proper nouns or product identifiers.

Questions:
{{questions}}
```

## 3. Unanswerable / anti-hallucination (10% of the mixture)

This is the highest-value slice.  It teaches the model that "I don't know" is a
*correct* answer, which is the only reliable defence against confident invention.

```
Create {{n}} Khmer customer questions that CANNOT be answered from company
documents, drawn evenly from these situations:

  a) the product does not exist (invent a plausible fake model number)
  b) the information is simply absent from any document
  c) the question asks about a competitor
  d) the question needs live data (today's stock level, today's exchange rate)
  e) the question asks about a promotion that has expired
  f) two documents would contradict each other
  g) the question asks for another customer's information

For each, write the assistant answer that:
  - states clearly and politely, in natural Khmer, that the information is not
    available to the assistant;
  - does NOT apologise more than once;
  - does NOT invent a substitute fact;
  - offers a concrete next step (contact support / provide the model number);
  - is 1-3 sentences.

Never produce an answer that begins "ខ្ញុំសុំទោស ខ្ញុំសុំទោស" or repeats a
formulaic apology - vary the phrasing so the model does not learn one template.

Output JSON Lines with metadata.intent = "unknown" or "unsupported".
```

## 4. Multi-turn and escalation (5% of the mixture)

```
Write {{n}} Khmer conversations of 3-5 turns where:
  - the customer's first message is vague ("ទូរទឹកកកខូច");
  - the assistant asks ONE clarifying question;
  - the customer supplies the model number;
  - the assistant answers from the source document below;
  - in {{escalation_fraction}} of the conversations the customer becomes
    dissatisfied and the assistant escalates to a human correctly.

SOURCE DOCUMENT:
---
{{document_text}}
---

The assistant must carry context forward: once the model number is known it must
not ask for it again. metadata.intent = "multi_turn" or "escalation".
```

## 5. Complaints and difficult customers (10% of the mixture)

```
Write {{n}} Khmer exchanges where the customer is frustrated, angry, or has
already had a bad experience. Cover: late delivery, a repeated fault, a rejected
warranty claim, rude service, a wrong item shipped.

The assistant must:
  - acknowledge the feeling in ONE sentence, without grovelling;
  - state what it can actually do, grounded in the source document;
  - escalate when the issue involves money or a claim decision;
  - never blame the customer;
  - never promise compensation.

Output JSON Lines with metadata.intent = "complaint" or "escalation".
```

## 6. Reviewer prompt (used for the automated pre-screen)

```
You are reviewing one Khmer customer-support training example.

SOURCE DOCUMENT:
{{document_text}}

EXAMPLE:
{{example_json}}

Answer strictly as JSON:
{"khmer_natural": true|false,
 "facts_supported": true|false,
 "unsupported_claims": ["..."],
 "intent_correct": true|false,
 "uncertainty_handled_correctly": true|false,
 "identifiers_preserved": true|false,
 "verdict": "human_approved"|"human_rejected"|"needs_human_review",
 "reason": "<one sentence>"}

Reject if any fact in the answer is absent from the source document, if the
Khmer reads like a machine translation, or if the answer invents a contact
detail, a date or a promotion.
```

---

## Review priority (§33)

Automated screening is not sufficient for these categories - route them to a
native Khmer reviewer:

| Priority | Category | Why |
|---|---|---|
| 1 | warranty, refund, pricing | a wrong answer creates a financial liability |
| 2 | policy, safety | a wrong answer creates a legal or safety exposure |
| 3 | technical troubleshooting | a wrong answer can damage the product |
| 4 | every synthetic example in the first 1,000 | calibrates the generator |
| 5 | random 5% sample thereafter | ongoing drift detection |

Review states: `unreviewed` -> `auto_checked` -> `human_approved` /
`human_rejected`.  Only `human_approved` and `auto_checked` examples enter the
training split; `human_rejected` examples are kept as DPO *rejected* candidates.
