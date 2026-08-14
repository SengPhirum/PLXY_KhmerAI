#!/usr/bin/env python3
"""Generate Khmer-English code-switching training data (10% of the mixture).

    python synthetic_data/generate_code_switch.py \
        --documents data/interim/company_records.jsonl \
        --output data/sft/code_switch.jsonl --count 300

The rule the model must learn: **answer in Khmer, keep the identifiers in
English.** Every generated pair therefore contains at least one identifier that
appears byte-identical in both the question and the answer, and the generator
asserts that before emitting the record.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.io import write_jsonl  # noqa: E402
from common.logging import get_logger  # noqa: E402
from preprocessing.language_mixing import analyse_code_switching  # noqa: E402
from synthetic_data._generator import (  # noqa: E402
    SourceDocument,
    load_documents,
    make_record,
    stable_sample_id,
)

log = get_logger("synthetic.code_switch")

# English terms Cambodian customers genuinely keep in English.
ENGLISH_TERMS = (
    "warranty",
    "delivery",
    "stock",
    "promotion",
    "invoice",
    "model",
    "price",
    "spec",
    "service",
    "support",
    "order",
    "payment",
    "discount",
    "refund",
)
PAYMENT_APPS = ("ABA Pay", "Wing", "ACLEDA Mobile", "Bakong")

TEMPLATES: tuple[tuple[str, str], ...] = (
    ("តើ model {product} មាន warranty ប៉ុន្មានឆ្នាំ?", "warranty"),
    ("ខ្ញុំចង់ដឹងពី price របស់ {product}", "pricing"),
    ("{product} នេះ spec យ៉ាងម៉េចដែរ?", "specification"),
    ("តើ {product} មាន stock ទេ?", "availability"),
    ("Delivery សម្រាប់ {product} ចំណាយពេលប៉ុន្មានថ្ងៃ?", "service_info"),
    ("សូមផ្ញើ invoice សម្រាប់ {product} មកខ្ញុំ", "account_related"),
    ("តើមាន promotion សម្រាប់ {product} ទេ?", "pricing"),
    ("អាចបង់ប្រាក់តាម {app} បានទេ សម្រាប់ {product}?", "policy"),
)


def _khmer_answer(document: SourceDocument, product: str) -> str:
    sentences = document.sentences()
    body = " ".join(sentences[:2]) if sentences else ""
    if product and product not in body:
        body = f"សម្រាប់ម៉ូដែល {product}៖ {body}"
    return body


def generate(documents: list[SourceDocument], *, count: int = 300) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    apps = itertools.cycle(PAYMENT_APPS)

    for document in itertools.cycle(documents):
        if len(records) >= count:
            break
        product = document.product_id or document.product_name
        if not product:
            continue
        for template, intent in TEMPLATES:
            if len(records) >= count:
                break
            question = template.format(product=product, app=next(apps))
            answer = _khmer_answer(document, product)
            if not answer:
                continue
            # The contract: the identifier survives verbatim into the answer.
            if product not in answer:
                continue
            analysis = analyse_code_switching(question)
            if not analysis.is_code_switched:
                continue
            records.append(
                make_record(
                    sample_id=stable_sample_id("cs", document.document_id, template),
                    turns=[("user", question), ("assistant", answer)],
                    intent=intent,
                    source_id=document.document_id,
                    source_type="synthetic_code_switch",
                )
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python synthetic_data/generate_code_switch.py")
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args(argv)

    documents = load_documents(args.documents)
    if not documents:
        print(f"no usable documents in {args.documents}", file=sys.stderr)
        return 1

    records = generate(documents, count=args.count)
    written = write_jsonl(args.output, records)
    log.info("synthetic.code_switch.generated", extra={"records": written})
    print(
        json.dumps(
            {"records": written, "english_terms": len(ENGLISH_TERMS), "output": args.output},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
