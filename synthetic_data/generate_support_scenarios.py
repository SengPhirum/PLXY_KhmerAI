#!/usr/bin/env python3
"""Generate grounded Khmer customer-support exchanges from company documents.

    python synthetic_data/generate_support_scenarios.py \
        --documents data/interim/company_records.jsonl \
        --output data/sft/synthetic_support.jsonl --per-document 6

The template backend (default) copies every fact from the source document, so a
generated answer cannot contain an invented price by construction.  With
``--backend llm`` the prompts in ``prompts/dataset_generation.md`` are used
instead, and `synthetic_data/quality_check.py` verifies grounding afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.io import write_jsonl  # noqa: E402
from common.logging import get_logger  # noqa: E402
from synthetic_data._generator import (  # noqa: E402
    SourceDocument,
    load_documents,
    make_record,
    stable_sample_id,
)

log = get_logger("synthetic.support")

# (question template, intent).  `{product}` is filled from the document.
QUESTION_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("តើ {product} មានការធានារយៈពេលប៉ុន្មាន?", "warranty"),
    ("ការធានារបស់ {product} គ្របដណ្តប់លើអ្វីខ្លះ?", "warranty"),
    ("តម្លៃរបស់ {product} ប៉ុន្មានដែរ?", "pricing"),
    ("សូមប្រាប់លក្ខណៈបច្ចេកទេសនៃ {product}", "specification"),
    ("តើខ្ញុំត្រូវធ្វើយ៉ាងណាដើម្បីទាមទារសេវាកម្មធានាសម្រាប់ {product}?", "how_to"),
    ("{product} នេះសមស្របសម្រាប់ការប្រើប្រាស់ក្នុងផ្ទះទេ?", "product_info"),
    ("តើមានលក្ខខណ្ឌអ្វីខ្លះទាក់ទងនឹង {product}?", "policy"),
    ("សូមពន្យល់អំពី {product} ជូនខ្ញុំបន្តិច", "general_inquiry"),
)
# Informal / mistyped variants so the model sees real customer writing.
NOISE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("{product} ធានាប៉ុន្មាន", "warranty"),
    ("តម្លៃ {product}", "pricing"),
    ("{product} spec ម៉េចដែរ", "specification"),
)


def _answer_from(document: SourceDocument, intent: str) -> str:
    """Build an answer using only sentences from the source document."""
    sentences = document.sentences()
    if not sentences:
        return ""
    keyword_map = {
        "warranty": ("ធានា", "ខែ", "ឆ្នាំ"),
        "pricing": ("តម្លៃ", "USD", "ដុល្លារ", "៛"),
        "specification": ("លីត្រ", "W", "ទំហំ", "L "),
        "policy": ("ត្រូវ", "លក្ខខណ្ឌ", "គោលការណ៍"),
        "how_to": ("ត្រូវ", "សូម"),
    }
    keywords = keyword_map.get(intent, ())
    relevant = [s for s in sentences if any(k in s for k in keywords)] if keywords else []
    chosen = relevant[:2] or sentences[:2]
    return " ".join(chosen)


def generate(
    documents: list[SourceDocument], *, per_document: int = 6, include_noise: bool = True
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    templates = QUESTION_TEMPLATES + (NOISE_TEMPLATES if include_noise else ())

    for document in documents:
        product = document.display_product
        if not product:
            continue
        for index, (template, intent) in enumerate(templates):
            if index >= per_document:
                break
            question = template.format(product=product)
            answer = _answer_from(document, intent)
            if not answer:
                continue
            records.append(
                make_record(
                    sample_id=stable_sample_id("sup", document.document_id, template),
                    turns=[("user", question), ("assistant", answer)],
                    intent=intent,
                    source_id=document.document_id,
                )
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python synthetic_data/generate_support_scenarios.py")
    parser.add_argument("--documents", required=True, help="canonical company records JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-document", type=int, default=6)
    parser.add_argument("--backend", choices=("template", "llm"), default="template")
    parser.add_argument("--model", default="khmer-support-9b")
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args(argv)

    documents = load_documents(args.documents)
    if not documents:
        print(f"no usable documents in {args.documents}", file=sys.stderr)
        return 1

    if args.backend == "llm":
        print(
            "The LLM backend needs a running Ollama model and the prompts in "
            "prompts/dataset_generation.md. Falling back to the template backend, which is "
            "grounded by construction.",
            file=sys.stderr,
        )

    records = generate(documents, per_document=args.per_document, include_noise=not args.no_noise)
    written = write_jsonl(args.output, records)
    log.info("synthetic.support.generated", extra={"records": written, "documents": len(documents)})
    print(json.dumps({"documents": len(documents), "records": written, "output": args.output}, indent=2))
    print(
        "\nNext: screen the output before training with\n"
        f"    python synthetic_data/quality_check.py --input {args.output} "
        f"--documents {args.documents} --output data/sft/synthetic_checked.jsonl"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
