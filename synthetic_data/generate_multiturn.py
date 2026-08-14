#!/usr/bin/env python3
"""Generate multi-turn and escalation training data (5% of the mixture).

    python synthetic_data/generate_multiturn.py \
        --documents data/interim/company_records.jsonl \
        --output data/sft/multiturn.jsonl --count 200

The behaviour being taught: a vague opening, ONE clarifying question, then a
grounded answer that does not ask for the model number again.  Asking twice for
information the customer already gave is the most common multi-turn failure and
the most irritating one.
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
from synthetic_data._generator import (  # noqa: E402
    SourceDocument,
    load_documents,
    make_record,
    stable_sample_id,
)

log = get_logger("synthetic.multiturn")

VAGUE_OPENINGS = (
    "ទូរទឹកកកខូច",
    "ខ្ញុំមានបញ្ហា",
    "ជួយខ្ញុំផង",
    "មិនដំណើរការ",
    "ខ្ញុំចង់សួរអំពីការធានា",
    "មានរឿងចង់សួរបន្តិច",
)
CLARIFYING = (
    "សូមទោស តើលោកអ្នកអាចប្រាប់លេខម៉ូដែលផលិតផលបានទេ? វាមាននៅលើស្លាកខាងក្រោយ ឬលើវិក្កយបត្រ។",
    "បាទ/ចាស ខ្ញុំរីករាយជួយ។ តើលេខម៉ូដែលរបស់ផលិតផលគឺអ្វី?",
    "ដើម្បីឆ្លើយឱ្យបានត្រឹមត្រូវ សូមប្រាប់លេខម៉ូដែលជាមុនសិន។",
)
FOLLOW_UPS = (
    ("តើវាធានារយៈពេលប៉ុន្មាន?", "warranty"),
    ("តម្លៃប៉ុន្មានដែរ?", "pricing"),
    ("តើវាមានលក្ខណៈបច្ចេកទេសយ៉ាងណា?", "specification"),
)
ESCALATION_TRIGGERS = (
    "ខ្ញុំបានរង់ចាំ ២ សប្តាហ៍ហើយ ខ្ញុំចង់សងប្រាក់វិញ",
    "ខ្ញុំមិនពេញចិត្តទេ ខ្ញុំចង់និយាយជាមួយអ្នកគ្រប់គ្រង",
    "នេះជាលើកទីបីហើយដែលវាខូច ខ្ញុំចង់ប្តូរថ្មី",
)
ESCALATION_ANSWERS = (
    "ខ្ញុំយល់ពីការខកចិត្តរបស់លោកអ្នក ហើយសូមអភ័យទោសចំពោះបញ្ហានេះ។ "
    "រឿងទាក់ទងនឹងការសងប្រាក់ត្រូវការការសម្រេចចិត្តពីបុគ្គលិក ដូច្នេះខ្ញុំនឹងភ្ជាប់លោកអ្នកទៅផ្នែកបម្រើអតិថិជនភ្លាមៗ។",
    "សូមអភ័យទោសចំពោះបទពិសោធន៍នេះ។ ខ្ញុំនឹងបញ្ជូនករណីរបស់លោកអ្នកទៅបុគ្គលិកដែលអាចសម្រេចចិត្តបាន។",
)


def generate(
    documents: list[SourceDocument], *, count: int = 200, escalation_fraction: float = 0.3
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    openings = itertools.cycle(VAGUE_OPENINGS)
    clarifiers = itertools.cycle(CLARIFYING)
    triggers = itertools.cycle(ESCALATION_TRIGGERS)
    escalation_answers = itertools.cycle(ESCALATION_ANSWERS)

    for index, document in enumerate(itertools.cycle(documents)):
        if len(records) >= count:
            break
        product = document.product_id or document.product_name
        if not product:
            continue
        sentences = document.sentences()
        if not sentences:
            continue

        follow_up, _intent = FOLLOW_UPS[index % len(FOLLOW_UPS)]
        answer = " ".join(sentences[:2])
        turns: list[tuple[str, str]] = [
            ("user", next(openings)),
            ("assistant", next(clarifiers)),
            ("user", product),
            ("assistant", f"សូមអរគុណ។ ខ្ញុំបានកត់ត្រាម៉ូដែល {product}។ តើលោកអ្នកចង់ដឹងអ្វីខ្លះ?"),
            ("user", follow_up),
            ("assistant", answer),
        ]
        escalate = (index % 10) < int(escalation_fraction * 10)
        if escalate:
            turns.append(("user", next(triggers)))
            turns.append(("assistant", next(escalation_answers)))

        records.append(
            make_record(
                sample_id=stable_sample_id("mt", document.document_id, str(index)),
                turns=turns,
                intent="escalation" if escalate else "multi_turn",
                source_id=document.document_id,
                source_type="synthetic_multiturn",
            )
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python synthetic_data/generate_multiturn.py")
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--escalation-fraction", type=float, default=0.3)
    args = parser.parse_args(argv)

    documents = load_documents(args.documents)
    if not documents:
        print(f"no usable documents in {args.documents}", file=sys.stderr)
        return 1

    records = generate(documents, count=args.count, escalation_fraction=args.escalation_fraction)
    written = write_jsonl(args.output, records)
    escalations = sum(1 for r in records if r["metadata"]["intent"] == "escalation")
    log.info(
        "synthetic.multiturn.generated", extra={"records": written, "escalations": escalations}
    )
    print(
        json.dumps(
            {"records": written, "escalations": escalations, "output": args.output}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
