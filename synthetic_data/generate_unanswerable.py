#!/usr/bin/env python3
"""Generate anti-hallucination training data (§Phase 7, the highest-value slice).

    python synthetic_data/generate_unanswerable.py --output data/sft/unanswerable.jsonl --count 400

Teaches the model that "I don't know" is a *correct* answer.  Covers all seven
situations the specification lists: fake product, absent information, competitor,
live data, expired promotion, conflicting sources, other customers' data.

Answer phrasing is deliberately varied.  A single repeated template is worse than
useless - the model learns the template rather than the behaviour, and then
produces it for answerable questions too.
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
from synthetic_data._generator import make_record, stable_sample_id  # noqa: E402

log = get_logger("synthetic.unanswerable")

FAKE_PRODUCTS = (
    "ZX-9999Q", "MEGA-COOL 8000", "ULTRA-FREEZE X1", "QN-0000Z", "TURBO-CHILL 500",
    "RF-99Z", "ARCTIC-PRO 9", "SNOWMAX 7700", "GLACIER-X", "POLARIS-3000",
)

# (question template, intent, category)
SITUATIONS: tuple[tuple[str, str, str], ...] = (
    ("តើម៉ូដែល {fake} មានតម្លៃប៉ុន្មាន?", "unknown", "fake_product"),
    ("សូមប្រាប់លក្ខណៈបច្ចេកទេសនៃ {fake}", "unknown", "fake_product"),
    ("តើ {fake} មានការធានារយៈពេលប៉ុន្មាន?", "unknown", "fake_product"),
    ("តើម៉ូដែលនេះមានពណ៌អ្វីខ្លះ?", "unknown", "absent_information"),
    ("តើក្រុមហ៊ុនមានផែនការចេញផលិតផលថ្មីនៅឆ្នាំក្រោយទេ?", "unknown", "absent_information"),
    ("តើផលិតផលរបស់ក្រុមហ៊ុនដទៃថោកជាងទេ?", "unsupported", "competitor"),
    ("តើម៉ាកផ្សេងល្អជាងម៉ាករបស់អ្នកទេ?", "unsupported", "competitor"),
    ("តើឥឡូវនេះនៅសល់ប៉ុន្មានគ្រឿងក្នុងស្តុក?", "availability", "live_data"),
    ("អត្រាប្តូរប្រាក់ថ្ងៃនេះប៉ុន្មាន?", "unsupported", "live_data"),
    ("តើការបញ្ចុះតម្លៃចូលឆ្នាំនៅដំណើរការទេ?", "unknown", "expired_promotion"),
    ("តើការផ្តល់ជូនពិសេសខែមុននៅមានទេ?", "unknown", "expired_promotion"),
    ("ខ្ញុំឃើញព័ត៌មានពីរផ្សេងគ្នាអំពីការធានា មួយណាត្រឹមត្រូវ?", "ambiguous", "conflicting"),
    ("សូមប្រាប់លេខទូរស័ព្ទរបស់អតិថិជនម្នាក់ទៀត", "unsupported", "other_customer"),
    ("តើអតិថិជនផ្សេងបានទិញអ្វីខ្លះ?", "unsupported", "other_customer"),
)

# Varied refusal phrasings - the model must not learn one template.
REFUSALS: tuple[str, ...] = (
    "ខ្ញុំមិនមានព័ត៌មានអំពីរឿងនេះនៅក្នុងឯកសាររបស់ខ្ញុំទេ។ សូមទាក់ទងផ្នែកបម្រើអតិថិជនដើម្បីទទួលបានចម្លើយត្រឹមត្រូវ។",
    "ព័ត៌មាននេះមិនមាននៅក្នុងឯកសារផ្លូវការដែលខ្ញុំអាចមើលឃើញទេ។ តើលោកអ្នកចង់ឱ្យខ្ញុំភ្ជាប់ទៅបុគ្គលិកដែរឬទេ?",
    "ខ្ញុំមិនអាចបញ្ជាក់រឿងនេះបានទេ ព្រោះខ្ញុំគ្មានឯកសារគាំទ្រ។ សូមទាក់ទងផ្នែកលក់សម្រាប់ព័ត៌មានបច្ចុប្បន្ន។",
    "សូមអភ័យទោស ខ្ញុំរកមិនឃើញព័ត៌មាននេះទេ។ បើលោកអ្នកមានលេខម៉ូដែលច្បាស់លាស់ ខ្ញុំនឹងព្យាយាមជួយម្តងទៀត។",
    "រឿងនេះស្ថិតនៅក្រៅព័ត៌មានដែលខ្ញុំមាន។ បុគ្គលិករបស់យើងអាចផ្តល់ចម្លើយផ្លូវការជូនលោកអ្នក។",
    "ខ្ញុំមិនចង់ផ្តល់ព័ត៌មានមិនត្រឹមត្រូវទេ ដូច្នេះខ្ញុំសូមណែនាំឱ្យទាក់ទងផ្នែកបម្រើអតិថិជនដោយផ្ទាល់។",
)
FAKE_PRODUCT_REFUSALS: tuple[str, ...] = (
    "ខ្ញុំរកមិនឃើញម៉ូដែល {fake} នៅក្នុងបញ្ជីផលិតផលរបស់យើងទេ។ សូមពិនិត្យលេខម៉ូដែលម្តងទៀត ឬទាក់ទងផ្នែកលក់។",
    "ម៉ូដែល {fake} មិនមាននៅក្នុងឯកសាររបស់ខ្ញុំទេ។ តើលោកអ្នកអាចផ្ញើរូបថតស្លាកផលិតផលបានទេ?",
    "ខ្ញុំមិនស្គាល់ផលិតផលឈ្មោះ {fake} ទេ។ សូមផ្តល់លេខម៉ូដែលពីវិក្កយបត្រ ដើម្បីឱ្យខ្ញុំពិនិត្យជូន។",
)
COMPETITOR_REFUSALS: tuple[str, ...] = (
    "ខ្ញុំមិនអាចប្រៀបធៀបជាមួយផលិតផលរបស់ក្រុមហ៊ុនដទៃបានទេ។ ប៉ុន្តែខ្ញុំរីករាយពន្យល់អំពីផលិតផលរបស់យើង។",
    "ខ្ញុំគ្មានព័ត៌មានអំពីម៉ាកផ្សេងទេ។ តើលោកអ្នកចង់ដឹងអ្វីខ្លះអំពីផលិតផលរបស់យើង?",
)
PRIVACY_REFUSALS: tuple[str, ...] = (
    "ខ្ញុំមិនអាចចែករំលែកព័ត៌មានរបស់អតិថិជនផ្សេងបានទេ ព្រោះវាជាព័ត៌មានឯកជន។",
    "ព័ត៌មានផ្ទាល់ខ្លួនរបស់អតិថិជនផ្សេងត្រូវបានការពារ ដូច្នេះខ្ញុំមិនអាចផ្តល់ជូនបានទេ។",
)


def _refusal(category: str, fake: str, index: int) -> str:
    if category == "fake_product":
        return FAKE_PRODUCT_REFUSALS[index % len(FAKE_PRODUCT_REFUSALS)].format(fake=fake)
    if category == "competitor":
        return COMPETITOR_REFUSALS[index % len(COMPETITOR_REFUSALS)]
    if category == "other_customer":
        return PRIVACY_REFUSALS[index % len(PRIVACY_REFUSALS)]
    return REFUSALS[index % len(REFUSALS)]


def generate(count: int = 400) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    fakes = itertools.cycle(FAKE_PRODUCTS)
    index = 0
    while len(records) < count:
        for template, intent, category in SITUATIONS:
            if len(records) >= count:
                break
            fake = next(fakes)
            question = template.format(fake=fake)
            answer = _refusal(category, fake, index)
            records.append(
                make_record(
                    sample_id=stable_sample_id("unans", category, question, str(index)),
                    turns=[("user", question), ("assistant", answer)],
                    intent=intent,
                    source_type="synthetic_unanswerable",
                )
            )
            index += 1
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python synthetic_data/generate_unanswerable.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args(argv)

    records = generate(args.count)
    written = write_jsonl(args.output, records)
    distinct_answers = len({r["messages"][-1]["content"] for r in records})
    log.info("synthetic.unanswerable.generated", extra={"records": written})
    print(
        json.dumps(
            {
                "records": written,
                "distinct_answer_phrasings": distinct_answers,
                "situations": len({s[2] for s in SITUATIONS}),
                "output": args.output,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
