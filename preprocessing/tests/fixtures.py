"""Real Khmer text fixtures shared by the preprocessing tests.

These strings are hand-checked, valid Khmer.  The central invariant of the whole
preprocessing layer is that **none of them may be altered by normalisation**, so
they are used as the regression corpus for every change to the Unicode rules.
"""

from __future__ import annotations

from typing import Final

# --- valid Khmer that must survive normalisation byte-for-byte --------------
VALID_KHMER: Final[tuple[tuple[str, str], ...]] = (
    ("greeting", "សួស្តី! តើខ្ញុំអាចជួយអ្វីបានខ្លះ?"),
    ("polite_question", "សូមទោស តើលោកអ្នកមានសេវាកម្មដឹកជញ្ជូនដល់ខេត្តទេ?"),
    ("warranty", "ទូរទឹកកកម៉ូដែល QN-4500A មានការធានារយៈពេល ២ ឆ្នាំ។"),
    ("coeng_cluster", "ស្ថានភាពនៃការដឹកជញ្ជូនត្រូវបានធ្វើបច្ចុប្បន្នភាព។"),
    ("subscript_stack", "ខ្ញុំបាទសូមអរគុណចំពោះការគាំទ្ររបស់លោកអ្នក។"),
    ("shifter", "ម៉ោងធ្វើការគឺចាប់ពីម៉ោង ៨:០០ ដល់ម៉ោង ១៧:០០។"),
    ("khmer_numerals", "តម្លៃ ១២០ ដុល្លារ សម្រាប់ចំនួន ៣ គ្រឿង។"),
    ("independent_vowel", "ឥឡូវនេះ ឧបករណ៍នេះអស់ពីស្តុកហើយ។"),
    ("punctuation", "សូមអរគុណ។ យើងនឹងទាក់ទងទៅវិញឆាប់ៗនេះ៕"),
    ("repeat_sign", "សូមរង់ចាំបន្តិចៗ ខ្ញុំកំពុងពិនិត្យមើល។"),
    ("complaint", "ខ្ញុំមិនសប្បាយចិត្តទេ ព្រោះទំនិញមកដល់យឺតពេលណាស់។"),
    ("riel_currency", "តម្លៃសរុបគឺ ៤៨០០០៛ រួមបញ្ចូលពន្ធរួចហើយ។"),
)

# --- code-switching -------------------------------------------------------
CODE_SWITCHED: Final[tuple[str, ...]] = (
    "តើ model QN-4500A មាន warranty ប៉ុន្មានឆ្នាំ?",
    "ខ្ញុំចង់ដឹងពី price របស់ Samsung RF-22B ជាបន្ទាន់។",
    "សូមផ្ញើ link តាម https://example.com/products/qn4500a មកខ្ញុំ។",
    "Delivery ទៅសៀមរាបចំណាយពេលប៉ុន្មានថ្ងៃ?",
    "អាចបង់ប្រាក់តាម ABA Pay ឬ Wing បានទេ?",
)

# --- noisy / mistyped input the assistant must still understand ------------
NOISY_KHMER: Final[tuple[str, ...]] = (
    "តើទូរទឹកកកនេះតម្លៃប៉ុន្មាន",  # no final punctuation
    "សូមជួយបន្តិច ខ្ញុំមិនយល់ទេ",  # informal, missing spaces
    "warranty ប៉ុន្មានឆ្នាំ",  # fragment, English first
    "តម្លៃ?",  # very short
    "ខ្ញំុចង់ដឹងអំពីការធានា",  # swapped nikahit/vowel (mis-typed)
)

# --- sequences that normalisation is expected to repair --------------------
# (label, input, expected output)
REPAIRABLE: Final[tuple[tuple[str, str, str], ...]] = (
    ("deprecated_qaa", "ឤយុ", "អាយុ"),
    ("deprecated_qaq", "ឣក្សរ", "អក្សរ"),
    ("split_vowel_oo", "កេាះ", "កោះ"),
    ("swapped_nikahit", "ខ្ញំុ", "ខ្ញុំ"),
    ("doubled_coeng", "ស្្ថាន", "ស្ថាន"),
    ("duplicate_vowel", "កាា", "កា"),
    ("shifter_after_vowel", "មោ៉ង", "ម៉ោង"),
    ("fullwidth_model", "ＱＮ－４５００Ａ", "QN-4500A"),
    ("nbsp", "តម្លៃ ១០០", "តម្លៃ ១០០"),
    ("bom_and_zwnj", "﻿តម្លៃ‌ថ្មី", "តម្លៃថ្មី"),
)

# --- garbage that quality filtering must reject ----------------------------
LOW_QUALITY: Final[tuple[tuple[str, str], ...]] = (
    ("empty", ""),
    ("whitespace", "   \n\t  "),
    ("char_run", "ាាាាាាាាាាាាាាាាាាាាាាាាាាាាាាាាាាាា"),
    ("symbol_noise", "!!!@@@###$$$%%%^^^&&&***((()))___+++==="),
    ("english_only", "This page is entirely in English and contains no Khmer text whatsoever."),
    (
        "link_farm",
        "https://a.example https://b.example https://c.example https://d.example "
        "https://e.example https://f.example",
    ),
    (
        "repeated_lines",
        "\n".join(["ចុចទីនេះដើម្បីអានបន្ថែម"] * 12),
    ),
)

HTML_SAMPLE: Final = """<!DOCTYPE html>
<html lang="km"><head><title>ការធានា</title>
<style>.x{display:none}</style><script>var a=1;</script></head>
<body>
<nav class="main-nav">ទំព័រដើម | ផលិតផល | ទំនាក់ទំនង</nav>
<div id="cookie-consent">This website uses cookies to improve your experience.</div>
<article><h1>គោលការណ៍ធានា</h1>
<p>ផលិតផលអេឡិចត្រូនិកទាំងអស់មានការធានារយៈពេល ២ ឆ្នាំ ចាប់ពីថ្ងៃទិញ។</p>
<p>អតិថិជនត្រូវរក្សាទុកវិក្កយបត្រដើម ដើម្បីទាមទារសេវាកម្មធានា។</p></article>
<footer>© 2026 Example Co. All rights reserved.</footer>
</body></html>"""

SAMPLE_DOCUMENT_KM: Final = (
    "គោលការណ៍ធានាផលិតផល\n"
    "ផលិតផលអេឡិចត្រូនិកទាំងអស់មានការធានារយៈពេល ២ ឆ្នាំ ចាប់ពីថ្ងៃទិញ។ "
    "ការធានាគ្របដណ្តប់លើកំហុសផលិតកម្ម ប៉ុន្តែមិនរាប់បញ្ចូលការខូចខាតដោយសារការប្រើប្រាស់មិនត្រឹមត្រូវឡើយ។ "
    "អតិថិជនត្រូវបង្ហាញវិក្កយបត្រដើមនៅពេលទាមទារសេវាកម្មធានា។"
)
