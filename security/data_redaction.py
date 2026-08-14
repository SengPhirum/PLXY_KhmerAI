"""PII and credential redaction.

Used in three places:

1. **Logging** - every structured log record passes through :func:`redact` so a
   customer phone number or an API key can never land in ``logs/``.
2. **Company-document ingestion** - `company_data/normalize.py` redacts personal
   data before a document is chunked into the retrieval index.
3. **Pre-upload scanning** - `scripts/` uses it before any dataset is copied to
   Colab (§36 of the specification).

Cambodian-specific patterns are included because the generic English-centric
regexes miss them: Cambodian mobile numbers (``0XX XXX XXX`` / ``+855 XX XXX
XXX``), Khmer national ID numbers, and Khmer-digit phone numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "RedactionResult",
    "Redactor",
    "redact",
    "DEFAULT_PATTERNS",
    "KHMER_DIGIT_TRANSLATION",
]

# Khmer numerals ០-៩ so a phone number typed in Khmer digits is still matched.
KHMER_DIGIT_TRANSLATION: Final = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")

# Ordering matters: the most specific patterns run first so that, for example, a
# Cambodian phone number is not first swallowed by the generic number rule.
DEFAULT_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    # --- credentials -------------------------------------------------------
    ("aws_access_key", r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    ("openai_key", r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ("hf_token", r"\bhf_[A-Za-z0-9]{30,}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ("private_key_block", r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z ]+ )?PRIVATE KEY-----"),
    # A leading `[A-Za-z0-9_]*` is required, not decorative: real configuration
    # keys are `db_password`, `ADMIN_API_KEY`, `service_auth_token`, and a plain
    # `\bpassword\b` never matches those because `_` is a word character.
    (
        "assigned_secret",
        r"(?i)(?:^|[^A-Za-z0-9])[A-Za-z0-9_]*"
        r"(?:api[_-]?key|secret[_-]?key|secret|password|passwd|pwd|access[_-]?token|"
        r"auth[_-]?token|client[_-]?secret|token|authorization|bearer)"
        r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-\.~+/]{8,})[\"']?",
    ),
    # --- contact / identity -------------------------------------------------
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ("cambodia_phone_intl", r"(?:\+?855[\s\-.]?)(?:0?[1-9]\d{1,2})[\s\-.]?\d{3}[\s\-.]?\d{3,4}\b"),
    ("cambodia_phone_local", r"\b0(?:1[0-9]|2[0-9]|3[1-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9])[\s\-.]?\d{3}[\s\-.]?\d{3,4}\b"),
    ("khmer_national_id", r"\b\d{9}\b(?=\s*(?:ID|អត្តសញ្ញាណប័ណ្ណ|លេខអត្តសញ្ញាណ))"),
    ("credit_card", r"\b(?:\d[ \-]?){13,19}\b"),
    ("ip_address", r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
)

_PLACEHOLDER = "[REDACTED:{kind}]"


@dataclass(slots=True)
class RedactionResult:
    """Redacted text plus an auditable count of what was removed."""

    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.counts)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class Redactor:
    """Compiled, reusable redactor.

    ``keep_kinds`` lets a caller opt a category out - company documents legitimately
    contain the *company's own* support email and hotline, so ingestion runs with
    ``keep_kinds={"email", "cambodia_phone_local", "cambodia_phone_intl"}`` for
    allow-listed public contact documents while customer transcripts do not.
    """

    def __init__(
        self,
        patterns: tuple[tuple[str, str], ...] = DEFAULT_PATTERNS,
        *,
        keep_kinds: frozenset[str] | set[str] | None = None,
        allowlist: frozenset[str] | set[str] | None = None,
    ) -> None:
        keep = set(keep_kinds or ())
        self._patterns = [
            (kind, re.compile(pattern))
            for kind, pattern in patterns
            if kind not in keep
        ]
        self._allowlist = {a.lower() for a in (allowlist or ())}

    def __call__(self, text: str) -> RedactionResult:
        return self.redact(text)

    def redact(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(text=text)
        counts: dict[str, int] = {}
        out = text
        for kind, pattern in self._patterns:
            placeholder = _PLACEHOLDER.format(kind=kind)

            def _replace(match: re.Match[str], _kind: str = kind, _ph: str = placeholder) -> str:
                matched = match.group(0)
                if matched.lower() in self._allowlist:
                    return matched
                if _kind == "credit_card" and not _luhn_ok(matched):
                    return matched
                counts[_kind] = counts.get(_kind, 0) + 1
                return _ph

            out = pattern.sub(_replace, out)
        return RedactionResult(text=out, counts=counts)

    def scan(self, text: str) -> dict[str, int]:
        """Count matches without rewriting the text (used by the pre-upload report)."""
        return self.redact(text).counts


def _luhn_ok(candidate: str) -> bool:
    """Guard the credit-card rule so ordinary long digit runs are not redacted."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


_DEFAULT_REDACTOR = Redactor()


def redact(text: str) -> RedactionResult:
    """Redact with the default pattern set (module-level, compiled once)."""
    return _DEFAULT_REDACTOR.redact(text)


def normalise_khmer_digits(text: str) -> str:
    """Map Khmer numerals to ASCII so numeric PII rules apply to Khmer input."""
    return text.translate(KHMER_DIGIT_TRANSLATION)


def redact_khmer_aware(text: str) -> RedactionResult:
    """Redact after mapping Khmer digits, then report against the original text.

    A phone number written ``០១២ ៣៤៥ ៦៧៨`` must be caught.  Because the digit
    mapping is 1:1 in code points, offsets are preserved and the redacted string
    can be returned directly.
    """
    return _DEFAULT_REDACTOR.redact(normalise_khmer_digits(text))
