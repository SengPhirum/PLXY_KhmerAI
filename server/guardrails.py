"""Input and output guardrails.

Input (before anything reaches the model):
  size limits, control-character stripping, injection scoring, intent
  classification, and PII detection on the customer's own message.

Output (before anything reaches the customer):
  grounding verification, identifier-corruption detection, system-prompt leak
  detection, secret leak detection, and escalation decisions.

Output validation is the last line of defence and the only one that sees what
the model actually said, so it is deliberately conservative: when it cannot
verify a business claim it downgrades the answer to an uncertainty response
rather than passing an unverified number to a customer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from common.logging import get_logger
from preprocessing.khmer_detection import TextLanguage, detect_language
from preprocessing.language_mixing import verify_protected_spans
from preprocessing.schemas import INTENTS
from rag.citations import GroundingReport, verify_grounding
from rag.schemas import RetrievedChunk
from security.data_redaction import redact
from security.prompt_injection import scan_for_injection
from security.secret_scanner import scan_text
from server.schemas import EscalationReason

log = get_logger(__name__)

__all__ = [
    "InputVerdict",
    "OutputVerdict",
    "InputGuard",
    "OutputGuard",
    "classify_intent",
    "GROUNDING_REQUIRED_INTENTS",
]

# Intents whose answers MUST be supported by retrieved company documents.
GROUNDING_REQUIRED_INTENTS = frozenset(
    {
        "pricing", "specification", "availability", "warranty", "returns",
        "refund", "policy", "product_info", "service_info",
    }
)

# Khmer-first intent cues.  Ordered: the first intent whose cue matches wins, so
# the more specific intents are listed before the general ones.
_INTENT_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("escalation", ("និយាយជាមួយមនុស្ស", "បុគ្គលិកពិត", "អ្នកគ្រប់គ្រង", "speak to a human", "talk to an agent", "manager")),
    ("refund", ("សងប្រាក់", "សំណង", "យកលុយវិញ", "refund", "money back")),
    ("returns", ("ប្តូរទំនិញ", "ប្រគល់មកវិញ", "return the item", "exchange")),
    ("warranty", ("ធានា", "warranty", "guarantee")),
    ("pricing", ("តម្លៃ", "ថ្លៃ", "ប៉ុន្មានលុយ", "price", "cost", "how much")),
    ("availability", ("មានស្តុក", "នៅសល់", "អស់ពីស្តុក", "in stock", "available")),
    ("installation", ("តំឡើង", "ដំឡើង", "install", "setup")),
    ("troubleshooting", ("ខូច", "មិនដំណើរការ", "បញ្ហា", "ជួសជុល", "not working", "broken", "error")),
    ("comparison", ("ប្រៀបធៀប", "ណាមួយល្អជាង", "compare", "versus", " vs ")),
    ("recommendation", ("ណែនាំ", "គួរទិញ", "recommend", "suggest")),
    ("specification", ("លក្ខណៈបច្ចេកទេស", "ទំហំ", "ថាមពល", "spec", "dimension", "capacity")),
    ("complaint", ("មិនពេញចិត្ត", "តូចចិត្ត", "សេវាកម្មអន់", "complain", "disappointed", "terrible")),
    ("policy", ("គោលការណ៍", "លក្ខខណ្ឌ", "policy", "terms")),
    ("account_related", ("គណនី", "ពាក្យសម្ងាត់", "account", "password", "login")),
    ("safety", ("គ្រោះថ្នាក់", "ភ្លើងឆេះ", "ឆក់ខ្សែ", "danger", "fire", "shock", "injury")),
    ("how_to", ("របៀប", "ធ្វើដូចម្តេច", "how do i", "how to")),
    ("greeting", ("សួស្តី", "ជំរាបសួរ", "អរុណសួស្តី", "hello", "hi ", "good morning")),
    ("product_info", ("ផលិតផល", "ម៉ូដែល", "product", "model")),
    ("service_info", ("សេវាកម្ម", "ដឹកជញ្ជូន", "service", "delivery", "shipping")),
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SYSTEM_PROMPT_LEAK = re.compile(
    r"(?i)\[(?:SYSTEM POLICY|LANGUAGE POLICY|GROUNDING POLICY|SECURITY POLICY|OUTPUT FORMAT)\]"
    r"|<retrieved_company_context>"
    r"|prompt_version\s*:"
)
_ESCALATION_CUES = {
    "escalation": EscalationReason.CUSTOMER_REQUESTED,
    "refund": EscalationReason.MONEY_OR_CLAIM,
    "returns": EscalationReason.MONEY_OR_CLAIM,
    "safety": EscalationReason.SAFETY,
    "account_related": EscalationReason.ACCOUNT_OR_PRIVACY,
}


def classify_intent(message: str) -> str:
    """Cheap, deterministic intent label used for routing and metrics.

    Not a replacement for the model's own understanding - it decides whether
    retrieval is *required*, whether to escalate immediately, and what to record
    in metrics.  Keyword-based on purpose: a classifier here would add a model
    call to every request's critical path.
    """
    lowered = message.lower()
    for intent, cues in _INTENT_CUES:
        if any(cue in message or cue in lowered for cue in cues):
            return intent
    if len(message.strip()) < 8:
        return "ambiguous"
    return "general_inquiry"


@dataclass(slots=True)
class InputVerdict:
    allowed: bool = True
    reason: str = ""
    message: str = ""
    intent: str = "general_inquiry"
    language: str = "km"
    injection_score: float = 0.0
    injection_matches: list[str] = field(default_factory=list)
    pii_found: dict[str, int] = field(default_factory=dict)
    escalate: EscalationReason = EscalationReason.NONE
    requires_grounding: bool = False

    def to_log(self) -> dict[str, Any]:
        """Loggable summary - never includes the message itself (§privacy)."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "intent": self.intent,
            "language": self.language,
            "message_chars": len(self.message),
            "injection_score": round(self.injection_score, 3),
            "injection_matches": self.injection_matches,
            "pii_kinds": sorted(self.pii_found),
            "requires_grounding": self.requires_grounding,
        }


@dataclass(slots=True)
class OutputVerdict:
    allowed: bool = True
    answer: str = ""
    reason: str = ""
    grounded: bool = True
    grounding_precision: float = 1.0
    escalate: EscalationReason = EscalationReason.NONE
    leaked_system_prompt: bool = False
    leaked_secret: bool = False
    corrupted_identifiers: list[str] = field(default_factory=list)
    unsupported_values: list[str] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "grounded": self.grounded,
            "grounding_precision": round(self.grounding_precision, 3),
            "escalate": str(self.escalate),
            "leaked_system_prompt": self.leaked_system_prompt,
            "leaked_secret": self.leaked_secret,
            "corrupted_identifiers": self.corrupted_identifiers,
            "unsupported_values": self.unsupported_values[:5],
        }


class InputGuard:
    """Validates and annotates a customer message before it reaches the model."""

    def __init__(
        self,
        *,
        max_chars: int = 4000,
        injection_block_threshold: float = 0.75,
        redact_pii: bool = True,
    ) -> None:
        self.max_chars = max_chars
        self.injection_block_threshold = injection_block_threshold
        self.redact_pii = redact_pii

    def check(self, message: str) -> InputVerdict:
        verdict = InputVerdict(message=message)

        if not message or not message.strip():
            return InputVerdict(allowed=False, reason="empty_message", message=message)

        cleaned = _CONTROL_CHARS.sub("", message)
        if len(cleaned) > self.max_chars:
            return InputVerdict(
                allowed=False,
                reason="message_too_long",
                message=cleaned[: self.max_chars],
            )
        verdict.message = cleaned

        language, _ = detect_language(cleaned, khmer_present=0.10)
        verdict.language = (
            "en" if language is TextLanguage.ENGLISH else "km"
        )

        verdict.intent = classify_intent(cleaned)
        verdict.requires_grounding = verdict.intent in GROUNDING_REQUIRED_INTENTS
        verdict.escalate = _ESCALATION_CUES.get(verdict.intent, EscalationReason.NONE)

        # A direct injection attempt is annotated, not necessarily blocked: the
        # system prompt handles most of them, and blocking outright teaches an
        # attacker exactly where the boundary is.  Only high-confidence attempts
        # are refused.
        scan = scan_for_injection(cleaned, block_threshold=self.injection_block_threshold)
        verdict.injection_score = scan.score
        verdict.injection_matches = [m.name for m in scan.matches]
        if scan.blocked:
            verdict.allowed = False
            verdict.reason = "prompt_injection"
            return verdict

        # The customer may legitimately include their own phone number; detect it
        # so it is never logged, but do not block the request.
        verdict.pii_found = redact(cleaned).counts
        return verdict


class OutputGuard:
    """Validates a generated answer before it is returned to the customer."""

    def __init__(
        self,
        *,
        min_grounding_precision: float = 0.90,
        enforce_grounding: bool = True,
        uncertainty_template_km: str | None = None,
    ) -> None:
        self.min_grounding_precision = min_grounding_precision
        self.enforce_grounding = enforce_grounding
        self.uncertainty_template_km = uncertainty_template_km or (
            "សូមអភ័យទោស ខ្ញុំមិនមានព័ត៌មានផ្លូវការគ្រប់គ្រាន់ដើម្បីឆ្លើយសំណួរនេះទេ។ "
            "សូមទាក់ទងផ្នែកបម្រើអតិថិជនរបស់យើង ដើម្បីទទួលបានចម្លើយត្រឹមត្រូវ។"
        )

    def check(
        self,
        answer: str,
        *,
        chunks: list[RetrievedChunk],
        requires_grounding: bool,
        intent: str = "general_inquiry",
    ) -> OutputVerdict:
        verdict = OutputVerdict(answer=answer)

        if not answer or not answer.strip():
            verdict.allowed = False
            verdict.reason = "empty_answer"
            verdict.answer = self.uncertainty_template_km
            verdict.escalate = EscalationReason.NO_INFORMATION
            return verdict

        # 1. System-prompt leakage.
        if _SYSTEM_PROMPT_LEAK.search(answer):
            verdict.leaked_system_prompt = True
            verdict.allowed = False
            verdict.reason = "system_prompt_leak"
            verdict.answer = (
                "ខ្ញុំមិនអាចចែករំលែកសេចក្តីណែនាំផ្ទៃក្នុងបានទេ "
                "ប៉ុន្តែខ្ញុំរីករាយជួយឆ្លើយសំណួរអំពីផលិតផល និងសេវាកម្មរបស់យើង។"
            )
            log.warning("guardrails.output.system_prompt_leak", extra={"intent": intent})
            return verdict

        # 2. Secret leakage.
        if scan_text(answer, path="<answer>"):
            verdict.leaked_secret = True
            verdict.allowed = False
            verdict.reason = "secret_leak"
            verdict.answer = self.uncertainty_template_km
            log.error("guardrails.output.secret_leak", extra={"intent": intent})
            return verdict

        # 3. Identifier corruption - a transliterated model number is a wrong answer.
        if chunks:
            context = "\n".join(c.text for c in chunks)
            ok, missing = verify_protected_spans(context, answer)
            if not ok:
                verdict.corrupted_identifiers = [s.text for s in missing]
                log.warning(
                    "guardrails.output.identifier_corruption",
                    extra={"identifiers": verdict.corrupted_identifiers},
                )

        # 4. Grounding.
        report: GroundingReport = verify_grounding(answer, chunks)
        verdict.grounded = report.is_grounded
        verdict.grounding_precision = report.grounding_precision
        verdict.unsupported_values = report.unsupported_values

        if requires_grounding and self.enforce_grounding:
            if not chunks and not report.hedged:
                verdict.allowed = False
                verdict.reason = "ungrounded_without_context"
                verdict.answer = self.uncertainty_template_km
                verdict.escalate = EscalationReason.NO_INFORMATION
                return verdict
            if report.grounding_precision < self.min_grounding_precision:
                verdict.allowed = False
                verdict.reason = "grounding_below_threshold"
                verdict.answer = self.uncertainty_template_km
                verdict.escalate = EscalationReason.NO_INFORMATION
                log.warning(
                    "guardrails.output.ungrounded",
                    extra={
                        "intent": intent,
                        "precision": round(report.grounding_precision, 3),
                        "unsupported": report.unsupported_values[:5],
                    },
                )
                return verdict

        if report.invalid_markers:
            # The model cited a source that does not exist; strip the markers
            # rather than discarding an otherwise correct answer.
            for marker in report.invalid_markers:
                verdict.answer = verdict.answer.replace(marker, "")
            verdict.answer = re.sub(r"\s{2,}", " ", verdict.answer).strip()
            verdict.reason = "invalid_citation_markers_stripped"

        return verdict


def known_intent(intent: str) -> str:
    """Coerce a label to the closed intent vocabulary used in metrics."""
    return intent if intent in INTENTS else "general_inquiry"
