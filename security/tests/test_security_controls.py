"""Redaction, secret scanning, access control, logging hygiene, prompt contract."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from common.logging import JsonFormatter, bind_request
from common.paths import PROJECT_ROOT, resolve_under_root
from security.access_control import (
    ADMIN,
    AGENT,
    CUSTOMER,
    Principal,
    can_quote,
    can_retrieve,
    can_see_field,
    filter_metadata,
)
from security.data_redaction import Redactor, redact, redact_khmer_aware
from security.secret_scanner import scan_repository, scan_text


# --- redaction --------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("contact me at someone@example.com", "email"),
        ("call 012 345 678 please", "cambodia_phone_local"),
        ("call +855 12 345 678", "cambodia_phone_intl"),
        # A branded key is classified by its brand rule, which is more precise
        # than the generic assignment rule.
        ('api_key="sk-abcdefghijklmnopqrst1234"', "openai_key"),  # pragma: allowlist secret
        ('db_password = "n0tAplaceh0lderValue"', "assigned_secret"),  # pragma: allowlist secret
        ("token AKIAIOSFODNN7EXAMPLE here", "aws_access_key"),  # pragma: allowlist secret
        ("server at 192.168.1.50", "ip_address"),
    ],
)
def test_pii_and_secrets_are_redacted(text: str, kind: str) -> None:
    result = redact(text)
    assert kind in result.counts, f"{kind} not detected in {text!r}"
    assert f"[REDACTED:{kind}]" in result.text


def test_khmer_digit_phone_numbers_are_redacted() -> None:
    result = redact_khmer_aware("ទូរស័ព្ទ ០១២ ៣៤៥ ៦៧៨")
    assert result.redacted


def test_ordinary_khmer_text_is_untouched() -> None:
    text = "ការធានារយៈពេល ២៤ ខែ ចាប់ពីថ្ងៃទិញ។"
    assert redact(text).text == text


def test_long_digit_runs_are_not_mistaken_for_card_numbers() -> None:
    """The Luhn guard stops a serial number being redacted as a credit card."""
    result = redact("serial 1234567890123456789")
    assert "credit_card" not in result.counts


def test_a_real_card_number_is_redacted() -> None:
    result = redact("card 4539578763621486")  # passes Luhn
    assert "credit_card" in result.counts


def test_keep_kinds_preserves_the_company_contact() -> None:
    redactor = Redactor(keep_kinds=frozenset({"email"}))
    result = redactor.redact("support@company.com and someone@example.com")
    assert "support@company.com" in result.text
    assert "email" not in result.counts


# --- secret scanner ---------------------------------------------------------
@pytest.mark.parametrize(
    "line",
    [
        "AWS_KEY = 'AKIAIOSFODNN7EXAMPLE'",  # pragma: allowlist secret
        "github_token = 'ghp_" + "a" * 40 + "'",
        'password = "sup3rS3cretV4lue!"',  # pragma: allowlist secret
        "-----BEGIN RSA PRIVATE KEY-----",  # pragma: allowlist secret
        "DATABASE_URL=postgres://user:hunter2pass@db:5432/app",  # pragma: allowlist secret
    ],
)
def test_secrets_are_detected(line: str) -> None:
    assert scan_text(line), f"missed: {line!r}"


@pytest.mark.parametrize(
    "line",
    [
        "KHMERAI_ADMIN_API_KEY=CHANGE_ME_ADMIN_KEY",
        'api_key = "your-api-key-here"',
        "password = os.environ['DB_PASSWORD']",
        "KHMERAI_QDRANT_API_KEY=${KHMERAI_QDRANT_API_KEY}",
        "ការធានារយៈពេល ២៤ ខែ",
    ],
)
def test_placeholders_are_not_flagged(line: str) -> None:
    assert not scan_text(line), f"false positive: {line!r}"


def test_findings_mask_the_secret_value() -> None:
    findings = scan_text("github_token = 'ghp_" + "a" * 40 + "'")
    assert findings
    assert "a" * 40 not in str(findings[0])
    assert "*" in findings[0].excerpt


def test_the_repository_contains_no_committed_secrets() -> None:
    """Release gate: nothing tracked in the repo may contain a credential."""
    findings = [
        f
        for f in scan_repository(PROJECT_ROOT)
        # The security module and its tests contain deliberate example patterns.
        if "security/" not in f.path and "/tests/" not in f.path
    ]
    critical = [f for f in findings if f.severity in ("critical", "high")]
    assert not critical, "\n".join(str(f) for f in critical)


# --- logging hygiene --------------------------------------------------------
def test_secrets_are_redacted_from_log_records() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='connecting with api_key="sk-abcdefghijklmnop1234567890"',  # pragma: allowlist secret
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "sk-abcdefghijklmnop1234567890" not in rendered  # pragma: allowlist secret
    assert "REDACTED" in rendered


def test_pii_in_log_extras_is_redacted() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="customer contact",
        args=(),
        exc_info=None,
    )
    record.customer_email = "someone@example.com"  # type: ignore[attr-defined]
    rendered = formatter.format(record)
    assert "someone@example.com" not in rendered


def test_log_records_carry_the_request_id() -> None:
    formatter = JsonFormatter()
    with bind_request("req-abc123"):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        payload = json.loads(formatter.format(record))
    assert payload["request_id"] == "req-abc123"


# --- access control ---------------------------------------------------------
PUBLIC_DOC = {"confidentiality": "public", "status": "active"}
INTERNAL_DOC = {"confidentiality": "internal", "status": "active"}
RESTRICTED_DOC = {"confidentiality": "restricted", "status": "active"}
EXPIRED_DOC = {"confidentiality": "public", "status": "expired"}
QUARANTINED_DOC = {
    "confidentiality": "public",
    "status": "active",
    "validation_status": "quarantined",
}


def test_customers_only_reach_public_documents() -> None:
    assert can_retrieve(CUSTOMER, PUBLIC_DOC)
    assert not can_retrieve(CUSTOMER, INTERNAL_DOC)
    assert not can_retrieve(CUSTOMER, RESTRICTED_DOC)


def test_agents_reach_internal_but_not_restricted() -> None:
    assert can_retrieve(AGENT, INTERNAL_DOC)
    assert not can_retrieve(AGENT, RESTRICTED_DOC)
    assert can_retrieve(ADMIN, RESTRICTED_DOC)


def test_internal_documents_are_never_quotable_to_a_customer() -> None:
    """An agent may read an internal doc but must not quote it to a customer."""
    assert can_retrieve(AGENT, INTERNAL_DOC)
    assert not can_quote(CUSTOMER, INTERNAL_DOC)


def test_expired_documents_are_excluded_by_default() -> None:
    assert not can_retrieve(CUSTOMER, EXPIRED_DOC)
    assert can_retrieve(CUSTOMER, EXPIRED_DOC, allow_non_active=True)


def test_quarantined_documents_are_never_retrievable() -> None:
    assert not can_retrieve(ADMIN, QUARANTINED_DOC)


def test_metadata_fields_are_filtered_by_principal() -> None:
    metadata = {
        "document_id": "d1",
        "document_title": "ធានា",
        "source_path": "/company/internal/warranty.md",
        "owner": "after-sales",
        "content_hash": "abc",
        "status": "active",
    }
    customer_view = filter_metadata(CUSTOMER, metadata)
    assert "source_path" not in customer_view
    assert "owner" not in customer_view
    assert customer_view["document_title"] == "ធានា"
    assert "source_path" in filter_metadata(ADMIN, metadata)


def test_anonymous_principal_is_the_default() -> None:
    principal = Principal()
    assert can_retrieve(principal, PUBLIC_DOC)
    assert not can_retrieve(principal, INTERNAL_DOC)
    assert not can_see_field(principal, "source_path")


# --- path traversal ---------------------------------------------------------
@pytest.mark.parametrize(
    "candidate",
    ["../../etc/passwd", "/etc/passwd", "data/../../etc/shadow", "~/../../root/.ssh/id_rsa"],
)
def test_path_traversal_is_blocked(candidate: str) -> None:
    with pytest.raises(ValueError, match="escapes"):
        resolve_under_root(candidate, PROJECT_ROOT / "data")


def test_legitimate_paths_resolve() -> None:
    resolved = resolve_under_root("interim/company_records.jsonl", PROJECT_ROOT / "data")
    assert str(resolved).endswith("data/interim/company_records.jsonl")


# --- prompt contract --------------------------------------------------------
REQUIRED_SECTIONS = (
    "[SYSTEM POLICY]",
    "[LANGUAGE POLICY]",
    "[CUSTOMER-SERVICE POLICY]",
    "[GROUNDING POLICY]",
    "[SECURITY POLICY]",
    "[ESCALATION POLICY]",
    "[OUTPUT FORMAT]",
)


@pytest.mark.parametrize("prompt_file", ["system_km.md", "system_en.md"])
def test_system_prompts_contain_every_required_section(prompt_file: str) -> None:
    text = (PROJECT_ROOT / "prompts" / prompt_file).read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"{prompt_file} is missing {section}"


def test_system_prompts_state_the_untrusted_context_rule() -> None:
    for name in ("system_km.md", "system_en.md"):
        text = (PROJECT_ROOT / "prompts" / name).read_text(encoding="utf-8")
        assert "retrieved_company_context" in text
        assert ("ទិន្នន័យ" in text) or ("data, not instructions" in text)


def test_khmer_prompt_mandates_khmer_replies() -> None:
    text = (PROJECT_ROOT / "prompts" / "system_km.md").read_text(encoding="utf-8")
    assert "ភាសាខ្មែរ" in text
    assert "ហាមបង្កើតការពិតអំពីក្រុមហ៊ុន" in text


def test_prompt_files_have_no_placeholder_left_unfilled() -> None:
    """Every `{{placeholder}}` must be one the PromptBuilder actually fills."""
    known = {
        "company_name",
        "conversation_summary",
        "retrieved_context",
        "user_message",
        "detected_product",
        "today",
        "escalation_reason",
        "hotline",
        "support_email",
        "business_hours",
        "topic",
    }
    import re

    for path in (PROJECT_ROOT / "prompts").glob("system_*.md"):
        for match in re.finditer(r"\{\{(\w+)\}\}", path.read_text(encoding="utf-8")):
            assert match.group(1) in known, f"{path.name}: unknown placeholder {match.group(0)}"


def test_escalation_prompt_forbids_inventing_contact_details() -> None:
    text = (PROJECT_ROOT / "prompts" / "escalation.md").read_text(encoding="utf-8")
    assert "ហាមបង្កើតលេខទូរស័ព្ទ" in text


# --- file-type allow list ---------------------------------------------------
def test_only_allowed_document_types_are_ingestible(tmp_path: Path) -> None:
    from company_data.loaders import SUPPORTED_EXTENSIONS, UnsupportedFormatError, load_any

    for suffix in (".exe", ".sh", ".so", ".zip", ".lnk", ".js"):
        assert suffix not in SUPPORTED_EXTENSIONS
        bad = tmp_path / f"payload{suffix}"
        bad.write_text("payload", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            load_any(bad)
