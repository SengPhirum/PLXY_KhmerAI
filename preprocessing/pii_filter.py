"""PII handling for corpora and for the pre-cloud-upload gate (§36).

Two modes:

``redact``  replace PII in place and keep the record.  Used for web corpora,
            where discarding every document containing an email address would
            throw away most of the usable Khmer text.
``drop``    discard the whole record.  Used for anything derived from customer
            conversations, and for every dataset that is about to leave the
            premises for Colab training.

``build_pre_upload_report`` produces the artefact the specification requires
before any dataset is uploaded to a cloud trainer: counts by category, the
files involved, and a hard ``approved`` boolean that ``training/*`` refuses to
run without.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from common.hashing import sha256_file
from security.data_redaction import Redactor, normalise_khmer_digits

__all__ = ["PiiPolicy", "PiiFilter", "PiiReport", "build_pre_upload_report"]

Mode = Literal["redact", "drop", "report_only"]

# Categories that make a dataset unfit to leave the premises, regardless of mode.
BLOCKING_CATEGORIES = frozenset(
    {
        "aws_access_key",
        "github_token",
        "slack_token",
        "openai_key",
        "hf_token",
        "jwt",
        "private_key_block",
        "assigned_secret",
        "credit_card",
    }
)


@dataclass(slots=True)
class PiiPolicy:
    mode: Mode = "redact"
    keep_kinds: frozenset[str] = frozenset()
    allowlist: frozenset[str] = frozenset()
    khmer_digit_aware: bool = True
    drop_if_categories: frozenset[str] = BLOCKING_CATEGORIES

    @classmethod
    def for_public_corpus(cls) -> PiiPolicy:
        return cls(mode="redact")

    @classmethod
    def for_cloud_upload(cls) -> PiiPolicy:
        """Nothing sensitive may reach Colab - drop instead of redacting."""
        return cls(mode="drop", drop_if_categories=BLOCKING_CATEGORIES | {"email", "credit_card"})

    @classmethod
    def for_company_public_docs(cls) -> PiiPolicy:
        """The company's own hotline and support address must survive ingestion."""
        return cls(
            mode="redact",
            keep_kinds=frozenset({"cambodia_phone_local", "cambodia_phone_intl", "email"}),
        )


@dataclass(slots=True)
class PiiReport:
    records_scanned: int = 0
    records_redacted: int = 0
    records_dropped: int = 0
    findings: dict[str, int] = field(default_factory=dict)
    blocking_findings: dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.blocking_findings

    def merge(self, counts: dict[str, int]) -> None:
        for kind, n in counts.items():
            self.findings[kind] = self.findings.get(kind, 0) + n

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_scanned": self.records_scanned,
            "records_redacted": self.records_redacted,
            "records_dropped": self.records_dropped,
            "findings": dict(sorted(self.findings.items(), key=lambda kv: -kv[1])),
            "blocking_findings": dict(self.blocking_findings),
            "clean": self.clean,
        }


class PiiFilter:
    """Apply a :class:`PiiPolicy` to text or to a record stream."""

    def __init__(self, policy: PiiPolicy | None = None) -> None:
        self.policy = policy or PiiPolicy()
        self._redactor = Redactor(
            keep_kinds=self.policy.keep_kinds, allowlist=self.policy.allowlist
        )
        self.report = PiiReport()

    def process_text(self, text: str) -> tuple[str | None, dict[str, int]]:
        """Return ``(text_or_None, findings)``.  ``None`` means "drop this record"."""
        self.report.records_scanned += 1
        candidate = normalise_khmer_digits(text) if self.policy.khmer_digit_aware else text
        result = self._redactor.redact(candidate)
        counts = result.counts
        self.report.merge(counts)

        blocking = {k: v for k, v in counts.items() if k in self.policy.drop_if_categories}
        for kind, n in blocking.items():
            self.report.blocking_findings[kind] = self.report.blocking_findings.get(kind, 0) + n

        if self.policy.mode == "report_only":
            return text, counts
        if self.policy.mode == "drop" and counts:
            self.report.records_dropped += 1
            return None, counts
        if blocking and self.policy.mode == "redact":
            # Credentials are never merely redacted in a corpus - drop the record,
            # because surrounding context usually leaks the secret too.
            self.report.records_dropped += 1
            return None, counts
        if counts:
            self.report.records_redacted += 1
            return result.text, counts
        return text, counts

    def process_records(
        self, records: Iterable[dict[str, Any]], *, text_key: str = "text"
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for record in records:
            cleaned, counts = self.process_text(str(record.get(text_key, "")))
            if cleaned is None:
                continue
            updated = dict(record)
            updated[text_key] = cleaned
            if counts:
                metadata = dict(updated.get("metadata") or {})
                metadata["pii_redactions"] = counts
                updated["metadata"] = metadata
            out.append(updated)
        return out


def build_pre_upload_report(
    paths: Iterable[str | Path],
    *,
    policy: PiiPolicy | None = None,
    text_key: str = "text",
    max_records_per_file: int | None = None,
) -> dict[str, Any]:
    """Scan JSONL datasets destined for cloud training.

    The returned dict is written to ``data/manifests/pre_upload_report.json`` and
    is checked by ``training/common.py`` before any dataset is read on Colab.
    """
    from common.io import read_jsonl  # noqa: PLC0415 - avoids a circular import at module load

    filter_ = PiiFilter(policy or PiiPolicy.for_cloud_upload())
    files: list[dict[str, Any]] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            files.append({"path": str(path), "error": "not_found"})
            continue
        file_findings: dict[str, int] = {}
        scanned = 0
        for record in read_jsonl(path, skip_invalid=True):
            if max_records_per_file is not None and scanned >= max_records_per_file:
                break
            scanned += 1
            text = record.get(text_key)
            if text is None and isinstance(record.get("messages"), list):
                text = "\n".join(str(m.get("content", "")) for m in record["messages"])
            _, counts = filter_.process_text(str(text or ""))
            for kind, n in counts.items():
                file_findings[kind] = file_findings.get(kind, 0) + n
        files.append(
            {
                "path": str(path),
                "sha256": sha256_file(str(path)),
                "records_scanned": scanned,
                "findings": file_findings,
                "clean": not (set(file_findings) & filter_.policy.drop_if_categories),
            }
        )

    report = filter_.report.to_dict()
    report["files"] = files
    report["policy"] = {
        "mode": filter_.policy.mode,
        "drop_if_categories": sorted(filter_.policy.drop_if_categories),
    }
    report["approved"] = report["clean"] and all(f.get("clean", False) for f in files)
    report["note"] = (
        "approved=false blocks cloud training. Resolve every blocking finding, or record a "
        "written data-handling approval in docs/security.md before overriding."
    )
    return report
