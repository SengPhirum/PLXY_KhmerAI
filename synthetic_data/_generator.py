"""Shared generation plumbing for the synthetic-data scripts.

Two generation backends:

``LLMBackend``       calls a local Ollama model with the prompts in
                     ``prompts/dataset_generation.md``.  This is the production
                     path once a base model is available.
``TemplateBackend``  deterministic Khmer template expansion over real company
                     documents.  It needs no model, produces grounded output by
                     construction (every fact is copied from the source), and is
                     what makes the pipeline runnable and testable today.

Both emit the same ``SFTRecord`` schema and both set ``source_id`` so that
``synthetic_data/quality_check.py`` can verify grounding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.io import read_jsonl
from common.logging import get_logger
from preprocessing.language_mixing import SpanKind, extract_protected_spans

log = get_logger("synthetic.generator")

__all__ = [
    "Backend",
    "LLMBackend",
    "SourceDocument",
    "TemplateBackend",
    "build_backend",
    "load_documents",
    "make_record",
    "stable_sample_id",
]


@dataclass(slots=True)
class SourceDocument:
    document_id: str
    title: str
    text: str
    product_id: str = ""
    product_name: str = ""
    category: str = "general"
    version: str = ""
    effective_date: str = ""

    @property
    def display_product(self) -> str:
        return self.product_name or self.product_id or self.title

    def facts(self) -> list[str]:
        """Concrete values the document asserts - used to build grounded answers."""
        return [
            s.text
            for s in extract_protected_spans(self.text)
            if s.kind in (SpanKind.CURRENCY, SpanKind.MEASUREMENT, SpanKind.NUMBER)
        ]

    def sentences(self) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[។!?\.])\s+", self.text) if s.strip()]


def load_documents(path: str | Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for row in read_jsonl(path, skip_invalid=True):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        documents.append(
            SourceDocument(
                document_id=str(row.get("document_id") or row.get("id") or ""),
                title=str(row.get("document_title") or row.get("title") or ""),
                text=text,
                product_id=str(row.get("product_id") or ""),
                product_name=str(row.get("product_name") or ""),
                category=str(row.get("category") or "general"),
                version=str(row.get("version") or ""),
                effective_date=str(row.get("effective_date") or ""),
            )
        )
    return documents


def stable_sample_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


DEFAULT_SYSTEM = (
    "អ្នកគឺជាជំនួយការបម្រើអតិថិជនរបស់ក្រុមហ៊ុន។ "
    "សូមឆ្លើយជាភាសាខ្មែរ ខ្លី ច្បាស់ និងគួរសម។ "
    "ប្រើតែព័ត៌មានពីឯកសារក្រុមហ៊ុន។ បើមិនដឹង សូមប្រាប់ដោយស្មោះត្រង់។"
)


def make_record(
    *,
    sample_id: str,
    turns: list[tuple[str, str]],
    intent: str,
    source_id: str = "",
    source_type: str = "synthetic_template",
    synthetic: bool = True,
    system: str | None = DEFAULT_SYSTEM,
) -> dict[str, Any]:
    """Assemble an SFT record dict from ``(role, content)`` turns."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages += [{"role": role, "content": content} for role, content in turns]
    return {
        "messages": messages,
        "metadata": {
            "id": sample_id,
            "language": "km",
            "intent": intent,
            "source_type": source_type,
            "source_id": source_id,
            "quality_score": 1.0,
            "review_status": "unreviewed",
            "synthetic": synthetic,
        },
    }


class Backend(ABC):
    name = "abstract"

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 1024) -> str: ...


class TemplateBackend(Backend):
    """Deterministic template expansion - no model required, grounded by construction."""

    name = "template"

    def generate(self, prompt: str, *, max_tokens: int = 1024) -> str:  # pragma: no cover
        raise NotImplementedError(
            "TemplateBackend does not do free-form generation; the generator scripts "
            "call its dedicated builders instead."
        )


class LLMBackend(Backend):
    """Ollama-backed generation using the prompts in prompts/dataset_generation.md."""

    name = "llm"

    def __init__(
        self,
        model: str = "khmer-support-9b",
        *,
        base_url: str | None = None,
        temperature: float = 0.8,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url or os.environ.get("KHMERAI_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self._client: Any = None

    def generate(self, prompt: str, *, max_tokens: int = 1024) -> str:
        import httpx

        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # Higher temperature than serving: variety is the point here.
                "options": {
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        return (response.json().get("message") or {}).get("content", "")

    def generate_jsonl(self, prompt: str, *, max_tokens: int = 2048) -> list[dict[str, Any]]:
        """Generate and parse JSON Lines, skipping unparsable lines."""
        rows: list[dict[str, Any]] = []
        for line in self.generate(prompt, max_tokens=max_tokens).splitlines():
            stripped = line.strip().strip("`")
            if not stripped.startswith("{"):
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
        return rows


def build_backend(kind: str = "template", **kwargs: Any) -> Backend:
    if kind == "template":
        return TemplateBackend()
    if kind == "llm":
        return LLMBackend(**kwargs)
    raise ValueError(f"unknown generation backend {kind!r}; expected 'template' or 'llm'")
