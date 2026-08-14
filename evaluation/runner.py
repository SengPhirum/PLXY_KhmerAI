"""Shared plumbing for every evaluator: golden-set loading and model calling.

Two model backends:

``OllamaRunner``  the real thing - talks to a local Ollama daemon.  Requires a
                  running model, so evaluators using it are release gates, not
                  CI jobs.
``ApiRunner``     talks to the FastAPI service, which means the answer has gone
                  through retrieval, prompting and guardrails.  This is the one
                  that measures the *product*, and it is what
                  ``scripts/evaluate_all.sh`` uses.
``StaticRunner``  replays recorded answers from a JSONL file, so a report can be
                  regenerated without a model and so the evaluators themselves
                  are testable in CI.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from common.io import read_jsonl, write_json, atomic_write_text
from common.logging import get_logger
from common.paths import EVAL_GOLDEN_DIR, EVAL_REPORT_DIR, ensure_dir
from common.versions import git_commit, load_versions
from evaluation.schemas import EvalReport, GoldenItem, ModelAnswer

log = get_logger(__name__)

__all__ = [
    "Runner",
    "OllamaRunner",
    "ApiRunner",
    "StaticRunner",
    "load_golden",
    "write_report",
    "build_runner",
]


def load_golden(path: str | Path) -> list[GoldenItem]:
    """Load a golden JSONL file, skipping (and reporting) malformed rows."""
    target = Path(path)
    if not target.is_absolute() and not target.exists():
        target = EVAL_GOLDEN_DIR / target.name
    if not target.is_file():
        raise FileNotFoundError(f"golden set not found: {path}")

    items: list[GoldenItem] = []
    for index, row in enumerate(read_jsonl(target, skip_invalid=True), start=1):
        try:
            row.setdefault("id", f"{target.stem}-{index:04d}")
            items.append(GoldenItem.model_validate(row))
        except Exception as exc:  # noqa: BLE001 - a bad row is reported, not fatal
            log.error(
                "evaluation.golden.invalid_row",
                extra={"file": str(target), "row": index, "error": str(exc)},
            )
    if not items:
        raise ValueError(f"no valid items in {target}")
    return items


class Runner(ABC):
    """Produces a :class:`ModelAnswer` for a golden item."""

    name = "abstract"

    @abstractmethod
    def answer(self, item: GoldenItem) -> ModelAnswer: ...

    def run_all(self, items: list[GoldenItem]) -> Iterator[ModelAnswer]:
        for index, item in enumerate(items, start=1):
            if index % 25 == 0:
                log.info("evaluation.progress", extra={"done": index, "total": len(items)})
            yield self.answer(item)

    def close(self) -> None:  # pragma: no cover - default no-op
        return


class OllamaRunner(Runner):
    """Direct model calls, no retrieval - used for the Phase 5 baseline."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        system_prompt: str = "",
        options: dict[str, Any] | None = None,
        timeout: float = 180.0,
    ) -> None:
        import httpx  # noqa: PLC0415

        self.model = model
        self.system_prompt = system_prompt
        self.options = options or {"temperature": 0.3, "top_p": 0.9, "num_predict": 768}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def answer(self, item: GoldenItem) -> ModelAnswer:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(item.turns or [])
        messages.append({"role": "user", "content": item.question})

        started = time.perf_counter()
        try:
            response = self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": self.options,
                },
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001 - one failed item must not stop the run
            return ModelAnswer(
                item_id=item.id,
                answer="",
                model=self.model,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        return ModelAnswer(
            item_id=item.id,
            answer=(body.get("message") or {}).get("content", ""),
            model=body.get("model", self.model),
            latency_ms=(time.perf_counter() - started) * 1000,
            time_to_first_token_ms=float(body.get("prompt_eval_duration", 0) or 0) / 1e6,
            prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(body.get("eval_count", 0) or 0),
        )

    def close(self) -> None:
        self._client.close()


class ApiRunner(Runner):
    """Calls the deployed FastAPI service, so RAG and guardrails are included."""

    name = "api"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        api_key: str = "",
        timeout: float = 180.0,
    ) -> None:
        import httpx  # noqa: PLC0415

        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )

    def answer(self, item: GoldenItem) -> ModelAnswer:
        conversation_id = f"eval-{item.id}"[:60]
        started = time.perf_counter()
        try:
            # Replay prior turns so multi-turn items are evaluated in context.
            for turn in item.turns:
                if turn.get("role") == "user":
                    self._client.post(
                        "/v1/chat",
                        json={
                            "message": turn["content"],
                            "conversation_id": conversation_id,
                            "stream": False,
                        },
                    )
            response = self._client.post(
                "/v1/chat",
                json={
                    "message": item.question,
                    "conversation_id": conversation_id,
                    "product_id": item.product_id,
                    "stream": False,
                },
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            return ModelAnswer(
                item_id=item.id,
                answer="",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        usage = body.get("usage", {}) or {}
        return ModelAnswer(
            item_id=item.id,
            answer=body.get("answer", ""),
            sources=[s.get("document_id", "") for s in body.get("sources", [])],
            model=body.get("model", ""),
            escalation_required=bool(body.get("escalation_required")),
            latency_ms=(time.perf_counter() - started) * 1000,
            time_to_first_token_ms=float(usage.get("time_to_first_token_ms", 0) or 0),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    def close(self) -> None:
        self._client.close()


class StaticRunner(Runner):
    """Replays recorded answers, keyed by item id.  No model required."""

    name = "static"

    def __init__(self, answers: dict[str, str] | str | Path, *, model: str = "recorded") -> None:
        self.model = model
        if isinstance(answers, dict):
            self._answers = dict(answers)
        else:
            self._answers = {
                str(row["item_id"]): str(row.get("answer", ""))
                for row in read_jsonl(answers, skip_invalid=True)
                if "item_id" in row
            }

    def answer(self, item: GoldenItem) -> ModelAnswer:
        return ModelAnswer(
            item_id=item.id,
            answer=self._answers.get(item.id, ""),
            model=self.model,
            error="" if item.id in self._answers else "no recorded answer",
        )


def build_runner(
    backend: str,
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    answers: str | Path | None = None,
    system_prompt: str = "",
) -> Runner:
    kind = backend.lower()
    if kind == "ollama":
        return OllamaRunner(
            model or "khmer-support-9b",
            base_url=base_url or "http://127.0.0.1:11434",
            system_prompt=system_prompt,
        )
    if kind == "api":
        return ApiRunner(base_url or "http://127.0.0.1:8000", api_key=api_key)
    if kind == "static":
        if answers is None:
            raise ValueError("the static runner requires --answers")
        return StaticRunner(answers, model=model or "recorded")
    raise ValueError(f"unknown runner backend {kind!r}; expected ollama, api or static")


def write_report(report: EvalReport, name: str, *, directory: Path | None = None) -> tuple[Path, Path]:
    """Write ``<name>.json`` and ``<name>.md``.  Returns both paths."""
    target = ensure_dir(directory or EVAL_REPORT_DIR)
    json_path = target / f"{name}.json"
    markdown_path = target / f"{name}.md"
    write_json(json_path, json.loads(report.model_dump_json()))
    atomic_write_text(markdown_path, report.to_markdown())
    log.info(
        "evaluation.report.written",
        extra={"json": str(json_path), "markdown": str(markdown_path), "passed": report.passed},
    )
    return json_path, markdown_path


def stamp_report(report: EvalReport) -> EvalReport:
    """Attach provenance so a report can always be tied to what produced it."""
    versions = load_versions()
    report.code_commit = git_commit()
    report.prompt_version = versions.prompt
    report.index_version = versions.knowledge_index
    report.dataset_version = versions.dataset
    return report
