"""Model performance benchmark: TTFT, tokens/sec, latency percentiles, memory.

Used for the Phase 5 baseline (4B vs 9B on Khmer), the Phase 12 quantization
comparison, and the Phase 13 Mac Studio tuning matrix.

Nothing here estimates: every number is measured from a real generation against
a running Ollama daemon.  With no daemon the script exits non-zero and prints
the command to start one, rather than emitting invented figures.

    python -m evaluation.benchmark_model --model khmer-support-9b --concurrency 1,5,10
    python -m evaluation.benchmark_model --compare khmer-support-4b,khmer-support-9b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common.io import write_json
from common.logging import get_logger
from common.paths import EVAL_REPORT_DIR, ensure_dir
from evaluation.metrics import khmer_fluency, latency_percentiles

log = get_logger(__name__)

__all__ = ["BenchmarkResult", "benchmark_model", "main", "DEFAULT_PROMPTS"]

# Deliberately covers every input shape from §Phase 12 "verification".
DEFAULT_PROMPTS: tuple[tuple[str, str], ...] = (
    ("khmer_short", "សួស្តី តើខ្ញុំអាចសួរអំពីផលិតផលបានទេ?"),
    ("khmer_question", "តើទូរទឹកកកម៉ូដែលថ្មីមានការធានារយៈពេលប៉ុន្មានឆ្នាំ?"),
    ("code_switch", "តើ model QN-4500A មាន warranty ប៉ុន្មានឆ្នាំ? សូមប្រាប់ price ផងដែរ។"),
    ("model_number", "សូមប្រាប់លក្ខណៈបច្ចេកទេសនៃម៉ូដែល RF-22B និង QN-4500A។"),
    ("numerals", "តម្លៃ ១២០ ដុល្លារ និង ៤៨០០០៛ ខុសគ្នាយ៉ាងណា? សូមគណនាជូន។"),
    (
        "khmer_long",
        "សូមពន្យល់លម្អិតអំពីរបៀបថែទាំទូរទឹកកក រួមទាំងការសម្អាត ការដោះទឹកកក "
        "ការពិនិត្យសីលរបស់ទ្វារ និងការកំណត់សីតុណ្ហភាពសមស្របសម្រាប់អាកាសធាតុក្តៅនៅកម្ពុជា។",
    ),
    (
        "multiturn",
        "ខ្ញុំបានប្រាប់អ្នកពីមុនថាខ្ញុំមានម៉ូដែល QN-4500A។ តើវាធានារហូតដល់ពេលណា?",
    ),
)


@dataclass(slots=True)
class SingleRun:
    prompt_id: str
    ok: bool
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    khmer_score: float = 0.0
    answer_chars: int = 0
    error: str = ""


@dataclass(slots=True)
class BenchmarkResult:
    model: str
    concurrency: int
    runs: list[SingleRun] = field(default_factory=list)
    wall_seconds: float = 0.0
    host: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def successes(self) -> list[SingleRun]:
        return [r for r in self.runs if r.ok]

    def summary(self) -> dict[str, Any]:
        ok = self.successes
        failures = [r for r in self.runs if not r.ok]
        latency = latency_percentiles([r.latency_ms for r in ok])
        ttft = latency_percentiles([r.ttft_ms for r in ok if r.ttft_ms])
        throughput = [r.tokens_per_second for r in ok if r.tokens_per_second]
        khmer = [r.khmer_score for r in ok]
        return {
            "model": self.model,
            "concurrency": self.concurrency,
            "requests": len(self.runs),
            "succeeded": len(ok),
            "failed": len(failures),
            "success_rate": round(len(ok) / len(self.runs), 4) if self.runs else 0.0,
            "wall_seconds": round(self.wall_seconds, 2),
            "requests_per_second": (
                round(len(ok) / self.wall_seconds, 3) if self.wall_seconds else 0.0
            ),
            "latency_ms": latency,
            "time_to_first_token_ms": ttft,
            "tokens_per_second_mean": (
                round(statistics.fmean(throughput), 2) if throughput else 0.0
            ),
            "total_completion_tokens": sum(r.completion_tokens for r in ok),
            "khmer_quality_mean": round(statistics.fmean(khmer), 4) if khmer else 0.0,
            "errors": [r.error for r in failures][:10],
            "options": self.options,
            "host": self.host,
        }


async def _one_request(
    client: Any, model: str, prompt_id: str, prompt: str, options: dict[str, Any]
) -> SingleRun:
    started = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    metrics: dict[str, Any] = {}
    try:
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "options": options,
            },
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")[:160]
                return SingleRun(prompt_id, False, error=f"HTTP {response.status_code}: {body}")
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                piece = (event.get("message") or {}).get("content", "")
                if piece and first_token_at is None:
                    first_token_at = time.perf_counter()
                if piece:
                    pieces.append(piece)
                if event.get("done"):
                    metrics = event
    except Exception as exc:  # noqa: BLE001 - a failed run is data, not a crash
        return SingleRun(prompt_id, False, error=f"{type(exc).__name__}: {exc}")

    latency_ms = (time.perf_counter() - started) * 1000
    answer = "".join(pieces)
    completion_tokens = int(metrics.get("eval_count", 0) or 0)
    eval_ns = float(metrics.get("eval_duration", 0) or 0)
    return SingleRun(
        prompt_id=prompt_id,
        ok=bool(answer.strip()),
        latency_ms=latency_ms,
        ttft_ms=((first_token_at - started) * 1000) if first_token_at else 0.0,
        prompt_tokens=int(metrics.get("prompt_eval_count", 0) or 0),
        completion_tokens=completion_tokens,
        tokens_per_second=(
            round(completion_tokens / (eval_ns / 1e9), 2) if eval_ns and completion_tokens else 0.0
        ),
        khmer_score=khmer_fluency(answer)["score"] if answer.strip() else 0.0,
        answer_chars=len(answer),
        error="" if answer.strip() else "empty response",
    )


async def _run(
    model: str,
    *,
    base_url: str,
    concurrency: int,
    repeats: int,
    options: dict[str, Any],
    timeout: float,
) -> BenchmarkResult:
    import httpx  # noqa: PLC0415

    prompts = [(pid, text) for pid, text in DEFAULT_PROMPTS for _ in range(repeats)]
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        async def guarded(pid: str, text: str) -> SingleRun:
            async with semaphore:
                return await _one_request(client, model, pid, text, options)

        started = time.perf_counter()
        runs = await asyncio.gather(*(guarded(pid, text) for pid, text in prompts))
        wall = time.perf_counter() - started

    return BenchmarkResult(
        model=model,
        concurrency=concurrency,
        runs=list(runs),
        wall_seconds=wall,
        options=options,
    )


def _host_stats() -> dict[str, Any]:
    from server.health import host_stats

    return host_stats()


def benchmark_model(
    model: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    concurrency_levels: tuple[int, ...] = (1,),
    repeats: int = 1,
    options: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Measure a model at each concurrency level.  Returns a report dict."""
    resolved_options = options or {"temperature": 0.3, "top_p": 0.9, "num_predict": 256}
    before = _host_stats()
    levels: list[dict[str, Any]] = []
    for concurrency in concurrency_levels:
        log.info("benchmark.level.start", extra={"model": model, "concurrency": concurrency})
        result = asyncio.run(
            _run(
                model,
                base_url=base_url,
                concurrency=concurrency,
                repeats=repeats,
                options=resolved_options,
                timeout=timeout,
            )
        )
        result.host = _host_stats()
        levels.append(result.summary())

    return {
        "model": model,
        "base_url": base_url,
        "options": resolved_options,
        "prompt_set": [pid for pid, _ in DEFAULT_PROMPTS],
        "repeats_per_prompt": repeats,
        "host_before": before,
        "levels": levels,
        "measured": True,
        "note": (
            "All values are measured from real generations against a running Ollama "
            "daemon. Peak unified memory must be read separately from Activity Monitor "
            "or `ollama ps` - see docs/deployment_guide.md."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.benchmark_model")
    parser.add_argument("--model", default="khmer-support-9b")
    parser.add_argument("--compare", default=None, help="comma-separated models to compare")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--concurrency", default="1", help="comma-separated levels, e.g. 1,5,10")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    levels = tuple(int(x) for x in args.concurrency.split(",") if x.strip())
    models = [m.strip() for m in (args.compare or args.model).split(",") if m.strip()]
    options = {
        "temperature": args.temperature,
        "top_p": 0.9,
        "num_predict": args.num_predict,
        "num_ctx": args.num_ctx,
    }

    reports = []
    for model in models:
        report = benchmark_model(
            model,
            base_url=args.base_url,
            concurrency_levels=levels,
            repeats=args.repeats,
            options=options,
        )
        reports.append(report)

    failed_everything = all(
        all(level["succeeded"] == 0 for level in r["levels"]) for r in reports
    )
    payload: dict[str, Any] = {"reports": reports}
    if len(reports) > 1:
        payload["comparison"] = _decision_matrix(reports)

    output = Path(args.output) if args.output else ensure_dir(EVAL_REPORT_DIR) / "model_benchmark.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if failed_everything:
        print(
            "\nNo generation succeeded. Is Ollama running?\n"
            "    bash ollama/start_server.sh\n"
            "    bash ollama/create_model.sh",
            file=sys.stderr,
        )
        return 1
    return 0


def _decision_matrix(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """§29 model-selection matrix, populated only with measured values."""
    rows: dict[str, dict[str, Any]] = {}
    for report in reports:
        model = report["model"]
        top = report["levels"][0] if report["levels"] else {}
        highest = report["levels"][-1] if report["levels"] else {}
        rows[model] = {
            "ttft_p95_ms": (top.get("time_to_first_token_ms") or {}).get("p95", 0.0),
            "tokens_per_second": top.get("tokens_per_second_mean", 0.0),
            "latency_p95_ms_at_max_concurrency": (highest.get("latency_ms") or {}).get("p95", 0.0),
            "max_concurrency_tested": highest.get("concurrency", 0),
            "success_rate_at_max_concurrency": highest.get("success_rate", 0.0),
            "khmer_quality_screen": top.get("khmer_quality_mean", 0.0),
        }
    return {
        "matrix": rows,
        "note": (
            "Khmer quality, support accuracy, hallucination and RAG grounding come from "
            "the evaluation suite, not from this benchmark. Fill those rows from "
            "evaluation/reports/ before making a model decision (§29)."
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
