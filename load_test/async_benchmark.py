#!/usr/bin/env python3
"""Async load test against the FastAPI service (Phase 18).

    python load_test/async_benchmark.py --clients 10 --duration 60
    python load_test/async_benchmark.py --pattern ramp --clients 20 --duration 120
    python load_test/async_benchmark.py --pattern burst --clients 20

Traffic mixture (§Phase 18): 50% short support questions, 25% RAG-heavy,
15% follow-up conversations, 5% long questions, 5% malformed/adversarial.

Patterns: ``constant``, ``ramp``, ``burst``, ``sustained``.

Measures success rate, latency percentiles, TTFT on the streaming path, stream
stability, tokens/sec, queue behaviour (503s) and error taxonomy.  Every number
is measured; nothing is estimated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.io import read_jsonl, write_json  # noqa: E402
from common.logging import get_logger  # noqa: E402
from evaluation.metrics import latency_percentiles  # noqa: E402

log = get_logger("load_test")

SCENARIO_DIR = _REPO_ROOT / "load_test" / "scenarios"
MIXTURE = (
    ("short_questions.jsonl", 0.50, False),
    ("rag_questions.jsonl", 0.25, True),
    ("multiturn.jsonl", 0.15, True),
    ("long_questions.jsonl", 0.05, True),
    ("adversarial.jsonl", 0.05, False),
)


@dataclass(slots=True)
class Sample:
    ok: bool
    status: int
    latency_ms: float
    ttft_ms: float = 0.0
    tokens: int = 0
    scenario: str = ""
    error: str = ""
    stream_complete: bool = True


@dataclass(slots=True)
class Scenario:
    name: str
    weight: float
    stream: bool
    prompts: list[dict[str, Any]] = field(default_factory=list)


def load_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for filename, weight, stream in MIXTURE:
        path = SCENARIO_DIR / filename
        prompts = list(read_jsonl(path, skip_invalid=True)) if path.is_file() else []
        if not prompts:
            log.warning("load_test.scenario_missing", extra={"file": str(path)})
            continue
        scenarios.append(Scenario(name=path.stem, weight=weight, stream=stream, prompts=prompts))
    if not scenarios:
        raise FileNotFoundError(f"no scenario files under {SCENARIO_DIR}")
    return scenarios


def _pick(scenarios: list[Scenario], rng: random.Random) -> tuple[Scenario, dict[str, Any]]:
    total = sum(s.weight for s in scenarios)
    target = rng.random() * total
    cumulative = 0.0
    for scenario in scenarios:
        cumulative += scenario.weight
        if target <= cumulative:
            return scenario, rng.choice(scenario.prompts)
    return scenarios[-1], rng.choice(scenarios[-1].prompts)


async def _one_request(
    client: Any, base_url: str, scenario: Scenario, prompt: dict[str, Any], client_id: int
) -> Sample:
    payload = {
        "message": prompt.get("message", ""),
        "conversation_id": f"load-{client_id}-{prompt.get('conversation', 'x')}"[:60],
        "stream": scenario.stream,
    }
    if prompt.get("product_id"):
        payload["product_id"] = prompt["product_id"]

    started = time.perf_counter()
    try:
        if scenario.stream:
            first_token_at: float | None = None
            tokens = 0
            saw_done = False
            async with client.stream("POST", "/v1/chat/stream", json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    return Sample(
                        False,
                        response.status_code,
                        (time.perf_counter() - started) * 1000,
                        scenario=scenario.name,
                        error=f"HTTP {response.status_code}",
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event.get("type") == "token":
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        tokens += 1
                    elif event.get("type") == "done":
                        saw_done = True
                    elif event.get("type") == "error":
                        return Sample(
                            False,
                            200,
                            (time.perf_counter() - started) * 1000,
                            scenario=scenario.name,
                            error=str(event.get("error", "stream error")),
                        )
            elapsed = (time.perf_counter() - started) * 1000
            return Sample(
                ok=saw_done,
                status=200,
                latency_ms=elapsed,
                ttft_ms=((first_token_at - started) * 1000) if first_token_at else 0.0,
                tokens=tokens,
                scenario=scenario.name,
                stream_complete=saw_done,
                error="" if saw_done else "stream ended without a done event",
            )

        response = await client.post("/v1/chat", json=payload)
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code >= 400:
            return Sample(
                False, response.status_code, elapsed, scenario=scenario.name,
                error=f"HTTP {response.status_code}",
            )
        body = response.json()
        return Sample(
            True,
            200,
            elapsed,
            tokens=int((body.get("usage") or {}).get("completion_tokens", 0) or 0),
            scenario=scenario.name,
        )
    except Exception as exc:  # noqa: BLE001 - a failed request is data
        return Sample(
            False,
            0,
            (time.perf_counter() - started) * 1000,
            scenario=scenario.name,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _client_loop(
    client: Any,
    base_url: str,
    scenarios: list[Scenario],
    client_id: int,
    deadline: float,
    samples: list[Sample],
    start_delay: float,
    think_time: float,
) -> None:
    rng = random.Random(1000 + client_id)  # noqa: S311 - load shaping, not crypto
    if start_delay:
        await asyncio.sleep(start_delay)
    while time.monotonic() < deadline:
        scenario, prompt = _pick(scenarios, rng)
        samples.append(await _one_request(client, base_url, scenario, prompt, client_id))
        if think_time:
            await asyncio.sleep(rng.uniform(0.5 * think_time, 1.5 * think_time))


async def run(
    *,
    base_url: str,
    clients: int,
    duration: float,
    pattern: str,
    api_key: str,
    think_time: float,
) -> dict[str, Any]:
    import httpx  # noqa: PLC0415

    scenarios = load_scenarios()
    samples: list[Sample] = []
    headers = {"X-API-Key": api_key} if api_key else {}

    limits = httpx.Limits(max_connections=clients * 2, max_keepalive_connections=clients)
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"), timeout=httpx.Timeout(300.0), limits=limits, headers=headers
    ) as client:
        started = time.monotonic()
        deadline = started + duration

        if pattern == "burst":
            # Everyone starts at once; the queue is the thing under test.
            delays = [0.0] * clients
        elif pattern == "ramp":
            delays = [i * (duration * 0.4 / max(1, clients)) for i in range(clients)]
        else:  # constant / sustained
            delays = [i * 0.1 for i in range(clients)]

        await asyncio.gather(
            *(
                _client_loop(
                    client, base_url, scenarios, i, deadline, samples, delays[i], think_time
                )
                for i in range(clients)
            )
        )
        wall = time.monotonic() - started

    return summarise(samples, clients=clients, duration=wall, pattern=pattern, base_url=base_url)


def summarise(
    samples: list[Sample], *, clients: int, duration: float, pattern: str, base_url: str
) -> dict[str, Any]:
    ok = [s for s in samples if s.ok]
    failed = [s for s in samples if not s.ok]
    errors: dict[str, int] = {}
    for sample in failed:
        key = sample.error.split(":")[0] or f"HTTP {sample.status}"
        errors[key] = errors.get(key, 0) + 1

    by_scenario: dict[str, dict[str, Any]] = {}
    for sample in samples:
        bucket = by_scenario.setdefault(sample.scenario, {"requests": 0, "ok": 0, "latencies": []})
        bucket["requests"] += 1
        bucket["ok"] += 1 if sample.ok else 0
        if sample.ok:
            bucket["latencies"].append(sample.latency_ms)
    for bucket in by_scenario.values():
        bucket["success_rate"] = round(bucket["ok"] / bucket["requests"], 4) if bucket["requests"] else 0.0
        bucket["latency_ms"] = latency_percentiles(bucket.pop("latencies"))

    ttfts = [s.ttft_ms for s in ok if s.ttft_ms > 0]
    tokens = sum(s.tokens for s in ok)
    incomplete = sum(1 for s in samples if not s.stream_complete)

    return {
        "measured": True,
        "base_url": base_url,
        "pattern": pattern,
        "connected_clients": clients,
        "duration_seconds": round(duration, 1),
        "requests": len(samples),
        "succeeded": len(ok),
        "failed": len(failed),
        "success_rate": round(len(ok) / len(samples), 4) if samples else 0.0,
        "requests_per_second": round(len(ok) / duration, 3) if duration else 0.0,
        "latency_ms": latency_percentiles([s.latency_ms for s in ok]),
        "time_to_first_token_ms": latency_percentiles(ttfts),
        "tokens_generated": tokens,
        "tokens_per_second": round(tokens / duration, 2) if duration else 0.0,
        "incomplete_streams": incomplete,
        "http_503": sum(1 for s in failed if s.status == 503),
        "http_429": sum(1 for s in failed if s.status == 429),
        "http_5xx": sum(1 for s in failed if 500 <= s.status < 600),
        "errors": errors,
        "by_scenario": by_scenario,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python load_test/async_benchmark.py")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--pattern", choices=("constant", "ramp", "burst", "sustained"), default="constant"
    )
    parser.add_argument("--think-time", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    try:
        report = asyncio.run(
            run(
                base_url=args.base_url,
                clients=args.clients,
                duration=args.duration,
                pattern=args.pattern,
                api_key=args.api_key,
                think_time=args.think_time,
            )
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else _REPO_ROOT / "reports" / f"load_test_{args.pattern}_{args.clients}c.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwritten: {output}")

    if report["requests"] == 0:
        print("\nNo requests completed. Is the API running?  make serve", file=sys.stderr)
        return 1
    if report["success_rate"] < 0.95:
        print(
            f"\nSuccess rate {report['success_rate']:.1%} is below 95% - investigate before release.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
