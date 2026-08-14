"""Locust load profile for the Khmer support API (Phase 18).

    pip install locust
    locust -f load_test/locustfile.py --host http://127.0.0.1:8000
    locust -f load_test/locustfile.py --host http://127.0.0.1:8000 \
           --headless -u 10 -r 2 -t 5m

Task weights match the §Phase 18 traffic mixture. Prefer
``load_test/async_benchmark.py`` for the numbers that go into a report - it
measures TTFT on the streaming path, which Locust's HTTP client does not.
Locust is here for interactive exploration and ramp shapes.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, events, task

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCENARIO_DIR = _REPO_ROOT / "load_test" / "scenarios"


def _load(name: str) -> list[dict]:
    path = SCENARIO_DIR / name
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


SHORT = _load("short_questions.jsonl")
RAG = _load("rag_questions.jsonl")
MULTITURN = _load("multiturn.jsonl")
LONG = _load("long_questions.jsonl")
ADVERSARIAL = _load("adversarial.jsonl")


class SupportCustomer(HttpUser):
    """One connected customer. Think time reflects real reading speed."""

    wait_time = between(2, 8)

    def on_start(self) -> None:
        self.conversation_id = f"locust-{random.randint(1, 10**9)}"  # noqa: S311
        self.client.headers.update({"Content-Type": "application/json"})

    def _chat(self, message: str, *, stream: bool, name: str, product_id: str = "") -> None:
        payload = {
            "message": message,
            "conversation_id": self.conversation_id,
            "stream": stream,
        }
        if product_id:
            payload["product_id"] = product_id

        if not stream:
            with self.client.post("/v1/chat", json=payload, name=name, catch_response=True) as response:
                if response.status_code == 503:
                    response.success()  # queue full is a controlled outcome, not a failure
                elif response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}")
            return

        started = time.perf_counter()
        first_token = None
        with self.client.post(
            "/v1/chat/stream", json=payload, name=name, stream=True, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            saw_done = False
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                if not text.startswith("data: "):
                    continue
                event = json.loads(text[6:])
                if event.get("type") == "token" and first_token is None:
                    first_token = time.perf_counter()
                    events.request.fire(
                        request_type="TTFT",
                        name=name,
                        response_time=(first_token - started) * 1000,
                        response_length=0,
                    )
                elif event.get("type") == "done":
                    saw_done = True
            if not saw_done:
                response.failure("stream ended without a done event")

    @task(50)
    def short_question(self) -> None:
        if SHORT:
            self._chat(random.choice(SHORT)["message"], stream=False, name="/v1/chat [short]")  # noqa: S311

    @task(25)
    def rag_question(self) -> None:
        if RAG:
            row = random.choice(RAG)  # noqa: S311
            self._chat(
                row["message"], stream=True, name="/v1/chat/stream [rag]",
                product_id=row.get("product_id", ""),
            )

    @task(15)
    def follow_up(self) -> None:
        if MULTITURN:
            for row in random.sample(MULTITURN, k=min(2, len(MULTITURN))):  # noqa: S311
                self._chat(row["message"], stream=True, name="/v1/chat/stream [multiturn]")

    @task(5)
    def long_question(self) -> None:
        if LONG:
            self._chat(random.choice(LONG)["message"], stream=True, name="/v1/chat/stream [long]")  # noqa: S311

    @task(5)
    def adversarial(self) -> None:
        if ADVERSARIAL:
            self._chat(
                random.choice(ADVERSARIAL)["message"], stream=False, name="/v1/chat [adversarial]"  # noqa: S311
            )

    @task(2)
    def health(self) -> None:
        self.client.get("/health", name="/health")


class RampToTwenty(LoadTestShape):
    """Ramp 1 -> 5 -> 10 -> 20 connected clients, holding each step.

        locust -f load_test/locustfile.py --headless --host http://127.0.0.1:8000 \
               --class-picker RampToTwenty
    """

    stages = [
        {"duration": 60, "users": 1, "spawn_rate": 1},
        {"duration": 180, "users": 5, "spawn_rate": 1},
        {"duration": 360, "users": 10, "spawn_rate": 1},
        {"duration": 540, "users": 20, "spawn_rate": 2},
    ]

    def tick(self):  # type: ignore[no-untyped-def]
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
