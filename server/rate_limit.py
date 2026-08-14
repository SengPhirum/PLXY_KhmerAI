"""Per-client rate limiting (token bucket) and request-size limiting.

A token bucket rather than a fixed window: a customer typing three quick
follow-up questions is normal support behaviour and must not be throttled, while
a script issuing 100 requests/second must be.  The bucket allows a burst up to
``burst`` and then refills at ``requests / window`` per second.

State is in-process.  With ``KHMERAI_WORKERS=1`` (the documented production
setting, because Ollama is the real concurrency bottleneck) that is exact.  If
the deployment ever scales to multiple workers or hosts, swap
:class:`InMemoryRateLimiter` for a Redis-backed implementation of the same
protocol - the call site does not change.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol

__all__ = ["RateLimitDecision", "RateLimiter", "InMemoryRateLimiter", "NullRateLimiter"]


@dataclass(slots=True, frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: float
    retry_after: float = 0.0
    limit: int = 0

    def headers(self) -> dict[str, str]:
        # limit == 0 means rate limiting is disabled; advertising a limit of 0
        # would tell a client it may make no requests at all.
        if self.limit <= 0:
            return {}
        remaining = 0 if not math.isfinite(self.remaining) else max(0, int(self.remaining))
        out = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(remaining),
        }
        if not self.allowed:
            out["Retry-After"] = str(max(1, int(round(self.retry_after))))
        return out


class RateLimiter(Protocol):
    def check(self, key: str, cost: float = 1.0) -> RateLimitDecision: ...
    def reset(self, key: str | None = None) -> None: ...


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class InMemoryRateLimiter:
    """Thread-safe token bucket keyed by client identity."""

    def __init__(
        self,
        *,
        requests: int = 30,
        window_seconds: int = 60,
        burst: int = 10,
        max_keys: int = 10_000,
    ) -> None:
        if requests <= 0 or window_seconds <= 0:
            raise ValueError("requests and window_seconds must be positive")
        self.requests = requests
        self.window_seconds = window_seconds
        self.capacity = float(max(burst, 1))
        self.refill_per_second = requests / window_seconds
        self.max_keys = max_keys
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str, cost: float = 1.0) -> RateLimitDecision:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self.max_keys:
                    self._evict(now)
                bucket = _Bucket(tokens=self.capacity, updated_at=now)
                self._buckets[key] = bucket

            elapsed = now - bucket.updated_at
            bucket.tokens = min(
                self.capacity, bucket.tokens + elapsed * self.refill_per_second
            )
            bucket.updated_at = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return RateLimitDecision(
                    allowed=True, remaining=bucket.tokens, limit=self.requests
                )

            deficit = cost - bucket.tokens
            retry_after = deficit / self.refill_per_second if self.refill_per_second else 60.0
            return RateLimitDecision(
                allowed=False,
                remaining=bucket.tokens,
                retry_after=retry_after,
                limit=self.requests,
            )

    def _evict(self, now: float) -> None:
        """Drop buckets that have fully refilled - they carry no state."""
        stale = [
            key
            for key, bucket in self._buckets.items()
            if bucket.tokens + (now - bucket.updated_at) * self.refill_per_second >= self.capacity
        ]
        for key in stale:
            del self._buckets[key]
        if len(self._buckets) >= self.max_keys:
            oldest = sorted(self._buckets.items(), key=lambda kv: kv[1].updated_at)
            for key, _ in oldest[: len(oldest) // 4 or 1]:
                del self._buckets[key]

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

    def stats(self) -> dict[str, float]:
        with self._lock:
            return {
                "tracked_keys": len(self._buckets),
                "capacity": self.capacity,
                "refill_per_second": round(self.refill_per_second, 4),
            }


class NullRateLimiter:
    """Used when rate limiting is disabled."""

    def check(self, key: str, cost: float = 1.0) -> RateLimitDecision:  # noqa: ARG002
        return RateLimitDecision(allowed=True, remaining=float("inf"), limit=0)

    def reset(self, key: str | None = None) -> None:  # noqa: ARG002
        return
