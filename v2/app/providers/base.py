"""Shared retry + circuit-breaker wrapper for provider HTTP calls.

Each provider client wraps its raw HTTP call with:
  - tenacity retry: up to 3 attempts, exponential backoff, only on transient
    errors (timeouts, 429, 5xx) - never retries on 4xx client errors like bad auth.
  - a hand-rolled async circuit breaker: after 5 consecutive failures, the breaker
    opens for 30s and calls fail fast (CircuitOpenError) instead of hammering a down
    provider. The router's decision layer catches this and falls back to the next
    tier up, logging the fallback explicitly.

Note on pybreaker: the obvious off-the-shelf choice (pybreaker.CircuitBreaker.call_async)
is broken in the currently pinned version - it references `tornado.gen` without importing
tornado, so every async call raises NameError. Verified directly against the installed
package rather than assumed from the README. Written here as a small state machine instead
of pulling in tornado as an undocumented transitive dependency to work around it.
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from prometheus_client import Gauge

circuit_breaker_state = Gauge(
    "router_circuit_breaker_state",
    "Circuit breaker state per provider (0=closed, 1=open)",
    ["provider"],
)


class ProviderError(Exception):
    """Raised for a provider call that failed after retries or via an open breaker."""

    def __init__(self, provider: str, message: str, retriable: bool = False):
        self.provider = provider
        self.retriable = retriable
        super().__init__(f"[{provider}] {message}")


class CircuitOpenError(Exception):
    pass


class _State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class AsyncCircuitBreaker:
    provider_name: str
    fail_max: int = 5
    reset_timeout_s: float = 30.0
    _state: _State = field(default=_State.CLOSED, init=False)
    _fail_count: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        circuit_breaker_state.labels(provider=self.provider_name).set(0)

    def _set_state(self, new_state: _State):
        if new_state != self._state:
            self._state = new_state
            circuit_breaker_state.labels(provider=self.provider_name).set(
                1 if new_state == _State.OPEN else 0
            )

    async def call(self, fn, *args, **kwargs):
        async with self._lock:
            if self._state == _State.OPEN:
                if time.monotonic() - self._opened_at >= self.reset_timeout_s:
                    self._set_state(_State.HALF_OPEN)
                else:
                    raise CircuitOpenError(f"circuit open for {self.provider_name}")

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            async with self._lock:
                self._fail_count += 1
                if self._state == _State.HALF_OPEN or self._fail_count >= self.fail_max:
                    self._set_state(_State.OPEN)
                    self._opened_at = time.monotonic()
            raise
        else:
            async with self._lock:
                self._fail_count = 0
                self._set_state(_State.CLOSED)
            return result

    @property
    def current_state(self) -> str:
        return self._state.value


def make_breaker(provider_name: str) -> AsyncCircuitBreaker:
    return AsyncCircuitBreaker(provider_name=provider_name)


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    mean_logprob: float | None = None  # average token logprob of the response, if the provider exposes it
    raw: dict = field(default_factory=dict)
    rate_limit_remaining_tokens: int | None = None  # from x-ratelimit-remaining-tokens, when the provider exposes it
    rate_limit_reset_tokens_seconds: float | None = None  # from x-ratelimit-reset-tokens


def parse_go_duration(s: str) -> float:
    """Parses a Go-style duration string (e.g. '547ms', '1m2.3s', '11h47m2.4s') to seconds.

    Groq's rate-limit headers use this format - verified directly against real responses,
    not assumed from docs.
    """
    import re

    pattern = r"(\d+\.?\d*)(h|ms|m|s)"
    total = 0.0
    for value, unit in re.findall(pattern, s):
        value = float(value)
        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        elif unit == "ms":
            total += value / 1000
        elif unit == "s":
            total += value
    return total


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


def with_retry(fn):
    """Decorator applying the standard transient-error retry policy."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_transient),
        reraise=True,
    )(fn)


async def call_with_protection(provider_name: str, breaker: AsyncCircuitBreaker, fn, *args, **kwargs):
    """Runs fn through the circuit breaker; raises ProviderError on open-breaker or exhausted retries."""
    try:
        return await breaker.call(fn, *args, **kwargs)
    except CircuitOpenError as e:
        raise ProviderError(provider_name, "circuit breaker open, provider likely down", retriable=False) from e
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        raise ProviderError(provider_name, str(e), retriable=True) from e
