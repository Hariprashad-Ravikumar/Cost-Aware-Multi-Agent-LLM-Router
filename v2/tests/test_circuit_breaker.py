import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.providers.base import make_breaker, call_with_protection, ProviderError


async def _ok():
    return "ok"


async def _fail():
    raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_breaker_starts_closed_and_passes_calls():
    b = make_breaker("t1")
    assert b.current_state == "closed"
    result = await call_with_protection("t1", b, _ok)
    assert result == "ok"
    assert b.current_state == "closed"


@pytest.mark.asyncio
async def test_breaker_opens_after_fail_max_consecutive_failures():
    b = make_breaker("t2")
    for _ in range(b.fail_max):
        with pytest.raises(RuntimeError):
            await call_with_protection("t2", b, _fail)
    assert b.current_state == "open"


@pytest.mark.asyncio
async def test_open_breaker_fails_fast_with_provider_error():
    b = make_breaker("t3")
    for _ in range(b.fail_max):
        with pytest.raises(RuntimeError):
            await call_with_protection("t3", b, _fail)
    with pytest.raises(ProviderError):
        await call_with_protection("t3", b, _ok)


@pytest.mark.asyncio
async def test_breaker_recovers_to_closed_after_reset_timeout():
    b = make_breaker("t4")
    for _ in range(b.fail_max):
        with pytest.raises(RuntimeError):
            await call_with_protection("t4", b, _fail)
    assert b.current_state == "open"

    b._opened_at -= b.reset_timeout_s + 1  # simulate time passing without a real sleep
    result = await call_with_protection("t4", b, _ok)
    assert result == "ok"
    assert b.current_state == "closed"
