import os
import time

import httpx

from .base import ProviderResponse, make_breaker, with_retry, call_with_protection

_BASE_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_breaker = make_breaker("anthropic")


@with_retry
async def _raw_complete(
    prompt: str, model: str, max_tokens: int = 4096, system_prompt: str | None = None
) -> ProviderResponse:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        body["system"] = system_prompt

    start = time.monotonic()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            _BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
    latency_ms = int((time.monotonic() - start) * 1000)

    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    usage = data.get("usage", {})

    return ProviderResponse(
        text=text,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        latency_ms=latency_ms,
        mean_logprob=None,  # Anthropic's Messages API does not expose token logprobs
        raw=data,
    )


async def complete(prompt: str, model: str, system_prompt: str | None = None) -> ProviderResponse:
    return await call_with_protection("anthropic", _breaker, _raw_complete, prompt, model, 4096, system_prompt)
