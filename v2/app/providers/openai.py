import os
import time

import httpx

from .base import ProviderResponse, make_breaker, with_retry, call_with_protection, parse_go_duration

_BASE_URL = "https://api.openai.com/v1/chat/completions"
_breaker = make_breaker("openai")


@with_retry
async def _raw_complete(prompt: str, model: str, logprobs: bool = False, temperature: float = 0) -> ProviderResponse:
    api_key = os.environ["OPENAI_API_KEY"]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if logprobs:
        body["logprobs"] = True
    # Some OpenAI models reject a non-default temperature; only send it when non-zero
    # sampling is explicitly requested (self-consistency data collection).
    if temperature:
        body["temperature"] = temperature

    start = time.monotonic()
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _BASE_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        headers = resp.headers
    latency_ms = int((time.monotonic() - start) * 1000)

    remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
    reset_tokens = headers.get("x-ratelimit-reset-tokens")

    choice = data["choices"][0]
    text = choice["message"]["content"]
    usage = data.get("usage", {})

    mean_logprob = None
    lp = choice.get("logprobs")
    if lp and lp.get("content"):
        token_logprobs = [t["logprob"] for t in lp["content"] if t.get("logprob") is not None]
        if token_logprobs:
            mean_logprob = sum(token_logprobs) / len(token_logprobs)

    return ProviderResponse(
        text=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        mean_logprob=mean_logprob,
        raw=data,
        rate_limit_remaining_tokens=int(remaining_tokens) if remaining_tokens else None,
        rate_limit_reset_tokens_seconds=parse_go_duration(reset_tokens) if reset_tokens else None,
    )


async def complete(prompt: str, model: str, logprobs: bool = False, temperature: float = 0) -> ProviderResponse:
    return await call_with_protection("openai", _breaker, _raw_complete, prompt, model, logprobs, temperature)
