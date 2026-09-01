import os
import time

import httpx

from .base import ProviderResponse, make_breaker, with_retry, call_with_protection

_breaker = make_breaker("gemini")


@with_retry
async def _raw_complete(prompt: str, model: str, system_prompt: str | None = None) -> ProviderResponse:
    api_key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    start = time.monotonic()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers={"Content-Type": "application/json"}, json=body)
        resp.raise_for_status()
        data = resp.json()
    latency_ms = int((time.monotonic() - start) * 1000)

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    usage = data.get("usageMetadata", {})

    return ProviderResponse(
        text=text,
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
        latency_ms=latency_ms,
        mean_logprob=None,  # Gemini's generateContent does not expose token logprobs
        raw=data,
    )


async def complete(prompt: str, model: str, system_prompt: str | None = None) -> ProviderResponse:
    return await call_with_protection("gemini", _breaker, _raw_complete, prompt, model, system_prompt)
