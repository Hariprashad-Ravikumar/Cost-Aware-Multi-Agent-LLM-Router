import hashlib
import json
import os

import redis.asyncio as redis

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _client = redis.from_url(url, decode_responses=True)
    return _client


def _cache_key(prompt: str, model: str) -> str:
    normalized = " ".join(prompt.strip().lower().split())
    digest = hashlib.sha256(f"{model}:{normalized}".encode()).hexdigest()
    return f"router:cache:{digest}"


async def get_cached_response(prompt: str, model: str) -> dict | None:
    client = _get_client()
    raw = await client.get(_cache_key(prompt, model))
    return json.loads(raw) if raw else None


async def set_cached_response(prompt: str, model: str, response: dict, ttl_seconds: int = 3600) -> None:
    client = _get_client()
    await client.set(_cache_key(prompt, model), json.dumps(response), ex=ttl_seconds)
