"""Shared tier config, pricing, and provider dispatch.

Extracted from app/main.py so app/agents/ (the multi-agent graph) can dispatch to the
same three provider clients and cost table as the single-call /route endpoint, instead
of duplicating TIER_PROVIDER/PRICING or importing main.py (which would be circular -
main.py registers the /route-multi-agent endpoint that imports the agent graph).
"""
import json
import os
from pathlib import Path

from app.providers import gemini, openai as openai_provider, anthropic as anthropic_provider

_PRICING_PATH = Path(__file__).resolve().parent.parent / "config" / "pricing.json"
with open(_PRICING_PATH) as f:
    PRICING = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

CHEAP_MODEL = os.environ.get("CHEAP_MODEL", "gpt-5.4-nano")
# See CASE_STUDY.md for the tier-pricing-inversion diagnosis behind this ordering.
MID_MODEL = os.environ.get("MID_MODEL", "gemini-3.1-flash-lite")
CAPABLE_MODEL = os.environ.get("CAPABLE_MODEL", "claude-sonnet-5")

TIER_PROVIDER = {
    "cheap": ("openai", CHEAP_MODEL),
    "mid": ("gemini", MID_MODEL),
    "capable": ("anthropic", CAPABLE_MODEL),
}

CONCISE_SYSTEM_PROMPT = (
    "Default to a concise answer: give the direct answer first, then at most one "
    "short line of explanation or reasoning. Do not add extra caveats, "
    "restatements, or elaboration beyond that. If the user's own message "
    "explicitly asks for more detail, more steps, a full explanation, or "
    "clarification, give the fuller answer they asked for instead."
)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(model)
    if not rates:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input_cost_per_1m"] + (
        output_tokens / 1_000_000
    ) * rates["output_cost_per_1m"]


async def call_tier(tier: str, prompt: str, logprobs: bool = False, system_prompt: str | None = None):
    provider_name, model = TIER_PROVIDER[tier]
    if provider_name == "openai":
        return await openai_provider.complete(prompt, model, logprobs=logprobs, system_prompt=system_prompt)
    if provider_name == "gemini":
        return await gemini.complete(prompt, model, system_prompt=system_prompt)
    if provider_name == "anthropic":
        return await anthropic_provider.complete(prompt, model, system_prompt=system_prompt)
    raise ValueError(f"Unknown provider for tier {tier}: {provider_name}")
