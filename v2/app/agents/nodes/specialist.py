"""Specialist node: executes one sub-task against its assigned tier.

Invoked once per sub-task via LangGraph's Send API (see graph.py's dispatch_specialists)
so independent sub-tasks genuinely run as separate parallel graph-node invocations, not
a single node looping over a list - that's the "specialist agents" half of the
planner+specialist design. Each invocation writes only a condensed SpecialistResult
into shared state (state.py's _merge_dicts reducer), never the full provider response.
"""
import logging

from app.providers.base import ProviderError
from app.tiers import CONCISE_SYSTEM_PROMPT, TIER_PROVIDER, call_tier, compute_cost

logger = logging.getLogger("router.agents.specialist")


async def run_specialist(payload: dict) -> dict:
    subtask = payload["subtask"]
    tier = subtask["tier"]

    try:
        response = await call_tier(tier, subtask["prompt"], system_prompt=CONCISE_SYSTEM_PROMPT)
    except ProviderError as e:
        # Escalate one rung to the trusted capable tier rather than dropping the
        # sub-task's contribution to the final answer - mirrors main.py's /route
        # escalation-target-down fallback.
        logger.warning(f"tier {tier} unavailable for subtask {subtask['id']}, falling back to capable: {e}")
        tier = "capable"
        try:
            response = await call_tier(tier, subtask["prompt"], system_prompt=CONCISE_SYSTEM_PROMPT)
        except ProviderError as e2:
            return {
                "results": {
                    subtask["id"]: {
                        "subtask_id": subtask["id"],
                        "tier": "failed",
                        "model": "none",
                        "answer": f"[unavailable: {e2}]",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "latency_ms": 0,
                    }
                },
                "trace": [f"specialist {subtask['id']}: all tiers unavailable"],
            }

    model = TIER_PROVIDER[tier][1]
    cost = compute_cost(model, response.input_tokens, response.output_tokens)

    return {
        "results": {
            subtask["id"]: {
                "subtask_id": subtask["id"],
                "tier": tier,
                "model": model,
                "answer": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": cost,
                "latency_ms": response.latency_ms,
            }
        },
        "trace": [f"specialist {subtask['id']}: {tier}/{model} answered ({response.latency_ms}ms, ${cost:.5f})"],
    }
