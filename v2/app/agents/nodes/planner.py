"""Planner node: decomposes the request into sub-tasks, each assigned to a tier.

Uses the capable tier to plan (a plan is a one-shot structured decision, not the kind
of repeated draft-then-escalate call the calibrator was trained around, so it doesn't
go through decide_tier - see app/router/decision.py's docstring on what that policy
actually governs). On any parse failure, falls back to a single subtask sent to the
cheap tier rather than failing the whole graph run - same graceful-degradation stance
as _get_embedder() in app/main.py.
"""
import json
import logging
import re

from app.agents.state import AgentState, SubTask
from app.providers.base import ProviderError
from app.tiers import call_tier

logger = logging.getLogger("router.agents.planner")

_PLANNER_SYSTEM_PROMPT = (
    "You are a planning agent. Break the user's request into 1-3 independent "
    "sub-tasks that, once each is answered, let you compose a complete final answer. "
    "Assign each sub-task a tier: 'cheap' for simple lookups/short factual questions, "
    "'mid' for moderate reasoning, 'capable' for complex or multi-step reasoning. "
    "Most requests need only one sub-task - only split when the request genuinely has "
    "independent parts. "
    "If the request references earlier conversation (e.g. 'that', 'it', 'the previous "
    "answer', 'what did I ask'), resolve it using ONLY the 'This conversation so far' "
    "section below, never the 'Unrelated background from other past sessions' section - "
    "the background section is NOT part of this conversation and was never said by "
    "this user in this session; treat it strictly as optional factual context, not as "
    "conversation history. If a request asks what was discussed/asked earlier and no "
    "'This conversation so far' section is present, say plainly that there is no prior "
    "turn in this session, rather than substituting the background section. "
    "Each sub-task prompt must stand alone as concrete, self-contained text. "
    "Respond with ONLY a JSON array, no prose, no markdown fences: "
    '[{"prompt": "...", "tier": "cheap|mid|capable"}, ...]'
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _fallback_subtasks(request: str) -> list[SubTask]:
    return [{"id": "t0", "prompt": request, "tier": "cheap"}]


def _parse_subtasks(raw_text: str, request: str) -> list[SubTask]:
    match = _JSON_ARRAY_RE.search(raw_text)
    if not match:
        return _fallback_subtasks(request)
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _fallback_subtasks(request)

    subtasks: list[SubTask] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict) or "prompt" not in item:
            continue
        tier = item.get("tier") if item.get("tier") in ("cheap", "mid", "capable") else "cheap"
        subtasks.append({"id": f"t{i}", "prompt": str(item["prompt"]), "tier": tier})

    return subtasks or _fallback_subtasks(request)


async def plan(state: AgentState) -> dict:
    # Two distinct sources, labeled so the model can't conflate them: conversation_history
    # is this session's actual turns (app/agents/session.py, Redis) - authoritative for
    # "what did I ask" style questions. retrieved_memory is pgvector similarity search
    # across OTHER sessions (app/agents/memory.py) - topically-related background facts
    # that were never said in this conversation. Verified this distinction matters: an
    # earlier version of this prompt let the planner answer "what was the first thing I
    # asked" using an unrelated retrieved_memory entry from a different session instead
    # of this session's real first turn.
    memory_context = ""
    if state.get("retrieved_memory"):
        lines = "\n".join(f"- {hit['text']}" for hit in state["retrieved_memory"])
        memory_context = f"\n\nUnrelated background from other past sessions (NOT this conversation):\n{lines}"

    history_context = ""
    if state.get("conversation_history"):
        turns = "\n".join(
            f"User: {t['request']}\nAssistant: {t['answer']}" for t in state["conversation_history"]
        )
        history_context = f"\n\nThis conversation so far (most recent last):\n{turns}"

    try:
        response = await call_tier(
            "capable",
            state["original_request"] + history_context + memory_context,
            system_prompt=_PLANNER_SYSTEM_PROMPT,
        )
        subtasks = _parse_subtasks(response.text, state["original_request"])
    except ProviderError as e:
        logger.warning(f"planner call failed, falling back to single cheap-tier subtask: {e}")
        subtasks = _fallback_subtasks(state["original_request"])

    trace = [f"planner: decomposed into {len(subtasks)} subtask(s): " + ", ".join(f"{t['id']}/{t['tier']}" for t in subtasks)]
    return {"subtasks": subtasks, "trace": trace}
