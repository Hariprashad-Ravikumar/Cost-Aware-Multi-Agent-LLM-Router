"""Short-term conversational memory: recent turns within one chat session.

Deliberately a third, distinct layer from the other two the multi-agent design
already has - see MULTI_AGENT.md:
  - AgentState (state.py): live, single-run coordination between planner/specialists
  - memory_entries (memory.py, pgvector): long-term recall across separate sessions,
    RAG-retrieved by topical similarity
  - this module: the last few verbatim turns of *this* conversation, in Redis with a
    TTL, so a follow-up like "minus 2 from that" can be resolved by the planner
    against what was actually just said - a pronoun-reference problem that semantic
    similarity search (pgvector) is the wrong tool for.

Opt-in only: a session is tracked only when the caller supplies a session_id on
/route/multi-agent. Omitting it keeps that endpoint's default one-shot, stateless
behavior - no conversation is tracked implicitly.
"""
import json
import os

from app import cache

SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "1800"))
# Bounds prompt size handed to the planner - a session that runs long enough to hit
# this cap still works, it just stops recalling the oldest turns.
MAX_TURNS = 8


def _session_key(session_id: str) -> str:
    return f"router:session:{session_id}"


async def get_history(session_id: str) -> list[dict]:
    client = cache.get_client()
    raw = await client.get(_session_key(session_id))
    return json.loads(raw) if raw else []


async def append_turn(session_id: str, request: str, answer: str) -> None:
    client = cache.get_client()
    history = await get_history(session_id)
    history.append({"request": request, "answer": answer})
    history = history[-MAX_TURNS:]
    await client.set(_session_key(session_id), json.dumps(history), ex=SESSION_TTL_SECONDS)
