"""Shared context object passed between planner/specialist/synthesizer graph nodes.

This is the structured-state mechanism from the multi-agent design (see
../../MULTI_AGENT.md): every node reads and writes this one typed object rather than
passing full raw transcripts to each other. Specialist nodes write only a condensed
SpecialistResult into `results`, matching the "condensed results, not raw transcripts"
pattern from Anthropic's multi-agent research system writeup - the planner and
synthesizer never see a specialist's full provider response, only its answer text,
the tier that produced it, and its cost/latency.
"""
from typing import Annotated, TypedDict


class SubTask(TypedDict):
    id: str
    prompt: str
    tier: str  # "cheap" | "mid" | "capable" - which existing tier client handles this


class SpecialistResult(TypedDict):
    subtask_id: str
    tier: str
    model: str
    answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


class MemoryHit(TypedDict):
    text: str
    metadata: dict
    similarity: float


class ConversationTurn(TypedDict):
    request: str
    answer: str


def _merge_dicts(left: dict, right: dict) -> dict:
    """Reducer for `results`: specialist nodes run concurrently and each contributes
    one key, so state updates must merge rather than overwrite."""
    return {**left, **right}


class AgentState(TypedDict):
    original_request: str
    ground_truth: str | None
    retrieved_memory: list[MemoryHit]
    # Recent verbatim turns of *this* conversation (app/agents/session.py, Redis-backed,
    # opt-in via session_id) - a plain last-write-wins channel: each turn's caller
    # passes the full history explicitly, no in-graph accumulation across separate
    # /route/multi-agent calls (each call uses its own checkpointer thread_id).
    conversation_history: list[ConversationTurn]
    subtasks: list[SubTask]
    results: Annotated[dict[str, SpecialistResult], _merge_dicts]
    final_answer: str | None
    trace: Annotated[list[str], lambda left, right: left + right]
