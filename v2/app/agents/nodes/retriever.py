"""Retriever node: populates AgentState.retrieved_memory before planning.

Runs first in the graph so the planner can condition its decomposition on relevant
past runs (see app/agents/memory.py) - the RAG/long-term-memory layer, kept separate
from the live planner<->specialist handoff which uses AgentState directly.
"""
from app.agents import memory
from app.agents.state import AgentState


async def retrieve(state: AgentState) -> dict:
    hits = memory.retrieve_memory(state["original_request"], k=3)
    trace = [f"retriever: found {len(hits)} relevant memory entr{'y' if len(hits) == 1 else 'ies'}"]
    return {"retrieved_memory": hits, "trace": trace}
