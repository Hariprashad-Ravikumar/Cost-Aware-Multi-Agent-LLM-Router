"""Wires the retriever -> planner -> specialist(s) -> synthesizer graph.

Dispatch to specialists uses LangGraph's Send API (dispatch_specialists) so each
sub-task genuinely runs as an independent parallel node invocation - real multi-agent
fan-out, not one node looping over a list. See state.py for the shared context object
these nodes read/write, and ../../MULTI_AGENT.md for the full design writeup.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.agents.nodes.planner import plan
from app.agents.nodes.retriever import retrieve
from app.agents.nodes.specialist import run_specialist
from app.agents.nodes.synthesizer import synthesize
from app.agents.state import AgentState


def dispatch_specialists(state: AgentState) -> list[Send]:
    return [Send("specialist", {"subtask": subtask}) for subtask in state["subtasks"]]


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("plan", plan)
    builder.add_node("specialist", run_specialist)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "plan")
    builder.add_conditional_edges("plan", dispatch_specialists, ["specialist"])
    builder.add_edge("specialist", "synthesize")
    builder.add_edge("synthesize", END)

    return builder


def compile_graph(checkpointer=None):
    return build_graph().compile(checkpointer=checkpointer)
