import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch

import pytest

from app.agents.graph import compile_graph
from app.agents.nodes.planner import _parse_subtasks
from app.providers.base import ProviderError, ProviderResponse


def test_parse_subtasks_valid_json():
    raw = 'Here you go:\n[{"prompt": "a", "tier": "cheap"}, {"prompt": "b", "tier": "capable"}]'
    subtasks = _parse_subtasks(raw, "original")
    assert len(subtasks) == 2
    assert subtasks[0] == {"id": "t0", "prompt": "a", "tier": "cheap"}
    assert subtasks[1]["tier"] == "capable"


def test_parse_subtasks_invalid_tier_defaults_to_cheap():
    raw = '[{"prompt": "a", "tier": "nonsense"}]'
    subtasks = _parse_subtasks(raw, "original")
    assert subtasks[0]["tier"] == "cheap"


def test_parse_subtasks_malformed_json_falls_back_to_single_subtask():
    raw = "not json at all"
    subtasks = _parse_subtasks(raw, "original request")
    assert subtasks == [{"id": "t0", "prompt": "original request", "tier": "cheap"}]


def test_parse_subtasks_empty_array_falls_back():
    subtasks = _parse_subtasks("[]", "original request")
    assert subtasks == [{"id": "t0", "prompt": "original request", "tier": "cheap"}]


async def _fake_call_tier(tier, prompt, logprobs=False, system_prompt=None):
    if system_prompt and "Break the user" in system_prompt:
        return ProviderResponse(
            text='[{"prompt": "sub one", "tier": "cheap"}]', input_tokens=10, output_tokens=10, latency_ms=1
        )
    return ProviderResponse(text=f"answer for {tier}", input_tokens=5, output_tokens=5, latency_ms=1)


@pytest.mark.asyncio
async def test_graph_single_subtask_end_to_end():
    graph = compile_graph(checkpointer=None)
    with patch("app.agents.nodes.planner.call_tier", _fake_call_tier), \
         patch("app.agents.nodes.specialist.call_tier", _fake_call_tier), \
         patch("app.agents.memory.retrieve_memory", lambda *a, **kw: []), \
         patch("app.agents.memory.write_memory", lambda *a, **kw: None):
        result = await graph.ainvoke(
            {
                "original_request": "a single-part question",
                "ground_truth": None,
                "retrieved_memory": [],
                "subtasks": [],
                "results": {},
                "final_answer": None,
                "trace": [],
            }
        )
    # Single sub-task: synthesizer returns the specialist's answer directly, no
    # extra capable-tier synthesis call - see nodes/synthesizer.py.
    assert result["final_answer"] == "answer for cheap"
    assert len(result["results"]) == 1


@pytest.mark.asyncio
async def test_planner_receives_conversation_history_in_prompt():
    """The planner should be given the conversation history so it can resolve a
    reference like 'that' into self-contained sub-task text (see planner.py's
    history_context) - this checks the history actually reaches the provider call,
    not that a live LLM resolves it correctly."""
    from app.agents.nodes.planner import plan

    captured_prompt = {}

    async def capturing_call_tier(tier, prompt, logprobs=False, system_prompt=None):
        captured_prompt["prompt"] = prompt
        return ProviderResponse(
            text='[{"prompt": "36 minus 2", "tier": "cheap"}]', input_tokens=1, output_tokens=1, latency_ms=1
        )

    with patch("app.agents.nodes.planner.call_tier", capturing_call_tier):
        result = await plan(
            {
                "original_request": "minus 2 from that",
                "conversation_history": [{"request": "What is 15% of 240?", "answer": "36"}],
                "retrieved_memory": [],
            }
        )

    assert "What is 15% of 240?" in captured_prompt["prompt"]
    assert "36" in captured_prompt["prompt"]
    assert result["subtasks"][0]["prompt"] == "36 minus 2"


@pytest.mark.asyncio
async def test_specialist_falls_back_to_capable_tier_on_provider_error():
    from app.agents.nodes.specialist import run_specialist

    call_log = []

    async def flaky_call_tier(tier, prompt, logprobs=False, system_prompt=None):
        call_log.append(tier)
        if tier == "mid":
            raise ProviderError("gemini", "simulated outage", retriable=True)
        return ProviderResponse(text="fallback answer", input_tokens=1, output_tokens=1, latency_ms=1)

    with patch("app.agents.nodes.specialist.call_tier", flaky_call_tier):
        out = await run_specialist({"subtask": {"id": "t0", "prompt": "x", "tier": "mid"}})

    assert call_log == ["mid", "capable"]
    assert out["results"]["t0"]["tier"] == "capable"
    assert out["results"]["t0"]["answer"] == "fallback answer"
