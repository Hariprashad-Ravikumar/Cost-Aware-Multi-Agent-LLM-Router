"""Synthesizer node: combines specialist results into one final answer, and writes
a summary of the run to long-term memory for future requests to retrieve.
"""
import logging

from app.agents import memory
from app.agents.state import AgentState
from app.providers.base import ProviderError
from app.tiers import CONCISE_SYSTEM_PROMPT, call_tier

logger = logging.getLogger("router.agents.synthesizer")


def _format_results(state: AgentState) -> str:
    ordered = sorted(state["results"].values(), key=lambda r: r["subtask_id"])
    return "\n\n".join(f"[{r['subtask_id']} / {r['tier']}]: {r['answer']}" for r in ordered)


async def synthesize(state: AgentState) -> dict:
    results_text = _format_results(state)

    # Single sub-task, no real synthesis needed - return its answer directly instead
    # of paying for another capable-tier call to reword one input into itself.
    if len(state["results"]) == 1:
        final_answer = next(iter(state["results"].values()))["answer"]
    else:
        prompt = (
            f"Original request: {state['original_request']}\n\n"
            f"Sub-task answers:\n{results_text}\n\n"
            "Combine these into one complete, coherent final answer to the original request."
        )
        try:
            response = await call_tier("capable", prompt, system_prompt=CONCISE_SYSTEM_PROMPT)
            final_answer = response.text
        except ProviderError as e:
            logger.warning(f"synthesis call failed, concatenating sub-answers instead: {e}")
            final_answer = results_text

    memory.write_memory(
        text=f"Q: {state['original_request']}\nA: {final_answer}",
        metadata={"num_subtasks": len(state["results"])},
    )

    return {"final_answer": final_answer, "trace": ["synthesizer: combined sub-task results into final answer"]}
