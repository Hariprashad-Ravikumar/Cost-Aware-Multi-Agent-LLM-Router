"""Calibrated Cost-Aware LLM Router service.

Design note on self-consistency (documented, not hidden): the self_consistency_dispersion
feature requires multiple sampled calls to the cheap tier, which is only affordable to
collect offline (scripts/collect_calibration_data.py, run once against the training split).
At serve time this router makes exactly one cheap-tier call per request - that single
call's text IS the cheap-tier answer if accepted, so no call is wasted - and the
calibrator receives a neutral default (0.0) for self_consistency_dispersion instead of a
live measurement. This is a real train/serve feature skew, reported explicitly in
results/eval_report.md rather than concealed by re-running expensive sampling at serve time.
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, HTMLResponse

from sqlalchemy import text as sql_text

from app import cache, db
from app.agents import session as agent_session
from app.agents.graph import compile_graph
from app.embeddings import get_embedder as _get_embedder
from app.providers import gemini, openai as openai_provider, anthropic as anthropic_provider
from app.providers.base import ProviderError
from app.router.decision import decide_tier
from app.router.calibrator import get_calibrator
from app.router.features import build_feature_vector
from app.tiers import (
    PRICING,
    CHEAP_MODEL,
    MID_MODEL,
    CAPABLE_MODEL,
    TIER_PROVIDER,
    CONCISE_SYSTEM_PROMPT,
    compute_cost as _compute_cost,
    call_tier as _call_tier,
)

ERROR_BUDGET = float(os.environ.get("ERROR_BUDGET", "0.05"))
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

logger = logging.getLogger("router")

_multi_agent_graph = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _multi_agent_graph
    database_url = os.environ.get("DATABASE_URL", "postgresql://router:router@localhost:5432/router")
    # Same graceful-degradation stance as _get_embedder(): the multi-agent endpoint's
    # state checkpointing is an enhancement (resumable/durable runs), not a hard
    # dependency - a DB outage at startup shouldn't take down the whole service,
    # including the unrelated single-call /route endpoint.
    try:
        async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
            await checkpointer.setup()
            _multi_agent_graph = compile_graph(checkpointer=checkpointer)
            logger.info("multi-agent graph compiled with Postgres checkpointing")
            yield
    except Exception as e:
        logger.error(f"Postgres checkpointer unavailable, multi-agent graph running without persistence: {e}")
        _multi_agent_graph = compile_graph(checkpointer=None)
        yield


app = FastAPI(title="Calibrated Cost-Aware LLM Router", lifespan=_lifespan)

requests_total = Counter("router_requests_total", "Total routed requests")
latency_hist = Histogram("router_latency_seconds", "End-to-end request latency")
cost_total = Counter("router_cost_usd_total", "Cumulative cost in USD")
tier_selected_total = Counter("router_tier_selected_total", "Requests by chosen tier", ["tier"])


class RouteRequest(BaseModel):
    id: str | None = None
    prompt: str
    ground_truth: str | None = None
    source: str | None = None


class RouteResponse(BaseModel):
    id: str | None
    chosen_tier: str
    model: str
    predicted_p_correct: float | None
    reason: str
    answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    correct: bool | None
    cache_hit: bool


def _check_correct(answer: str, ground_truth: str | None) -> bool | None:
    if ground_truth is None:
        return None
    return ground_truth.strip().lower() in answer.strip().lower()


@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest):
    start = time.monotonic()
    requests_total.inc()

    cached = await cache.get_cached_response(req.prompt, "router")
    if cached:
        cached["cache_hit"] = True
        cached["correct"] = _check_correct(cached["answer"], req.ground_truth)
        cached["latency_ms"] = int((time.monotonic() - start) * 1000)  # actual cache-hit latency, not the stale cached value
        # A cache entry written before the "reason" field was added won't have it -
        # backfill honestly rather than 500ing on a schema change mid-TTL.
        cached.setdefault("reason", "unknown - cached before routing reasons were tracked")
        latency_hist.observe(cached["latency_ms"] / 1000)
        return RouteResponse(id=req.id, **cached)

    # Single draft call to the cheap tier - doubles as the feature-extraction probe
    # and, if accepted, as the final answer (see module docstring on self-consistency).
    try:
        draft = await _call_tier("cheap", req.prompt, logprobs=True)
    except ProviderError as e:
        raise HTTPException(status_code=502, detail=f"cheap tier unavailable: {e}")

    embedder = _get_embedder()
    prompt_embedding = embedder.encode(req.prompt) if embedder is not None else None
    hard_cluster_centroid = _get_hard_cluster_centroid()

    features = build_feature_vector(
        draft_text=draft.text,
        draft_mean_logprob=draft.mean_logprob,
        sample_texts=[],  # self-consistency not sampled at serve time - see module docstring
        prompt_embedding=prompt_embedding,
        hard_cluster_centroid=hard_cluster_centroid,
    )

    p_correct_cheap = get_calibrator().predict_p_correct(features)
    decision = decide_tier(p_correct_cheap, ERROR_BUDGET)
    tier_selected_total.labels(tier=decision.chosen_tier).inc()

    if decision.chosen_tier == "cheap":
        final = draft
        model_used = CHEAP_MODEL
    else:
        try:
            final = await _call_tier(decision.chosen_tier, req.prompt, system_prompt=CONCISE_SYSTEM_PROMPT)
        except ProviderError:
            # Escalation target down - fall back to the trusted top tier rather than
            # silently returning the (already-rejected) cheap answer.
            try:
                final = await _call_tier("capable", req.prompt, system_prompt=CONCISE_SYSTEM_PROMPT)
                decision.chosen_tier = "capable"
            except ProviderError as e2:
                raise HTTPException(status_code=502, detail=f"all escalation targets unavailable: {e2}")
        model_used = TIER_PROVIDER[decision.chosen_tier][1]

    cost = _compute_cost(model_used, final.input_tokens, final.output_tokens)
    latency_ms = int((time.monotonic() - start) * 1000)
    correct = _check_correct(final.text, req.ground_truth)

    cost_total.inc(cost)
    latency_hist.observe(latency_ms / 1000)

    result = {
        "chosen_tier": decision.chosen_tier,
        "model": model_used,
        "predicted_p_correct": decision.predicted_p_correct,
        "reason": decision.reason,
        "answer": final.text,
        "input_tokens": final.input_tokens,
        "output_tokens": final.output_tokens,
        "cost_usd": cost,
        "latency_ms": latency_ms,
        "cache_hit": False,
    }
    await cache.set_cached_response(req.prompt, "router", result, ttl_seconds=CACHE_TTL)

    # Logging is best-effort: the LLM answer above already succeeded and is the thing
    # the caller actually wants. A transient DB issue (e.g. a serverless Postgres
    # provider closing an idle connection) shouldn't turn a successful routing decision
    # into a 500 - log it and still return the real answer.
    try:
        session = db.get_session()
        try:
            session.add(
                db.RoutedRequest(
                    prompt=req.prompt,
                    features=features.as_array().tolist(),
                    predicted_p_correct=decision.predicted_p_correct,
                    error_budget=ERROR_BUDGET,
                    chosen_tier=decision.chosen_tier,
                    escalated=decision.escalated,
                    input_tokens=final.input_tokens,
                    output_tokens=final.output_tokens,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    correct=correct,
                    cache_hit=False,
                )
            )
            session.commit()
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Failed to log request to DB (non-fatal, answer still returned): {e}")

    return RouteResponse(id=req.id, correct=correct, **result)


class MultiAgentRequest(BaseModel):
    id: str | None = None
    prompt: str
    # Opt-in: omitted means one-shot/stateless, same as before. Provided means this
    # request is one turn of an ongoing conversation - the caller generates it for the
    # first turn and passes the same value back on every follow-up. See
    # app/agents/session.py for what this actually tracks and why it's a separate
    # layer from the long-term pgvector memory.
    session_id: str | None = None


class MultiAgentSubtaskResult(BaseModel):
    subtask_id: str
    tier: str
    model: str
    answer: str
    cost_usd: float
    latency_ms: int


class MultiAgentResponse(BaseModel):
    id: str | None
    session_id: str | None
    answer: str
    subtasks: list[MultiAgentSubtaskResult]
    trace: list[str]
    total_cost_usd: float
    latency_ms: int


@app.post("/route/multi-agent", response_model=MultiAgentResponse)
async def route_multi_agent(req: MultiAgentRequest):
    """Planner + specialist-tiers + synthesizer graph (app/agents/), separate from the
    single-call /route endpoint above - see v2/MULTI_AGENT.md for the design.
    """
    start = time.monotonic()
    run_id = req.id or str(uuid.uuid4())

    conversation_history = await agent_session.get_history(req.session_id) if req.session_id else []

    config = {"configurable": {"thread_id": run_id}}
    final_state = await _multi_agent_graph.ainvoke(
        {
            "original_request": req.prompt,
            "ground_truth": None,
            "retrieved_memory": [],
            "conversation_history": conversation_history,
            "subtasks": [],
            "results": {},
            "final_answer": None,
            "trace": [],
        },
        config=config,
    )

    latency_ms = int((time.monotonic() - start) * 1000)
    subtask_results = sorted(final_state["results"].values(), key=lambda r: r["subtask_id"])
    total_cost = sum(r["cost_usd"] for r in subtask_results)
    final_answer = final_state["final_answer"] or ""

    if req.session_id:
        await agent_session.append_turn(req.session_id, req.prompt, final_answer)

    return MultiAgentResponse(
        id=run_id,
        session_id=req.session_id,
        answer=final_answer,
        subtasks=[MultiAgentSubtaskResult(**r) for r in subtask_results],
        trace=final_state["trace"],
        total_cost_usd=total_cost,
        latency_ms=latency_ms,
    )


_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_STATIC_DIR / "index.html").read_text()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/detail")
async def health_detail():
    """Real dependency checks - not a static 'all green' claim. Each check is
    isolated so one dependency being down still reports the others accurately
    instead of throwing away the whole response.
    """
    try:
        session = db.get_session()
        try:
            session.execute(sql_text("SELECT 1"))
            db_status = "ok"
        finally:
            session.close()
    except Exception as e:
        db_status = f"error: {e}"

    try:
        redis_status = "ok" if await cache.ping() else "error: ping returned falsy"
    except Exception as e:
        redis_status = f"error: {e}"

    return {
        "db": db_status,
        "redis": redis_status,
        "providers": {
            "openai": openai_provider._breaker.current_state,
            "gemini": gemini._breaker.current_state,
            "anthropic": anthropic_provider._breaker.current_state,
        },
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Registered after the explicit routes above, so it only catches paths those routes
# don't handle (e.g. /headshot.webp) - the "/" route above still serves the
# rendered index.html rather than a raw file listing.
app.mount("/", StaticFiles(directory=_STATIC_DIR), name="static")


_hard_cluster_centroid = None
_hard_cluster_load_attempted = False


def _get_hard_cluster_centroid():
    global _hard_cluster_centroid, _hard_cluster_load_attempted
    if not _hard_cluster_load_attempted:
        _hard_cluster_load_attempted = True
        centroid_path = Path(__file__).resolve().parent.parent / "models" / "hard_cluster_centroid.npy"
        if centroid_path.exists():
            _hard_cluster_centroid = np.load(centroid_path)
    return _hard_cluster_centroid
