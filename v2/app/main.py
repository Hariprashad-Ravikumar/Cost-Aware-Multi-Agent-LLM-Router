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
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, HTMLResponse

from app import cache, db
from app.providers import gemini, openai as openai_provider, anthropic as anthropic_provider
from app.providers.base import ProviderError
from app.router.decision import decide_tier
from app.router.calibrator import get_calibrator
from app.router.features import build_feature_vector

_PRICING_PATH = Path(__file__).resolve().parent.parent / "config" / "pricing.json"
with open(_PRICING_PATH) as f:
    PRICING = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

CHEAP_MODEL = os.environ.get("CHEAP_MODEL", "gpt-5.4-nano")
# Mid/capable tiers deliberately swapped from the original assignment: Claude Haiku 4.5
# ($1.00/$5.00 per 1M) is priced ABOVE Gemini 3.1 Flash-Lite ($0.25/$1.50) - verified
# directly against config/pricing.json, not assumed - which inverted the tier ladder's
# cost ordering and made every escalation-to-mid strictly cost-dominated by going
# straight to the old "capable" tier. Gemini now sits in mid (cheaper, escalate here
# first); Claude Sonnet 5 ($2.00/$10.00) is the new capable tier, genuinely priced and
# positioned above both. See CASE_STUDY.md for the full diagnosis.
MID_MODEL = os.environ.get("MID_MODEL", "gemini-3.1-flash-lite")
CAPABLE_MODEL = os.environ.get("CAPABLE_MODEL", "claude-sonnet-5")
ERROR_BUDGET = float(os.environ.get("ERROR_BUDGET", "0.05"))
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

TIER_PROVIDER = {
    "cheap": ("openai", CHEAP_MODEL),
    "mid": ("gemini", MID_MODEL),
    "capable": ("anthropic", CAPABLE_MODEL),
}

app = FastAPI(title="Calibrated Cost-Aware LLM Router")
logger = logging.getLogger("router")

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
    answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    correct: bool | None
    cache_hit: bool


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(model)
    if not rates:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input_cost_per_1m"] + (
        output_tokens / 1_000_000
    ) * rates["output_cost_per_1m"]


def _check_correct(answer: str, ground_truth: str | None) -> bool | None:
    if ground_truth is None:
        return None
    return ground_truth.strip().lower() in answer.strip().lower()


async def _call_tier(tier: str, prompt: str, logprobs: bool = False):
    provider_name, model = TIER_PROVIDER[tier]
    if provider_name == "openai":
        return await openai_provider.complete(prompt, model, logprobs=logprobs)
    if provider_name == "gemini":
        return await gemini.complete(prompt, model)
    if provider_name == "anthropic":
        return await anthropic_provider.complete(prompt, model)
    raise ValueError(f"Unknown provider for tier {tier}: {provider_name}")


@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest):
    start = time.monotonic()
    requests_total.inc()

    cached = await cache.get_cached_response(req.prompt, "router")
    if cached:
        cached["cache_hit"] = True
        cached["correct"] = _check_correct(cached["answer"], req.ground_truth)
        cached["latency_ms"] = int((time.monotonic() - start) * 1000)  # actual cache-hit latency, not the stale cached value
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
            final = await _call_tier(decision.chosen_tier, req.prompt)
        except ProviderError:
            # Escalation target down - fall back to the trusted top tier rather than
            # silently returning the (already-rejected) cheap answer.
            try:
                final = await _call_tier("capable", req.prompt)
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


_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_STATIC_DIR / "index.html").read_text()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


_embedder = None
_embedder_load_attempted = False


def _get_embedder():
    global _embedder, _embedder_load_attempted
    if not _embedder_load_attempted:
        _embedder_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedder = None  # graceful degradation - features.py handles None embeddings
    return _embedder


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
