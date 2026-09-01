"""Held-out evaluation: calibrated router vs. three baselines, on data/eval_holdout.jsonl -
prompts never touched during calibrator training (see build_eval_sets.py's disjoint split).

Four policies compared, per prompt:
  1. calibrated_router - the actual serve-time policy (app/main.py): one cheap-tier draft
     call, calibrator predicts P(correct), escalate to mid if predicted error > budget
     (matching decision.py's real default - no mid-tier calibrator exists, so escalation
     to mid is unconditional once cheap fails budget, exactly as the live service behaves).
  2. naive_classifier - v1's original approach: ask the cheap-tier model to rate difficulty
     1-5 in a single extra call, route <=3 to cheap and >=4 to capable (no mid tier - this
     matches v1's actual two-tier architecture, the real ablation baseline for "trained
     calibrator vs. prompted LLM guess").
  3. always_cheap - every prompt answered by the cheap tier only.
  4. always_capable - every prompt answered by the capable tier only (the "no routing"
     baseline, same role as v1's baseline.csv).

Calls are shared/reused where the policies overlap (e.g. always_cheap reuses the same
draft call the calibrated router uses) to avoid paying for the same answer twice.

Output: results/eval_report.md (accuracy w/ Wilson CI, cost w/ bootstrap CI, ablation
table), results/pareto_frontier.png (cost vs accuracy per policy).
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import wilson_interval, bootstrap_ci

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.providers import openai as openai_provider, anthropic as anthropic_provider, gemini
from app.providers.base import ProviderError
from app.router.decision import decide_tier
from app.router.calibrator import get_calibrator
from app.router.features import build_feature_vector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

CHEAP_MODEL = os.environ.get("CHEAP_MODEL", "gpt-5.4-nano")
MID_MODEL = os.environ.get("MID_MODEL", "gemini-3.1-flash-lite")
CAPABLE_MODEL = os.environ.get("CAPABLE_MODEL", "claude-sonnet-5")
ERROR_BUDGET = float(os.environ.get("ERROR_BUDGET", "0.05"))

with open(CONFIG_DIR / "pricing.json") as f:
    PRICING = {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(model)
    if not rates:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input_cost_per_1m"] + (output_tokens / 1_000_000) * rates[
        "output_cost_per_1m"
    ]


def evaluate_correct(answer: str, ground_truth: str) -> bool:
    if not answer or not ground_truth:
        return False
    return ground_truth.strip().lower() in answer.strip().lower()


async def call_with_retry(coro_fn, *args, retries=3, base_delay=3, **kwargs):
    delay = base_delay
    for attempt in range(retries):
        try:
            return await coro_fn(*args, **kwargs)
        except ProviderError as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2


async def classify_difficulty(prompt: str) -> int:
    """v1's naive classifier: ask the cheap tier to rate 1-5, parse the digit."""
    resp = await call_with_retry(
        openai_provider.complete,
        f"Rate difficulty 1-5, respond with only the digit for this prompt: {prompt}",
        CHEAP_MODEL,
    )
    import re

    match = re.search(r"\d", resp.text)
    return int(match.group()) if match else 3, resp


async def evaluate_prompt(record: dict, embedder, hard_cluster_centroid) -> dict:
    prompt = record["prompt"]
    ground_truth = record["ground_truth"]
    result = {"id": record["id"], "source": record["source"]}

    # Shared draft call (cheap tier) - reused by calibrated_router (if accepted) and always_cheap
    draft = await call_with_retry(openai_provider.complete, prompt, CHEAP_MODEL, logprobs=True)
    draft_cost = compute_cost(CHEAP_MODEL, draft.input_tokens, draft.output_tokens)
    draft_correct = evaluate_correct(draft.text, ground_truth)

    result["always_cheap"] = {
        "correct": draft_correct,
        "cost": draft_cost,
        "latency_ms": draft.latency_ms,
    }

    # Calibrated router policy
    prompt_embedding = embedder.encode(prompt) if embedder is not None else None
    features = build_feature_vector(
        draft_text=draft.text,
        draft_mean_logprob=draft.mean_logprob,
        sample_texts=[],  # matches serve-time behavior - see module docstring
        prompt_embedding=prompt_embedding,
        hard_cluster_centroid=hard_cluster_centroid,
    )
    p_correct = get_calibrator().predict_p_correct(features)
    decision = decide_tier(p_correct, ERROR_BUDGET)

    # Always compute the mid-tier outcome too (not just when the live decision escalates)
    # so results/eval_raw.json lets scripts/budget_sweep.py recompute the calibrated_router
    # policy at ANY error budget afterward with zero additional API calls - a real
    # cost-accuracy tradeoff curve instead of hand-picking one threshold and rerunning
    # per candidate value. This is the standard practice the cascade-routing literature
    # describes: treat the deferral threshold as a hyperparameter tuned on held-out data,
    # and report the tradeoff curve rather than a single cherry-picked operating point.
    mid = await call_with_retry(gemini.complete, prompt, MID_MODEL)
    mid_cost = compute_cost(MID_MODEL, mid.input_tokens, mid.output_tokens)
    mid_outcome = {
        "correct": evaluate_correct(mid.text, ground_truth),
        "cost": mid_cost,
        "latency_ms": mid.latency_ms,
    }

    result["_raw"] = {
        "p_correct_cheap": p_correct,
        "draft": {"correct": draft_correct, "cost": draft_cost, "latency_ms": draft.latency_ms},
        "mid": mid_outcome,
    }

    if decision.chosen_tier == "cheap":
        result["calibrated_router"] = {
            "correct": draft_correct,
            "cost": draft_cost,
            "latency_ms": draft.latency_ms,
            "tier": "cheap",
        }
    else:
        result["calibrated_router"] = {
            "correct": mid_outcome["correct"],
            "cost": draft_cost + mid_cost,  # includes the draft call spent on the (rejected) cheap attempt
            "latency_ms": draft.latency_ms + mid.latency_ms,
            "tier": "mid",
        }

    # Naive v1-style classifier policy (separate classifier call, cheap vs capable only)
    difficulty, classifier_resp = await classify_difficulty(prompt)
    classifier_cost = compute_cost(CHEAP_MODEL, classifier_resp.input_tokens, classifier_resp.output_tokens)
    if difficulty <= 3:
        result["naive_classifier"] = {
            "correct": draft_correct,
            "cost": classifier_cost + draft_cost,
            "latency_ms": classifier_resp.latency_ms + draft.latency_ms,
            "tier": "cheap",
        }
    else:
        capable_for_naive = await call_with_retry(anthropic_provider.complete, prompt, CAPABLE_MODEL)
        cap_cost = compute_cost(CAPABLE_MODEL, capable_for_naive.input_tokens, capable_for_naive.output_tokens)
        result["naive_classifier"] = {
            "correct": evaluate_correct(capable_for_naive.text, ground_truth),
            "cost": classifier_cost + cap_cost,
            "latency_ms": classifier_resp.latency_ms + capable_for_naive.latency_ms,
            "tier": "capable",
        }
        result["_naive_capable_reused"] = capable_for_naive  # reuse for always_capable if difficulty was also "hard"

    # Always-capable policy (independent call - every prompt goes to the capable tier)
    if "_naive_capable_reused" in result:
        cap = result.pop("_naive_capable_reused")
    else:
        cap = await call_with_retry(anthropic_provider.complete, prompt, CAPABLE_MODEL)
    cap_cost = compute_cost(CAPABLE_MODEL, cap.input_tokens, cap.output_tokens)
    result["always_capable"] = {
        "correct": evaluate_correct(cap.text, ground_truth),
        "cost": cap_cost,
        "latency_ms": cap.latency_ms,
    }

    return result


def summarize_policy(name: str, rows: list[dict]) -> dict:
    correct = [r[name]["correct"] for r in rows]
    costs = [r[name]["cost"] for r in rows]
    latencies = [r[name]["latency_ms"] for r in rows]
    n_correct = sum(correct)
    n = len(rows)
    acc_lo, acc_hi = wilson_interval(n_correct, n)
    cost_point, cost_lo, cost_hi = bootstrap_ci(costs, statistic=np.sum, n_resamples=3000)
    return {
        "policy": name,
        "n": n,
        "accuracy": n_correct / n,
        "accuracy_ci": (acc_lo, acc_hi),
        "total_cost": cost_point,
        "total_cost_ci": (cost_lo, cost_hi),
        "avg_latency_ms": float(np.mean(latencies)),
    }


async def main():
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    centroid_path = Path(__file__).resolve().parent.parent / "models" / "hard_cluster_centroid.npy"
    hard_cluster_centroid = np.load(centroid_path) if centroid_path.exists() else None

    records = []
    with open(DATA_DIR / "eval_holdout.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    if os.environ.get("TEST_MODE", "false").lower() == "true":
        print("TEST_MODE enabled. Running on 5 prompts only.")
        records = records[:5]

    print(f"Evaluating {len(records)} held-out prompts across 4 policies...")
    rows = []
    for i, record in enumerate(records):
        print(f"[{i + 1}/{len(records)}] {record['id']}...")
        try:
            rows.append(await evaluate_prompt(record, embedder, hard_cluster_centroid))
        except ProviderError as e:
            print(f"  FAILED {record['id']}: {e} (skipping this prompt for all policies)")
        await asyncio.sleep(0.5)

    policies = ["calibrated_router", "naive_classifier", "always_cheap", "always_capable"]
    summaries = {p: summarize_policy(p, rows) for p in policies}

    RESULTS_DIR.mkdir(exist_ok=True)

    raw_path = RESULTS_DIR / "eval_raw.json"
    with open(raw_path, "w") as f:
        json.dump([{"id": r["id"], **r["_raw"]} for r in rows], f, indent=2)
    print(f"Wrote raw per-prompt data to {raw_path} (used by scripts/budget_sweep.py)")
    report_path = RESULTS_DIR / "eval_report.md"
    with open(report_path, "w") as f:
        f.write("# Held-Out Evaluation Report\n\n")
        f.write(f"Evaluated on {len(rows)} prompts from `data/eval_holdout.jsonl` ")
        f.write("(disjoint from calibration_train.jsonl - never seen during calibrator training).\n\n")
        f.write("| Policy | Accuracy (95% CI) | Total Cost (95% CI) | Avg Latency (ms) |\n")
        f.write("| --- | --- | --- | --- |\n")
        for p in policies:
            s = summaries[p]
            f.write(
                f"| {p} | {s['accuracy']:.3f} ({s['accuracy_ci'][0]:.3f}-{s['accuracy_ci'][1]:.3f}) "
                f"| ${s['total_cost']:.5f} (${s['total_cost_ci'][0]:.5f}-${s['total_cost_ci'][1]:.5f}) "
                f"| {s['avg_latency_ms']:.0f} |\n"
            )

        f.write("\n## Ablation: calibrated router vs. naive prompted classifier\n\n")
        cr, nc = summaries["calibrated_router"], summaries["naive_classifier"]
        f.write(
            f"Calibrated router: {cr['accuracy']:.3f} accuracy, ${cr['total_cost']:.5f} total cost.\n\n"
            f"Naive classifier (v1-style): {nc['accuracy']:.3f} accuracy, ${nc['total_cost']:.5f} total cost.\n\n"
        )
        acc_delta = cr["accuracy"] - nc["accuracy"]
        cost_delta = cr["total_cost"] - nc["total_cost"]
        f.write(
            f"Delta: {acc_delta:+.3f} accuracy, ${cost_delta:+.5f} cost "
            f"(point estimates on n={len(rows)} - given the confidence intervals above, "
            "treat this delta as suggestive, not a settled result, at this sample size).\n"
        )

    # Pareto frontier: cost vs accuracy
    fig, ax = plt.subplots(figsize=(7, 6))
    for p in policies:
        s = summaries[p]
        ax.errorbar(
            s["total_cost"],
            s["accuracy"],
            yerr=[[s["accuracy"] - s["accuracy_ci"][0]], [s["accuracy_ci"][1] - s["accuracy"]]],
            xerr=[[s["total_cost"] - s["total_cost_ci"][0]], [s["total_cost_ci"][1] - s["total_cost"]]],
            fmt="o",
            markersize=10,
            capsize=4,
            label=p,
        )
    ax.set_xlabel("Total Cost ($)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Cost-Accuracy Pareto Frontier (n={len(rows)} held-out prompts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pareto_frontier.png")

    print(f"Wrote {report_path} and {RESULTS_DIR / 'pareto_frontier.png'}")
    for p in policies:
        s = summaries[p]
        print(f"{p}: acc={s['accuracy']:.3f} cost=${s['total_cost']:.5f} latency={s['avg_latency_ms']:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
