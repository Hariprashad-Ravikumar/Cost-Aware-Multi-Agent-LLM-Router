"""Offline calibration data collection.

Unlike the serve-time router (app/main.py), this script CAN afford self-consistency
sampling (k=3 calls per prompt at temperature>0.7) because it runs once, offline,
against the training split - not on every live request. This is the one place in
the project where self_consistency_dispersion is a real, measured feature rather
than the neutral default the service falls back to at serve time (see
app/main.py's module docstring for that documented train/serve skew).

For each prompt in data/calibration_train.jsonl:
  1. One temperature=0 "draft" call to the cheap tier (also the eval-time answer).
  2. Three temperature=0.7 samples for self-consistency dispersion.
  3. A local sentence-transformers embedding, compared against a hard-cluster
     centroid built from data/hard_ood.jsonl (computed once, cached to disk).
  4. Correctness label via simple substring match against ground_truth (same
     evaluation convention as v1's evaluate_answer).

Output: data/calibration_features.jsonl - one row per prompt with the full
FeatureVector plus the binary label, ready for train_calibrator.py.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.providers import openai as openai_provider
from app.providers.base import ProviderError
from app.router.features import build_feature_vector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CHEAP_MODEL = os.environ.get("CHEAP_MODEL", "gpt-5.4-nano")
N_SAMPLES = 3
MAX_RETRIES = 4
# OpenAI's rate limit for this model/key is 200,000 tokens/minute (verified via the
# x-ratelimit-limit-tokens response header) - 25x Groq's free-tier cap that originally
# motivated this pacing logic. At our volume (~100 prompts x 4 calls x ~500 tokens),
# this limit shouldn't bind at all, but the adaptive pacing is kept as a real safety
# net rather than assumed away - it costs nothing when there's headroom (the "only
# sleep when close to the cap" branch just won't fire) and protects against a burst
# still being possible in principle. Pace adaptively off the real x-ratelimit-remaining-tokens
# / x-ratelimit-reset-tokens headers Groq returns on every response: only sleep when
# we're actually close to the budget, and sleep exactly as long as the server says the
# window needs to reset - not a guess. This is a token-bucket limiter driven by the
# provider's own signal rather than fixed pacing.
TOKEN_SAFETY_MARGIN = 1500  # assume the next call could use up to ~this many tokens
FALLBACK_DELAY_SECONDS = 0.5  # used only when no rate-limit signal was returned


def evaluate_correct(answer: str, ground_truth: str) -> bool:
    if not answer or not ground_truth:
        return False
    return ground_truth.strip().lower() in answer.strip().lower()


def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def build_hard_cluster_centroid(embedder) -> np.ndarray:
    centroid_path = MODELS_DIR / "hard_cluster_centroid.npy"
    if centroid_path.exists():
        return np.load(centroid_path)

    prompts = []
    with open(DATA_DIR / "hard_ood.jsonl") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])

    embeddings = embedder.encode(prompts)
    centroid = embeddings.mean(axis=0)
    MODELS_DIR.mkdir(exist_ok=True)
    np.save(centroid_path, centroid)
    print(f"Built and cached hard-cluster centroid from {len(prompts)} prompts -> {centroid_path}")
    return centroid


async def adaptive_pace(response) -> None:
    """Sleep only as much as the real rate-limit headers say is needed before the
    next call is safe, instead of a fixed guess."""
    remaining = response.rate_limit_remaining_tokens
    reset_seconds = response.rate_limit_reset_tokens_seconds
    if remaining is None or reset_seconds is None:
        await asyncio.sleep(FALLBACK_DELAY_SECONDS)
        return
    if remaining < TOKEN_SAFETY_MARGIN:
        wait = reset_seconds + 0.25  # small margin over the server's own reset estimate
        print(f"  [rate limiter] {remaining} tokens left, waiting {wait:.1f}s for reset...")
        await asyncio.sleep(wait)
    else:
        await asyncio.sleep(FALLBACK_DELAY_SECONDS)


async def call_with_manual_retry(coro_fn, *args, **kwargs):
    delay = 8  # start well above the sub-second reset seen when under the cap, since a
    # 429 here means we're over the per-minute token budget, not a one-off blip
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except ProviderError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  retrying after error: {e} (attempt {attempt + 1}/{MAX_RETRIES})")
            await asyncio.sleep(delay)
            delay *= 2


async def process_prompt(record: dict, embedder, hard_cluster_centroid: np.ndarray) -> dict | None:
    prompt = record["prompt"]
    ground_truth = record["ground_truth"]

    try:
        # gpt-5.4-nano (unlike Groq's openai/gpt-oss-120b) actually supports logprobs -
        # verified directly against the API - so logprob_uncertainty is now a real
        # measured feature instead of the neutral-default placeholder v2 shipped with initially.
        draft = await call_with_manual_retry(openai_provider.complete, prompt, CHEAP_MODEL, logprobs=True)
    except ProviderError as e:
        print(f"  FAILED (draft) {record['id']}: {e}")
        return None
    await adaptive_pace(draft)

    sample_texts = []
    for _ in range(N_SAMPLES):
        try:
            sample = await call_with_manual_retry(openai_provider.complete, prompt, CHEAP_MODEL, temperature=0.8)
            sample_texts.append(sample.text)
            await adaptive_pace(sample)
        except ProviderError as e:
            print(f"  sample call failed for {record['id']}: {e} (continuing with fewer samples)")

    prompt_embedding = embedder.encode(prompt)
    features = build_feature_vector(
        draft_text=draft.text,
        draft_mean_logprob=draft.mean_logprob,
        sample_texts=sample_texts,
        prompt_embedding=prompt_embedding,
        hard_cluster_centroid=hard_cluster_centroid,
    )
    label = int(evaluate_correct(draft.text, ground_truth))

    return {
        "id": record["id"],
        "source": record["source"],
        "features": features.as_array().tolist(),
        "label": label,
    }


async def main():
    embedder = get_embedder()
    hard_cluster_centroid = build_hard_cluster_centroid(embedder)

    records = []
    with open(DATA_DIR / "calibration_train.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    if os.environ.get("TEST_MODE", "false").lower() == "true":
        print("TEST_MODE enabled. Running on 5 prompts only.")
        records = records[:5]

    out_path = DATA_DIR / "calibration_features.jsonl"
    already_done = set()
    if os.environ.get("RESUME", "false").lower() == "true" and out_path.exists():
        with open(out_path) as f:
            already_done = {json.loads(line)["id"] for line in f if line.strip()}
        print(f"RESUME enabled: {len(already_done)} prompts already collected, skipping them.")
        records = [r for r in records if r["id"] not in already_done]

    print(f"Collecting calibration features for {len(records)} prompts against {CHEAP_MODEL}...")
    mode = "a" if already_done else "w"
    n_ok, n_failed = 0, 0
    with open(out_path, mode) as out_f:
        for i, record in enumerate(records):
            print(f"[{i + 1}/{len(records)}] {record['id']}...")
            result = await process_prompt(record, embedder, hard_cluster_centroid)
            if result is None:
                n_failed += 1
                continue
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            n_ok += 1

    print(f"Done. {n_ok} succeeded, {n_failed} failed. Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
