"""Feature extraction for the calibrator.

Deliberately decoupled from any provider API: callers gather a draft (temperature=0)
response and a few sampled (temperature>0) responses from the cheap tier elsewhere
(see scripts/collect_calibration_data.py and app/main.py), then hand them here.
This keeps the feature math itself unit-testable with fixtures instead of live calls.
"""
import math
import re
from dataclasses import dataclass

import numpy as np


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_final_answer(text: str) -> str:
    """Heuristic extraction of the "actual answer" from a free-form response, so
    self-consistency compares answers rather than prose wording (which varies between
    samples even when the underlying answer agrees). Tries, in order: a standalone
    multiple-choice letter (A-D) near the end, then the last standalone number, then
    falls back to the full normalized text.
    """
    normalized = _normalize_answer(text)
    tail = normalized[-200:]  # final-answer statements are almost always near the end

    letter_matches = re.findall(r"\b([a-d])\b", tail)
    if letter_matches and len(text) < 50:
        # short responses that are basically just a letter (e.g. MMLU "C.") - trust it
        return letter_matches[-1]

    number_matches = re.findall(r"-?\d[\d,]*\.?\d*", normalized)
    if number_matches:
        return number_matches[-1].replace(",", "")

    if letter_matches:
        return letter_matches[-1]

    return normalized


def self_consistency_dispersion(samples: list[str]) -> float:
    """1 - (fraction of samples matching the majority answer). 0 = fully consistent, ~1 = fully inconsistent.

    Compares extracted final answers (see _extract_final_answer), not raw response text -
    two verbose step-by-step explanations reaching the same number would otherwise look
    "inconsistent" purely from wording differences.
    """
    if not samples:
        return 0.0
    normalized = [_extract_final_answer(s) for s in samples]
    counts: dict[str, int] = {}
    for s in normalized:
        counts[s] = counts.get(s, 0) + 1
    majority_count = max(counts.values())
    return 1.0 - (majority_count / len(normalized))


def logprob_uncertainty(mean_logprob: float | None) -> float:
    """Maps a mean token logprob to a [0, 1]-ish uncertainty score via sigmoid on -logprob.

    Higher = more uncertain. Returns 0.5 (neutral) if the provider didn't expose logprobs,
    so the feature vector stays well-formed even when this signal is unavailable.
    """
    if mean_logprob is None:
        return 0.5
    # -mean_logprob is typically in [0, ~5] for confident-to-uncertain tokens; squash to (0, 1)
    return 1.0 / (1.0 + math.exp(-(-mean_logprob - 1.0)))


def embedding_distance_to_hard_cluster(prompt_embedding: np.ndarray, hard_cluster_centroid: np.ndarray | None) -> float:
    """Cosine distance (0=identical direction, 2=opposite) between the prompt and a centroid
    of embeddings for known-hard prompts. Returns 0.5 (neutral) if no centroid is available yet
    (e.g. before the hard-cluster reference set has been embedded)."""
    if hard_cluster_centroid is None:
        return 0.5
    a, b = prompt_embedding, hard_cluster_centroid
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.5
    cosine_sim = float(np.dot(a, b) / denom)
    return 1.0 - cosine_sim  # 0 = very close to hard cluster, 2 = opposite direction


@dataclass
class FeatureVector:
    logprob_uncertainty: float
    self_consistency_dispersion: float
    hard_cluster_distance: float
    response_length_chars: int

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.logprob_uncertainty,
                self.self_consistency_dispersion,
                self.hard_cluster_distance,
                self.response_length_chars,
            ],
            dtype=float,
        )

    FEATURE_NAMES = (
        "logprob_uncertainty",
        "self_consistency_dispersion",
        "hard_cluster_distance",
        "response_length_chars",
    )


def build_feature_vector(
    draft_text: str,
    draft_mean_logprob: float | None,
    sample_texts: list[str],
    prompt_embedding: np.ndarray | None,
    hard_cluster_centroid: np.ndarray | None,
) -> FeatureVector:
    return FeatureVector(
        logprob_uncertainty=logprob_uncertainty(draft_mean_logprob),
        self_consistency_dispersion=self_consistency_dispersion(sample_texts),
        hard_cluster_distance=(
            embedding_distance_to_hard_cluster(prompt_embedding, hard_cluster_centroid)
            if prompt_embedding is not None
            else 0.5
        ),
        response_length_chars=len(draft_text),
    )
