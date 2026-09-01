"""Statistical helpers shared by calibration training and evaluation.

No dependency on any model provider - pure numpy/scipy math, so this
module is fully unit-testable offline.
"""
import math
import numpy as np


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (accuracy with a small n)."""
    if n == 0:
        return (0.0, 0.0)
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def bootstrap_ci(values: list[float], statistic=np.mean, n_resamples: int = 5000,
                  confidence: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap confidence interval for an arbitrary statistic (e.g. mean cost).

    Returns (point_estimate, lower, upper).
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    point = float(statistic(arr))
    if len(arr) == 1:
        return (point, point, point)
    resample_stats = np.empty(n_resamples)
    n = len(arr)
    for i in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        resample_stats[i] = statistic(sample)
    alpha = 1 - confidence
    lo = float(np.percentile(resample_stats, 100 * alpha / 2))
    hi = float(np.percentile(resample_stats, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


def expected_calibration_error(probs: list[float], labels: list[int], n_bins: int = 10) -> float:
    """ECE: weighted average gap between predicted confidence and observed accuracy per bin."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def brier_score(probs: list[float], labels: list[int]) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def reliability_bins(probs: list[float], labels: list[int], n_bins: int = 10):
    """Returns (bin_centers, bin_confidence, bin_accuracy, bin_counts) for a reliability diagram."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, confs, accs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        centers.append((lo + hi) / 2)
        if mask.any():
            confs.append(float(probs[mask].mean()))
            accs.append(float(labels[mask].mean()))
        else:
            confs.append(float("nan"))
            accs.append(float("nan"))
        counts.append(int(mask.sum()))
    return centers, confs, accs, counts
