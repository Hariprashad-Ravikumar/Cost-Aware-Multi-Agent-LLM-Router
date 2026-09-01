"""Sweeps ERROR_BUDGET across several values and recomputes the calibrated router's
accuracy/cost at each, using the raw per-prompt data already collected by run_eval.py
(results/eval_raw.json) - zero additional API calls.

This is the honest way to answer "does a looser error budget actually save cost": the
cascade-routing literature treats the deferral threshold as a hyperparameter tuned on
held-out data and reports the full tradeoff curve, rather than hand-picking one value
and re-running per candidate (see CASE_STUDY.md for citations). Every point on the
resulting curve is a real recomputation from real logged (p_correct, outcome) pairs -
nothing here is simulated or estimated.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import wilson_interval, bootstrap_ci

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.router.decision import decide_tier

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def main():
    raw_path = RESULTS_DIR / "eval_raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found - run scripts/run_eval.py first")

    with open(raw_path) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} prompts with cached (p_correct, draft, mid) outcomes.")

    sweep_results = []
    for budget in BUDGETS:
        correct, costs, latencies, escalated_count = [], [], [], 0
        for row in rows:
            decision = decide_tier(row["p_correct_cheap"], budget)
            if decision.chosen_tier == "cheap":
                outcome = row["draft"]
            else:
                outcome = row["mid"]
                escalated_count += 1
            correct.append(outcome["correct"])
            # Escalated cost includes the draft call already spent probing the cheap tier
            cost = row["draft"]["cost"] + (row["mid"]["cost"] if decision.chosen_tier != "cheap" else 0)
            costs.append(cost)
            latency = row["draft"]["latency_ms"] + (row["mid"]["latency_ms"] if decision.chosen_tier != "cheap" else 0)
            latencies.append(latency)

        n = len(rows)
        n_correct = sum(correct)
        acc_lo, acc_hi = wilson_interval(n_correct, n)
        cost_point, cost_lo, cost_hi = bootstrap_ci(costs, statistic=np.sum, n_resamples=3000)
        sweep_results.append(
            {
                "error_budget": budget,
                "accuracy": n_correct / n,
                "accuracy_ci": (acc_lo, acc_hi),
                "total_cost": cost_point,
                "total_cost_ci": (cost_lo, cost_hi),
                "avg_latency_ms": float(np.mean(latencies)),
                "escalation_rate": escalated_count / n,
            }
        )

    report_path = RESULTS_DIR / "budget_sweep.md"
    with open(report_path, "w") as f:
        f.write("# Error Budget Sweep\n\n")
        f.write(
            f"Recomputed from `results/eval_raw.json` ({len(rows)} held-out prompts) - "
            "every row below is a real recomputation from logged (calibrated P(correct), "
            "actual cheap/mid outcome) pairs, not a new API run. Threshold treated as a "
            "hyperparameter tuned on held-out data, per the cascade-routing literature "
            "(see CASE_STUDY.md).\n\n"
        )
        f.write("| Error Budget (ε) | Escalation Rate | Accuracy (95% CI) | Total Cost (95% CI) | Avg Latency (ms) |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for r in sweep_results:
            f.write(
                f"| {r['error_budget']:.2f} | {r['escalation_rate']:.1%} "
                f"| {r['accuracy']:.3f} ({r['accuracy_ci'][0]:.3f}-{r['accuracy_ci'][1]:.3f}) "
                f"| ${r['total_cost']:.5f} (${r['total_cost_ci'][0]:.5f}-${r['total_cost_ci'][1]:.5f}) "
                f"| {r['avg_latency_ms']:.0f} |\n"
            )

        # Compare best-cost budget against the always_capable baseline if eval_report.md exists
        eval_report_path = RESULTS_DIR / "eval_report.md"
        if eval_report_path.exists():
            f.write("\n## Comparison against fixed baselines\n\n")
            f.write("See `results/eval_report.md` for `always_cheap`, `naive_classifier`, and `always_capable` costs/accuracy at the original ε=0.05 run. ")
            f.write("Compare each row above against those fixed baselines directly - a budget only \"wins\" if it beats always_capable on cost without giving up meaningful accuracy.\n")

    # Plot: cost vs accuracy across budgets
    fig, ax = plt.subplots(figsize=(7, 6))
    costs_plot = [r["total_cost"] for r in sweep_results]
    accs_plot = [r["accuracy"] for r in sweep_results]
    ax.plot(costs_plot, accs_plot, "o-", color="tab:blue", label="calibrated_router (swept ε)")
    for r in sweep_results:
        ax.annotate(f"ε={r['error_budget']:.2f}", (r["total_cost"], r["accuracy"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Total Cost ($)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Calibrated Router: Cost-Accuracy Tradeoff Across Error Budgets (n={len(rows)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "budget_sweep.png")

    print(f"Wrote {report_path} and {RESULTS_DIR / 'budget_sweep.png'}")
    for r in sweep_results:
        print(
            f"ε={r['error_budget']:.2f}: escalation={r['escalation_rate']:.1%} "
            f"acc={r['accuracy']:.3f} cost=${r['total_cost']:.5f} latency={r['avg_latency_ms']:.0f}ms"
        )


if __name__ == "__main__":
    main()
