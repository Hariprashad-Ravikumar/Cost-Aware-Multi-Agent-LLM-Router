# Error Budget Sweep

Recomputed from `results/eval_raw.json` (50 held-out prompts) - every row below is a real recomputation from logged (calibrated P(correct), actual cheap/mid outcome) pairs, not a new API run. Threshold treated as a hyperparameter tuned on held-out data, per the cascade-routing literature (see CASE_STUDY.md).

| Error Budget (ε) | Escalation Rate | Accuracy (95% CI) | Total Cost (95% CI) | Avg Latency (ms) |
| --- | --- | --- | --- | --- |
| 0.05 | 70.0% | 0.940 (0.838-0.979) | $0.00819 ($0.00647-$0.01007) | 1973 |
| 0.10 | 52.0% | 0.960 (0.865-0.989) | $0.00593 ($0.00496-$0.00693) | 1433 |
| 0.15 | 50.0% | 0.960 (0.865-0.989) | $0.00569 ($0.00483-$0.00660) | 1403 |
| 0.20 | 32.0% | 0.960 (0.865-0.989) | $0.00545 ($0.00452-$0.00643) | 1279 |
| 0.30 | 10.0% | 0.860 (0.738-0.930) | $0.00505 ($0.00407-$0.00610) | 1133 |
| 0.50 | 8.0% | 0.860 (0.738-0.930) | $0.00501 ($0.00401-$0.00606) | 1120 |

## Comparison against fixed baselines

See `results/eval_report.md` for `always_cheap`, `naive_classifier`, and `always_capable` costs/accuracy at the original ε=0.05 run. Compare each row above against those fixed baselines directly - a budget only "wins" if it beats always_capable on cost without giving up meaningful accuracy.
