# Held-Out Evaluation Report

Evaluated on 50 prompts from `data/eval_holdout.jsonl` (disjoint from calibration_train.jsonl - never seen during calibrator training).

| Policy | Accuracy (95% CI) | Total Cost (95% CI) | Avg Latency (ms) |
| --- | --- | --- | --- |
| calibrated_router | 0.940 (0.838-0.979) | $0.00819 ($0.00647-$0.01007) | 1973 |
| naive_classifier | 0.840 (0.715-0.917) | $0.00626 ($0.00528-$0.00728) | 1796 |
| always_cheap | 0.840 (0.715-0.917) | $0.00490 ($0.00387-$0.00597) | 1067 |
| always_capable | 0.920 (0.812-0.968) | $0.09387 ($0.07199-$0.11680) | 2373 |

## Ablation: calibrated router vs. naive prompted classifier

Calibrated router: 0.940 accuracy, $0.00819 total cost.

Naive classifier (v1-style): 0.840 accuracy, $0.00626 total cost.

Delta: +0.100 accuracy, $+0.00193 cost (point estimates on n=50 - given the confidence intervals above, treat this delta as suggestive, not a settled result, at this sample size).
