# Calibrator Training Report

- Training rows: 80
- Validation rows: 20
- Feature names: logprob_uncertainty, self_consistency_dispersion, hard_cluster_distance, response_length_chars
- Validation accuracy (0.5 threshold): 0.800
- Expected Calibration Error (5 bins): 0.1671
- Brier score: 0.1521

See `calibration_curve.png` for the reliability diagram.

**Caveat:** this validation split is drawn from the same calibration_train.jsonl distribution the model fit on, just held out at fit time - it is a sanity check, not the real generalization test. `scripts/run_eval.py` against the fully disjoint `eval_holdout.jsonl` (never touched here) is the test that actually matters.
