"""Fits the calibration model: FeatureVector -> P(cheap-tier answer is correct).

Splits calibration_features.jsonl into an internal 80/20 train/validation split
(fixed seed) so the calibration report (ECE, Brier score, reliability diagram) is
measured on data the model did not fit on - this is a *preliminary* generalization
check; the real held-out test is run_eval.py against eval_holdout.jsonl later,
processed live through the router rather than through this offline feature file.

Model: logistic regression behind a StandardScaler (features have very different
scales - uncertainty scores in [0,1] vs. response_length_chars in the hundreds).
Chosen over gradient-boosted trees for this dataset size (~100 rows): fewer
parameters to overfit, and it stays monotonic in each feature, which matches the
project's actual assumption (higher uncertainty -> higher predicted error) instead
of letting a tree carve out noise-driven, hard-to-explain splits on a small sample.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import expected_calibration_error, brier_score, reliability_bins

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.router.features import FeatureVector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42


def main():
    rows = []
    with open(DATA_DIR / "calibration_features.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))

    if len(rows) < 20:
        print(
            f"WARNING: only {len(rows)} labeled rows available. Calibration metrics "
            "on this few samples are noisy - treat this as a smoke test, not a final report."
        )

    X = np.array([r["features"] for r in rows])
    y = np.array([r["label"] for r in rows])

    print(f"Loaded {len(rows)} rows. Label balance: {y.mean():.2%} correct.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y if len(set(y)) > 1 else None
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    model.fit(X_train, y_train)

    classes = list(model.classes_)
    val_probs = model.predict_proba(X_val)[:, classes.index(1)] if 1 in classes else model.predict_proba(X_val)[:, -1]

    ece = expected_calibration_error(val_probs.tolist(), y_val.tolist(), n_bins=5)
    brier = brier_score(val_probs.tolist(), y_val.tolist())
    accuracy = float((model.predict(X_val) == y_val).mean())

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    import pickle

    with open(MODELS_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(model, f)

    centers, confs, accs, counts = reliability_bins(val_probs.tolist(), y_val.tolist(), n_bins=5)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    valid = [(c, a) for c, a in zip(confs, accs) if c == c]  # drop NaN (empty) bins
    if valid:
        cs, as_ = zip(*valid)
        ax.plot(cs, as_, "o-", label="calibrator (validation split)")
    ax.set_xlabel("Predicted P(correct)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability Diagram (ECE={ece:.3f}, Brier={brier:.3f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "calibration_curve.png")

    report_path = RESULTS_DIR / "calibration_report.md"
    with open(report_path, "w") as f:
        f.write("# Calibrator Training Report\n\n")
        f.write(f"- Training rows: {len(X_train)}\n")
        f.write(f"- Validation rows: {len(X_val)}\n")
        f.write(f"- Feature names: {', '.join(FeatureVector.FEATURE_NAMES)}\n")
        f.write(f"- Validation accuracy (0.5 threshold): {accuracy:.3f}\n")
        f.write(f"- Expected Calibration Error (5 bins): {ece:.4f}\n")
        f.write(f"- Brier score: {brier:.4f}\n\n")
        f.write("See `calibration_curve.png` for the reliability diagram.\n\n")
        f.write(
            "**Caveat:** this validation split is drawn from the same calibration_train.jsonl "
            "distribution the model fit on, just held out at fit time - it is a sanity check, "
            "not the real generalization test. `scripts/run_eval.py` against the fully disjoint "
            "`eval_holdout.jsonl` (never touched here) is the test that actually matters.\n"
        )

    print(f"Validation accuracy: {accuracy:.3f} | ECE: {ece:.4f} | Brier: {brier:.4f}")
    print(f"Saved model to {MODELS_DIR / 'calibrator.pkl'}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
