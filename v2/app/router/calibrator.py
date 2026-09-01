"""Loads the trained calibration model and exposes a single predict_p_correct(features) call.

The model itself is trained offline by scripts/train_calibrator.py (logistic regression
or gradient-boosted trees over the FeatureVector fields) and saved to models/calibrator.pkl.
This module is intentionally a thin wrapper so the service doesn't need scikit-learn's
training machinery loaded at request time beyond predict_proba.
"""
import pickle
from pathlib import Path

import numpy as np

from .features import FeatureVector

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "calibrator.pkl"


class Calibrator:
    def __init__(self, model_path: Path = _DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"No trained calibrator at {self.model_path}. "
                    "Run scripts/train_calibrator.py first."
                )
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)

    def predict_p_correct(self, features: FeatureVector) -> float:
        self._ensure_loaded()
        x = features.as_array().reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        # class 1 = "correct" by convention (see train_calibrator.py labeling)
        classes = list(self._model.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])


_singleton: Calibrator | None = None


def get_calibrator() -> Calibrator:
    global _singleton
    if _singleton is None:
        _singleton = Calibrator()
    return _singleton
