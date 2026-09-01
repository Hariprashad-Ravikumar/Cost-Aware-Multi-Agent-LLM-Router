import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from stats_utils import wilson_interval, bootstrap_ci, expected_calibration_error, brier_score


def test_wilson_interval_bounds_contain_point_estimate():
    lo, hi = wilson_interval(89, 100)
    assert lo < 0.89 < hi


def test_wilson_interval_zero_n():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_bootstrap_ci_single_value():
    point, lo, hi = bootstrap_ci([5.0])
    assert point == lo == hi == 5.0


def test_bootstrap_ci_contains_true_mean_for_low_variance_data():
    point, lo, hi = bootstrap_ci([10.0, 10.1, 9.9, 10.0, 10.05], n_resamples=2000)
    assert lo <= point <= hi


def test_perfect_calibration_has_zero_ece():
    probs = [1.0, 1.0, 0.0, 0.0]
    labels = [1, 1, 0, 0]
    assert expected_calibration_error(probs, labels) == 0.0


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_score_worst_case_is_one():
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0
