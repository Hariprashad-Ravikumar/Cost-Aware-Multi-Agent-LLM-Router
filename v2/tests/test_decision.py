import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.router.decision import decide_tier


def test_accepts_cheap_when_within_budget():
    d = decide_tier(p_correct_cheap=0.98, error_budget=0.05)
    assert d.chosen_tier == "cheap"
    assert not d.escalated


def test_escalates_to_mid_when_no_mid_calibration_available():
    d = decide_tier(p_correct_cheap=0.5, error_budget=0.05)
    assert d.chosen_tier == "mid"
    assert d.escalated
    assert d.predicted_p_correct is None  # never fabricate a confidence for an unscored tier


def test_escalates_to_mid_when_mid_within_budget():
    d = decide_tier(p_correct_cheap=0.5, error_budget=0.05, p_correct_mid=0.97)
    assert d.chosen_tier == "mid"
    assert d.escalated


def test_escalates_to_capable_when_mid_also_fails_budget():
    d = decide_tier(p_correct_cheap=0.5, error_budget=0.05, p_correct_mid=0.5)
    assert d.chosen_tier == "capable"
    assert d.predicted_p_correct is None


def test_boundary_exactly_at_budget_is_accepted():
    d = decide_tier(p_correct_cheap=0.95, error_budget=0.05)
    assert d.chosen_tier == "cheap"
