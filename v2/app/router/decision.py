"""Expected-cost-under-error-budget routing policy.

Given a calibrated P(correct) for the cheapest tier that could plausibly answer
a prompt, decide whether to accept that tier or escalate. This is a Neyman-Pearson
style constraint: accept the cheap answer unless its predicted error probability
exceeds the configured error budget epsilon, in which case escalate up the ladder
and re-evaluate (mid tier is assumed to have a lower error rate than cheap; capable
tier is treated as the trusted fallback with an assumed near-zero calibration error).
"""
from dataclasses import dataclass


@dataclass
class Tier:
    name: str
    input_cost_per_1m: float
    output_cost_per_1m: float


@dataclass
class RoutingDecision:
    chosen_tier: str
    # None means "not separately calibrated" (mid/capable when no dedicated score was
    # computed for them) - never fabricate a confidence number for a tier that wasn't
    # actually scored, since that value gets logged and would corrupt any later
    # calibration analysis built from production data.
    predicted_p_correct: float | None
    error_budget: float
    escalated: bool
    reason: str


def decide_tier(
    p_correct_cheap: float,
    error_budget: float,
    p_correct_mid: float | None = None,
) -> RoutingDecision:
    """Decide which tier to use given the cheap tier's calibrated P(correct).

    Policy: accept the cheap tier if its predicted error probability (1 - p_correct)
    is within the error budget; otherwise escalate exactly one rung, to mid. Only a
    single calibrator (trained on the cheap tier) currently exists, so mid is treated
    as a trusted-but-unscored escalation target by default - pass p_correct_mid if a
    mid-tier calibrator is ever trained, in which case mid's own budget check decides
    whether to accept it or escalate again to capable.
    """
    predicted_error_cheap = 1.0 - p_correct_cheap
    if predicted_error_cheap <= error_budget + 1e-9:
        return RoutingDecision(
            chosen_tier="cheap",
            predicted_p_correct=p_correct_cheap,
            error_budget=error_budget,
            escalated=False,
            reason=f"predicted error {predicted_error_cheap:.3f} <= budget {error_budget:.3f}",
        )

    if p_correct_mid is None:
        return RoutingDecision(
            chosen_tier="mid",
            predicted_p_correct=None,
            error_budget=error_budget,
            escalated=True,
            reason=f"cheap predicted error {predicted_error_cheap:.3f} > budget; no mid-tier calibrator available, escalating one rung",
        )

    predicted_error_mid = 1.0 - p_correct_mid
    if predicted_error_mid <= error_budget + 1e-9:
        return RoutingDecision(
            chosen_tier="mid",
            predicted_p_correct=p_correct_mid,
            error_budget=error_budget,
            escalated=True,
            reason=(
                f"cheap predicted error {predicted_error_cheap:.3f} > budget; "
                f"mid predicted error {predicted_error_mid:.3f} <= budget"
            ),
        )

    return RoutingDecision(
        chosen_tier="capable",
        predicted_p_correct=None,  # capable tier is the trusted top of the ladder, not separately calibrated
        error_budget=error_budget,
        escalated=True,
        reason=f"cheap and mid predicted error both exceeded budget {error_budget:.3f}",
    )
