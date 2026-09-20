"""
Business decision layer for the Telco Customer Churn solution.

The ML model predicts churn risk. This module converts the probability into
an operational risk band and a retention action. Thresholds are configurable
and should be calibrated against intervention economics in production.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionDecision:
    risk_band: str
    recommended_action: str


def classify_risk(churn_probability: float) -> RetentionDecision:
    if not 0.0 <= churn_probability <= 1.0:
        raise ValueError("churn_probability must be between 0 and 1")

    if churn_probability >= 0.75:
        return RetentionDecision(
            "High Risk",
            "Prioritized retention outreach"
        )
    if churn_probability >= 0.50:
        return RetentionDecision(
            "Medium Risk",
            "Proactive digital engagement and targeted retention"
        )
    return RetentionDecision(
        "Low Risk",
        "Normal lifecycle engagement"
    )


def calculate_priority_score(
    churn_probability: float,
    monthly_charge: float
) -> float:
    """
    Simple prototype prioritization heuristic:
    churn probability × monthly charge.

    This is not a causal revenue-retention estimate.
    """
    if churn_probability < 0 or churn_probability > 1:
        raise ValueError("churn_probability must be between 0 and 1")
    if monthly_charge < 0:
        raise ValueError("monthly_charge cannot be negative")
    return churn_probability * monthly_charge
