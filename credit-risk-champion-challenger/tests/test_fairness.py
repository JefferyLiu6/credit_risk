import numpy as np

from credit_risk_champion_challenger.fairness import fairness_summary, group_decision_report


def test_group_report_uses_approval_as_probability_below_threshold():
    y = np.array([0, 1, 0, 1])
    probability = np.array([0.1, 0.9, 0.2, 0.8])
    group = np.array(["a", "a", "b", "b"])
    report = group_decision_report(y, probability, group, decline_threshold=0.5)
    assert report.loc["a", "approval_rate"] == 0.5
    assert report.loc["b", "default_catch_rate"] == 1.0
    assert fairness_summary(report)["disparate_impact"] == 1.0

