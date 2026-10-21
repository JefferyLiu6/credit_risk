from credit_risk_champion_challenger.governance import (
    PromotionPolicy,
    compare_candidate,
)

CHAMPION = {
    "roc_auc": 0.72,
    "average_precision": 0.20,
    "brier_score": 0.07,
    "ks_statistic": 0.34,
}


def test_candidate_advances_when_every_gate_passes():
    candidate = {
        "roc_auc": 0.74,
        "average_precision": 0.22,
        "brier_score": 0.069,
        "ks_statistic": 0.36,
    }
    result = compare_candidate(CHAMPION, candidate)
    assert result["validation_decision"] == "advance_to_locked_test"
    assert result["production_status"] == "not_promoted"


def test_single_failed_gate_retains_champion():
    candidate = {**CHAMPION, "roc_auc": 0.73, "average_precision": 0.19}
    result = compare_candidate(CHAMPION, candidate, PromotionPolicy(min_auc_lift=0.005))
    assert result["validation_decision"] == "retain_champion"
    assert result["gates"]["average_precision"] is False
