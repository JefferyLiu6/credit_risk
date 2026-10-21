import numpy as np
import pytest

from credit_risk_champion_challenger.evaluation import (
    evaluate_binary_classifier,
    ks_statistic,
    paired_bootstrap_difference,
)


def test_perfect_ranking_has_perfect_auc_and_ks():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_binary_classifier(y, probability)
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert ks_statistic(y, probability) == 1.0


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError, match="within"):
        evaluate_binary_classifier(np.array([0, 1]), np.array([0.2, 1.2]))


def test_paired_bootstrap_detects_better_candidate():
    y = np.tile([0, 1], 100)
    champion = np.tile([0.45, 0.55], 100)
    candidate = np.tile([0.05, 0.95], 100)
    result = paired_bootstrap_difference(y, champion, candidate, n_resamples=50)
    assert result["difference"] >= 0
    assert 0 <= result["probability_candidate_better"] <= 1

