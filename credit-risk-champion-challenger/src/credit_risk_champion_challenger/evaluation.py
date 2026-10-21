"""Credit-risk evaluation with paired uncertainty estimates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def _as_arrays(y_true: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and probability must be one-dimensional and equally sized")
    if len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("y_true must contain both binary classes")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    return y, p


def ks_statistic(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Maximum separation between defaulter and non-defaulter score CDFs."""
    y, p = _as_arrays(y_true, probability)
    false_positive_rate, true_positive_rate, _ = roc_curve(y, p)
    return float(np.max(true_positive_rate - false_positive_rate))


def evaluate_binary_classifier(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return discrimination and calibration metrics under stable names."""
    y, p = _as_arrays(y_true, probability)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ks_statistic": ks_statistic(y, p),
    }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    champion_probability: np.ndarray,
    candidate_probability: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap candidate-minus-champion lift on identical observations."""
    y, champion = _as_arrays(y_true, champion_probability)
    _, candidate = _as_arrays(y_true, candidate_probability)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    rng = np.random.default_rng(random_state)
    differences: list[float] = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size < 2:
            continue
        difference = metric(y[index], candidate[index]) - metric(y[index], champion[index])
        differences.append(float(difference))
    if not differences:
        raise ValueError("No valid bootstrap samples contained both classes")

    alpha = (1 - confidence) / 2
    low, high = np.quantile(differences, [alpha, 1 - alpha])
    observed = float(metric(y, candidate) - metric(y, champion))
    return {
        "difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_candidate_better": float(np.mean(np.asarray(differences) > 0)),
    }
