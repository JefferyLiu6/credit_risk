"""Build the validation-stage champion/challenger decision report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluation import evaluate_binary_classifier, paired_bootstrap_difference
from .governance import compare_candidate

PROBABILITY_COLUMNS = {
    "Logistic Regression": "logistic_probability",
    "XGBoost": "xgboost_probability",
    "LightGBM": "lightgbm_probability",
}


def build_validation_report(predictions_path: str | Path, *, n_resamples: int = 500) -> dict:
    predictions = pd.read_csv(predictions_path)
    required = {"TARGET", *PROBABILITY_COLUMNS.values()}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file missing: {sorted(missing)}")

    y = predictions["TARGET"].to_numpy()
    metrics = {
        name: evaluate_binary_classifier(y, predictions[column].to_numpy())
        for name, column in PROBABILITY_COLUMNS.items()
    }
    champion = metrics["Logistic Regression"]
    candidates = {}
    for name in ("XGBoost", "LightGBM"):
        candidates[name] = {
            "metrics": metrics[name],
            "governance": compare_candidate(champion, metrics[name]),
            "auc_uncertainty": paired_bootstrap_difference(
                y,
                predictions[PROBABILITY_COLUMNS["Logistic Regression"]].to_numpy(),
                predictions[PROBABILITY_COLUMNS[name]].to_numpy(),
                n_resamples=n_resamples,
            ),
        }
    ranked = sorted(candidates, key=lambda name: metrics[name]["roc_auc"], reverse=True)
    return {
        "stage": "validation",
        "incumbent_champion": "Logistic Regression",
        "recommended_candidate": ranked[0],
        "test_set_accessed": False,
        "models": metrics,
        "candidates": candidates,
        "decision_note": (
            "A validation winner may advance to the locked test, but the incumbent remains "
            "champion until test, stability, fairness, and review gates pass."
        ),
    }


def write_validation_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

