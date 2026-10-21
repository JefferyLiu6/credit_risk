"""Model loading and scoring, independent of the HTTP framework."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .constants import MODEL_FEATURES


def risk_band(probability: float) -> str:
    if probability < 0.05:
        return "low"
    if probability < 0.10:
        return "moderate"
    if probability < 0.20:
        return "high"
    return "very_high"


class ModelService:
    def __init__(self, model_path: str | Path, *, model_name: str = "logistic_champion") -> None:
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.pipeline = joblib.load(self.model_path)

    def score(self, application: dict[str, object]) -> dict[str, object]:
        missing = set(MODEL_FEATURES).difference(application)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        row = {
            feature: np.nan if application[feature] is None else application[feature]
            for feature in MODEL_FEATURES
        }
        frame = pd.DataFrame([row])
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        return {
            "default_probability": round(probability, 6),
            "risk_band": risk_band(probability),
            "model": self.model_name,
        }
