"""Population-stability monitoring for numeric and categorical inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-6


def drift_band(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "review"
    return "material"


def numeric_psi(reference: np.ndarray, current: np.ndarray, *, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        raise ValueError("PSI requires non-empty finite reference and current samples")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.all(current == reference[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_share = np.clip(ref_counts / ref_counts.sum(), EPSILON, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    reference = reference.fillna("__MISSING__").astype(str)
    current = current.fillna("__MISSING__").astype(str)
    categories = reference.unique().tolist() + current.unique().tolist()
    categories = pd.Index(categories).unique()
    ref_share = (
        reference.value_counts(normalize=True).reindex(categories, fill_value=0).clip(EPSILON)
    )
    cur_share = (
        current.value_counts(normalize=True).reindex(categories, fill_value=0).clip(EPSILON)
    )
    return float(((cur_share - ref_share) * np.log(cur_share / ref_share)).sum())


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    missing = set(numeric + categorical).difference(reference.columns).union(
        set(numeric + categorical).difference(current.columns)
    )
    if missing:
        raise ValueError(f"Missing monitored columns: {sorted(missing)}")
    rows = []
    for column in numeric:
        value = numeric_psi(reference[column], current[column])
        rows.append(
            {"feature": column, "kind": "numeric", "psi": value, "status": drift_band(value)}
        )
    for column in categorical:
        value = categorical_psi(reference[column], current[column])
        rows.append(
            {"feature": column, "kind": "categorical", "psi": value, "status": drift_band(value)}
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
