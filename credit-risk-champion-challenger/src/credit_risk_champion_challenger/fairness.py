"""Decision-level group checks for a default-risk model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_decision_report(
    y_true: np.ndarray,
    probability: np.ndarray,
    group: np.ndarray,
    *,
    decline_threshold: float,
) -> pd.DataFrame:
    """Summarize approval and error rates by protected group."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    groups = np.asarray(group)
    if not (len(y) == len(p) == len(groups)):
        raise ValueError("All inputs must have the same length")
    approved = p < decline_threshold
    rows = []
    for value in pd.unique(groups):
        mask = groups == value
        defaults = mask & (y == 1)
        non_defaults = mask & (y == 0)
        rows.append(
            {
                "group": value,
                "n": int(mask.sum()),
                "default_rate": float(y[mask].mean()),
                "approval_rate": float(approved[mask].mean()),
                "default_catch_rate": (
                    float((~approved[defaults]).mean()) if defaults.any() else np.nan
                ),
                "good_customer_decline_rate": float((~approved[non_defaults]).mean())
                if non_defaults.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("group")


def fairness_summary(report: pd.DataFrame) -> dict[str, float]:
    """Compute compact gaps from :func:`group_decision_report`."""
    approval = report["approval_rate"].dropna()
    catch = report["default_catch_rate"].dropna()
    false_decline = report["good_customer_decline_rate"].dropna()
    return {
        "disparate_impact": float(approval.min() / approval.max()) if approval.max() else np.nan,
        "approval_rate_gap": float(approval.max() - approval.min()),
        "default_catch_rate_gap": float(catch.max() - catch.min()),
        "false_decline_rate_gap": float(false_decline.max() - false_decline.min()),
    }
