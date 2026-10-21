"""Explicit model-promotion gates, separate from model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PromotionPolicy:
    """Minimum validation evidence required before a locked-test review."""

    min_auc_lift: float = 0.005
    min_ap_lift: float = 0.0
    min_ks_lift: float = 0.0
    max_brier_increase: float = 0.002


def compare_candidate(
    champion: dict[str, float],
    candidate: dict[str, float],
    policy: PromotionPolicy | None = None,
) -> dict[str, object]:
    """Apply validation gates without claiming production promotion."""
    policy = policy or PromotionPolicy()
    required = {"roc_auc", "average_precision", "brier_score", "ks_statistic"}
    for label, metrics in (("champion", champion), ("candidate", candidate)):
        missing = required.difference(metrics)
        if missing:
            raise ValueError(f"{label} metrics missing: {sorted(missing)}")

    deltas = {
        "roc_auc": candidate["roc_auc"] - champion["roc_auc"],
        "average_precision": candidate["average_precision"] - champion["average_precision"],
        "brier_score": candidate["brier_score"] - champion["brier_score"],
        "ks_statistic": candidate["ks_statistic"] - champion["ks_statistic"],
    }
    gates = {
        "auc_lift": deltas["roc_auc"] >= policy.min_auc_lift,
        "average_precision": deltas["average_precision"] >= policy.min_ap_lift,
        "ks": deltas["ks_statistic"] >= policy.min_ks_lift,
        "calibration": deltas["brier_score"] <= policy.max_brier_increase,
    }
    return {
        "policy": asdict(policy),
        "deltas": deltas,
        "gates": gates,
        "validation_decision": (
            "advance_to_locked_test" if all(gates.values()) else "retain_champion"
        ),
        "production_status": "not_promoted",
    }
