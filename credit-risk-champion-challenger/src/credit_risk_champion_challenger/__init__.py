"""Governance-first champion/challenger tooling for consumer credit risk."""

from .evaluation import evaluate_binary_classifier, ks_statistic
from .governance import PromotionPolicy, compare_candidate

__all__ = [
    "PromotionPolicy",
    "compare_candidate",
    "evaluate_binary_classifier",
    "ks_statistic",
]

__version__ = "0.1.0"
