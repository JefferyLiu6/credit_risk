"""Dataset validation and persistent split helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import ID_COLUMN, RANDOM_SEED, TARGET_COLUMN


def replace_home_credit_sentinels(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize documented Home Credit sentinel values inside model pipelines."""
    transformed = frame.copy()
    if "DAYS_EMPLOYED" in transformed.columns:
        transformed["DAYS_EMPLOYED"] = transformed["DAYS_EMPLOYED"].replace(365243, np.nan)
    return transformed


def validate_training_frame(frame: pd.DataFrame) -> None:
    """Fail early when the minimum modeling contract is broken."""
    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not frame[ID_COLUMN].is_unique:
        raise ValueError(f"{ID_COLUMN} must be unique")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values")
    labels = set(frame[TARGET_COLUMN].unique())
    if labels != {0, 1}:
        raise ValueError(f"{TARGET_COLUMN} must contain both binary labels; got {labels}")


def make_split_assignments(
    frame: pd.DataFrame,
    *,
    train_size: float = 0.60,
    validation_size: float = 0.20,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create deterministic stratified train/validation/test assignments."""
    validate_training_frame(frame)
    if train_size <= 0 or validation_size <= 0 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must leave a non-empty test split")

    ids_train, ids_holdout, y_train, y_holdout = train_test_split(
        frame[ID_COLUMN],
        frame[TARGET_COLUMN],
        train_size=train_size,
        stratify=frame[TARGET_COLUMN],
        random_state=random_state,
    )
    validation_fraction = validation_size / (1 - train_size)
    ids_validation, ids_test = train_test_split(
        ids_holdout,
        train_size=validation_fraction,
        stratify=y_holdout,
        random_state=random_state,
    )

    assignments = pd.concat(
        [
            pd.DataFrame({ID_COLUMN: ids_train, "split": "train"}),
            pd.DataFrame({ID_COLUMN: ids_validation, "split": "val"}),
            pd.DataFrame({ID_COLUMN: ids_test, "split": "test"}),
        ],
        ignore_index=True,
    )
    if len(assignments) != len(frame) or not assignments[ID_COLUMN].is_unique:
        raise RuntimeError("Split assignment integrity check failed")
    return assignments.sort_values(ID_COLUMN).reset_index(drop=True)


def load_split_data(
    data_path: str | Path, assignments_path: str | Path
) -> dict[str, pd.DataFrame]:
    """Load data and attach the frozen split without silently dropping rows."""
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(assignments_path)
    validate_training_frame(frame)
    merged = frame.merge(assignments, on=ID_COLUMN, how="left", validate="one_to_one")
    if merged["split"].isna().any():
        raise ValueError("Some applications have no split assignment")
    return {name: group.drop(columns="split") for name, group in merged.groupby("split")}
