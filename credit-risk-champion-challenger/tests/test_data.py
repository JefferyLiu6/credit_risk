import pandas as pd
import pytest

from credit_risk_champion_challenger.data import (
    make_split_assignments,
    replace_home_credit_sentinels,
    validate_training_frame,
)


def _frame(n=100):
    return pd.DataFrame({"SK_ID_CURR": range(n), "TARGET": [0, 1] * (n // 2)})


def test_split_is_complete_unique_and_reproducible():
    first = make_split_assignments(_frame())
    second = make_split_assignments(_frame())
    pd.testing.assert_frame_equal(first, second)
    assert first["SK_ID_CURR"].is_unique
    assert first["split"].value_counts().to_dict() == {"train": 60, "val": 20, "test": 20}


def test_duplicate_applicant_is_rejected():
    frame = _frame()
    frame.loc[1, "SK_ID_CURR"] = 0
    with pytest.raises(ValueError, match="unique"):
        validate_training_frame(frame)


def test_employment_sentinel_is_normalized_without_mutating_input():
    frame = pd.DataFrame({"DAYS_EMPLOYED": [365243, -1000]})
    transformed = replace_home_credit_sentinels(frame)
    assert pd.isna(transformed.loc[0, "DAYS_EMPLOYED"])
    assert transformed.loc[1, "DAYS_EMPLOYED"] == -1000
    assert frame.loc[0, "DAYS_EMPLOYED"] == 365243
