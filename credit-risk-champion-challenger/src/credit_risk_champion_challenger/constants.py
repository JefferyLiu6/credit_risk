"""Shared schema and project defaults.

Keeping the model contract in one place prevents notebook-to-service drift.
"""

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
RANDOM_SEED = 42

NUMERIC_FEATURES = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
)

CATEGORICAL_FEATURES = (
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

