from pathlib import Path

import pandas as pd

from credit_risk_champion_challenger.constants import TARGET_COLUMN
from credit_risk_champion_challenger.data import make_split_assignments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"

df = pd.read_csv(DATA_PATH)

print("\n=== Dataset Overview ===")
print("Shape:", df.shape)
print("Unique applicants:", df["SK_ID_CURR"].nunique())
print("Duplicate IDs:", df["SK_ID_CURR"].duplicated().sum())

print("\n=== Target Distribution ===")
print(df[TARGET_COLUMN].value_counts())
print("Positive rate:", f"{df[TARGET_COLUMN].mean():.2%}")

print("\n=== Top 10 Columns by Missing Rate ===")
missing_rate = df.isna().mean().sort_values(ascending=False).head(10)
print((missing_rate * 100).round(2))

# Home Credit uses 365243 as an anomalous employment-days value.
print("\nDAYS_EMPLOYED sentinel count:", (df["DAYS_EMPLOYED"] == 365243).sum())

# Save only applicant-to-split assignments. Preprocessing stays inside the model pipeline.
splits = make_split_assignments(df)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
splits.to_csv(OUTPUT_PATH, index=False)

print("\n=== Dataset Splits ===")
split_with_target = splits.merge(df[["SK_ID_CURR", TARGET_COLUMN]], on="SK_ID_CURR")
for name, group in split_with_target.groupby("split"):
    print(
        f"{name:5s} | rows={len(group):6d} "
        f"| positive_rate={group[TARGET_COLUMN].mean():.2%}"
    )

print(f"\nSaved: {OUTPUT_PATH}")
