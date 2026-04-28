import pandas as pd
import gc

OUT_PATH = "data/raw/hmda_extended.parquet"

KEEP_COLS = [
    "activity_year", "derived_race", "derived_ethnicity", "derived_sex",
    "action_taken", "loan_purpose", "loan_amount", "income",
    "debt_to_income_ratio", "denial_reason-1", "denial_reason-2",
    "denial_reason-3", "denial_reason-4", "state_code",
    "combined_loan_to_value_ratio", "institution", "approved", "denied"
]

print("Loading 2018-2020...")
df1 = pd.read_parquet("data/raw/hmda_2018_2020.parquet")
df1 = df1.loc[:, ~df1.columns.duplicated()]
print(f"  Shape: {df1.shape}")

print("Loading existing 2021-2023...")
existing = pd.read_parquet("data/raw/hmda_truist.parquet")
existing = existing.loc[:, ~existing.columns.duplicated()]
existing.columns = [c.lower().strip() for c in existing.columns]

# slim to needed columns only
keep = [c for c in KEEP_COLS if c in existing.columns]
existing = existing[keep].copy()

for col in ["activity_year", "action_taken", "loan_amount", "income"]:
    if col in existing.columns:
        existing[col] = pd.to_numeric(existing[col].squeeze(), errors="coerce")

if "approved" not in existing.columns:
    existing["approved"] = (existing["action_taken"] == 1).astype(int)
if "denied" not in existing.columns:
    existing["denied"] = (existing["action_taken"] == 3).astype(int)

print(f"  Shape: {existing.shape}")

# combine
common = [c for c in df1.columns if c in existing.columns]
print(f"  Common columns: {len(common)}")

combined = pd.concat([df1[common], existing[common]], ignore_index=True)
del df1, existing
gc.collect()

combined.to_parquet(OUT_PATH, index=False)

print(f"\nTotal records: {len(combined):,}")
print(f"Years: {sorted(combined['activity_year'].dropna().astype(int).unique())}")
print(f"Saved to {OUT_PATH}")
print(combined.groupby(["institution","activity_year"]).size().unstack(fill_value=0).to_string())