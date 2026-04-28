import requests
import pandas as pd
import os
from io import StringIO

OUT_PATH = "data/raw/hmda_extended.parquet"
os.makedirs("data/raw", exist_ok=True)

BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

INSTITUTIONS = {
    "Truist Bank":    "JJKC32MCHWDI71265Z06",
    "Wells Fargo":    "KB1H1DSPRFMYMCUFXT09",
    "Bank of America":"B4TYDEB6GKMZO031MB27",
    "JPMorgan Chase": "7H6GLXDRUGQFU57RNE97",
    "Regions Bank":   "EQTWLK1G7ODGC2MGLV11",
    "PNC Bank":       "AD6GFRVSDT01YPT1CS68",
    "Fifth Third Bank":"QFROUN1UWUYU0DVIWD51",
    "Huntington Bank":"2WHM8VNJH63UN14OL754",
    "Citizens Bank":  "DRMSV1Q0EKMEXLAU1P80",
    "U.S. Bank":      "6BYL5QZYBDK8S7L73M02",
}

PRE_MERGER = {
    "SunTrust Bank":  "549300JM5CPASQV8RB76",
    "BB&T":           "549300LN4MK776JHG785",
}

YEARS_NEW = [2018, 2019, 2020]

# Only keep the columns we actually use — drastically cuts memory
KEEP_COLS = [
    "activity_year", "derived_race", "derived_ethnicity", "derived_sex",
    "action_taken", "loan_purpose", "loan_amount", "income",
    "debt_to_income_ratio", "denial_reason-1", "denial_reason-2",
    "denial_reason-3", "denial_reason-4", "state_code",
    "combined_loan_to_value_ratio"
]


def pull_one(name, lei, year):
    print(f"  {name} {year}...", end=" ", flush=True)
    try:
        resp = requests.get(BASE, params={"leis": lei, "years": year}, timeout=120)
        if resp.status_code != 200 or len(resp.text) < 100:
            print(f"empty/error (HTTP {resp.status_code})")
            return pd.DataFrame()
        df = pd.read_csv(StringIO(resp.text), dtype=str, low_memory=False)
        df.columns = [c.lower().strip() for c in df.columns]
        # keep only needed columns
        keep = [c for c in KEEP_COLS if c in df.columns]
        df = df[keep].copy()
        df["institution"] = name
        df["activity_year"] = str(year)
        print(f"{len(df):,} rows, {df.shape[1]} cols")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def run_all():
    frames = []

    for name, lei in INSTITUTIONS.items():
        for year in YEARS_NEW:
            df = pull_one(name, lei, year)
            if not df.empty:
                frames.append(df)

    for name, lei in PRE_MERGER.items():
        for year in [2018, 2019]:
            df = pull_one(name, lei, year)
            if not df.empty:
                df["institution"] = "Truist Bank"
                frames.append(df)

    if not frames:
        print("No data pulled.")
        return

    new_data = pd.concat(frames, ignore_index=True)
    
    # deduplicate any duplicate column names
    new_data = new_data.loc[:, ~new_data.columns.duplicated()]
    print(f"\nNew data shape: {new_data.shape}")

    # save new data first to free memory
    new_data["activity_year"] = pd.to_numeric(new_data["activity_year"], errors="coerce")
    new_data["action_taken"]  = pd.to_numeric(new_data["action_taken"], errors="coerce")
    new_data["loan_amount"]   = pd.to_numeric(new_data["loan_amount"], errors="coerce")
    new_data["income"]        = pd.to_numeric(new_data["income"], errors="coerce")
    new_data["approved"] = (new_data["action_taken"] == 1).astype(int)
    new_data["denied"]   = (new_data["action_taken"] == 3).astype(int)
    new_data.to_parquet("data/raw/hmda_2018_2020.parquet", index=False)
    print(f"Saved 2018-2020 data: {new_data.shape}")
    del new_data
    import gc; gc.collect()

    # load existing, slim it
    print("Loading existing 2021-2023 data...")
    existing = pd.read_parquet("data/raw/hmda_truist.parquet")
    existing.columns = [c.lower().strip() for c in existing.columns]
    existing = existing.loc[:, ~existing.columns.duplicated()]
    keep_exist = [c for c in KEEP_COLS + ["institution", "activity_year"] if c in existing.columns]
    existing = existing[keep_exist].copy()
    existing["activity_year"] = pd.to_numeric(existing["activity_year"], errors="coerce")
    existing["action_taken"]  = pd.to_numeric(existing["action_taken"], errors="coerce")
    existing["loan_amount"]   = pd.to_numeric(existing["loan_amount"], errors="coerce")
    existing["income"]        = pd.to_numeric(existing["income"], errors="coerce")
    existing["approved"] = (existing["action_taken"] == 1).astype(int)
    existing["denied"]   = (existing["action_taken"] == 3).astype(int)
    existing.to_parquet("data/raw/hmda_2021_2023.parquet", index=False)
    print(f"Saved 2021-2023 slim: {existing.shape}")
    del existing
    gc.collect()

    # combine from disk
    print("Combining from disk...")
    df1 = pd.read_parquet("data/raw/hmda_2018_2020.parquet")
    df2 = pd.read_parquet("data/raw/hmda_2021_2023.parquet")
    common = [c for c in df1.columns if c in df2.columns]
    combined = pd.concat([df1[common], df2[common]], ignore_index=True)
    combined.to_parquet(OUT_PATH, index=False)

    print(f"\nTotal records: {len(combined):,}")
    print(f"Years: {sorted(combined['activity_year'].dropna().astype(int).unique())}")
    print(f"Saved to {OUT_PATH}")
    print(combined.groupby(["institution","activity_year"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    run_all()