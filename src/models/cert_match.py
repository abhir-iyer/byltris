import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

DATA_CFPB = "data/raw/cfpb_complaints.parquet"
DATA_FDIC = "data/raw/fdic_financials.parquet"
OUT = "data/processed/cfpb_fdic_matched.parquet"
os.makedirs("data/processed", exist_ok=True)

INSTITUTION_MAP = {
    "truist": "Truist Bank",
    "suntrust": "SunTrust Banks",
    "bb&t": "BB&T Corporation",
    "bank of america": "Bank of America",
    "wells fargo": "Wells Fargo",
    "jpmorgan chase": "JPMorgan Chase",
    "citibank": "Citibank",
    "chase bank": "JPMorgan Chase",
}

def normalize(name):
    n = str(name).lower()
    for k, v in INSTITUTION_MAP.items():
        if k in n:
            return v
    return None

if __name__ == "__main__":
    print("Loading CFPB data...")
    cfpb = pd.read_parquet(DATA_CFPB)
    cfpb["institution"] = cfpb["company"].apply(normalize)
    cfpb = cfpb[cfpb["institution"].notna()].copy()

    print("Loading FDIC data...")
    fdic = pd.read_parquet(DATA_FDIC)

    print(f"\nCFPB institutions matched: {cfpb['institution'].notna().sum():,}")
    print(f"Total CFPB complaints in filtered set: {len(cfpb):,}")
    print(f"Match rate: {cfpb['institution'].notna().mean():.1%}")

    cfpb["date"] = pd.to_datetime(cfpb["date_received"], errors="coerce")
    cfpb["quarter"] = cfpb["date"].dt.to_period("Q")

    vel = (
        cfpb.groupby(["institution", "product", "quarter"])
        .size()
        .reset_index(name="count")
        .sort_values(["institution", "product", "quarter"])
    )
    vel["rolling_mean"] = vel.groupby(["institution", "product"])["count"].transform(
        lambda x: x.rolling(4, min_periods=2).mean()
    )
    vel["rolling_std"] = vel.groupby(["institution", "product"])["count"].transform(
        lambda x: x.rolling(4, min_periods=2).std()
    )
    vel["z_score"] = (vel["count"] - vel["rolling_mean"]) / vel["rolling_std"].replace(0, np.nan)
    vel["flagged"] = vel["z_score"] > 2.0

    flagged = vel[vel["flagged"]].sort_values("z_score", ascending=False)
    print(f"\nFlagged complaint velocity spikes (z > 2.0): {len(flagged)}")
    print(flagged[["institution", "product", "quarter", "count", "z_score"]].head(20).to_string(index=False))

    vel.to_parquet(OUT, index=False)
    print(f"\nSaved to {OUT}")