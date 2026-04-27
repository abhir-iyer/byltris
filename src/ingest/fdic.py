import requests
import pandas as pd
import os
import time

BASE = "https://banks.data.fdic.gov/api"

FIELDS = [
    "REPDTE", "CERT", "INSTNAME", "STNAME", "CITY",
    "ASSET", "DEP", "LNLSNET", "INTINC", "EINTEXP",
    "NETINC", "RBCT1J", "NCLNLS", "EQ", "LNLSNTV",
    "LNRE", "LNCI", "LNAG"
]

def pull_financials(year_start: int = 2005) -> pd.DataFrame:
    """
    Pull quarterly financial data for all FDIC-insured banks
    from year_start to present. Saves to data/raw/fdic_financials.parquet
    """
    print(f"Pulling FDIC call report data from {year_start} onwards...")
    frames = []
    offset = 0
    batch = 0

    while True:
        try:
            resp = requests.get(
                f"{BASE}/financials",
                params={
                    "filters": f"REPDTE:[{year_start}0101 TO 99991231]",
                    "fields": ",".join(FIELDS),
                    "limit": 10000,
                    "offset": offset,
                    "sort_by": "REPDTE",
                    "sort_order": "ASC",
                    "output": "json"
                },
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                print("No more data. Done.")
                break

            batch_df = pd.DataFrame([x["data"] for x in data])
            frames.append(batch_df)
            batch += 1
            offset += 10000

            print(f"  Batch {batch}: pulled {len(batch_df)} rows (total so far: {offset})")
            time.sleep(0.3)  # be polite to the API

        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break

    if not frames:
        print("No data pulled. Check your internet connection.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # clean up types
    df["REPDTE"] = pd.to_datetime(df["REPDTE"], format="%Y%m%d", errors="coerce")
    numeric_cols = [c for c in df.columns if c not in ["REPDTE", "INSTNAME", "STNAME", "CITY"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # save
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/fdic_financials.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\nDone. Shape: {df.shape}")
    print(f"Saved to {out_path}")
    print(f"Date range: {df['REPDTE'].min()} → {df['REPDTE'].max()}")
    print(f"Unique banks: {df['CERT'].nunique():,}")
    return df


if __name__ == "__main__":
    df = pull_financials(year_start=2005)
    print("\nSample:")
    print(df[["REPDTE", "CERT", "INSTNAME", "ASSET", "DEP", "NCLNLS", "EQ"]].head(10))