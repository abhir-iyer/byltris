import pandas as pd
import os
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
OUT_PATH = "data/raw/fred_macro.parquet"
os.makedirs("data/raw", exist_ok=True)

SERIES = {
    "fed_funds_rate":        "FEDFUNDS",
    "yield_curve_10y2y":     "T10Y2Y",
    "unemployment_rate":     "UNRATE",
    "credit_card_delinquency": "DRCCLACBS",
    "mortgage_delinquency":  "DRSFRMACBS",
    "consumer_sentiment":    "UMCSENT",
    "gdp_growth":            "A191RL1Q225SBEA",
    "cpi_yoy":               "CPIAUCSL",
}


def pull_fred():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("ERROR: FRED_API_KEY not found in .env file")
        return

    fred = Fred(api_key=api_key)
    frames = {}

    for name, series_id in SERIES.items():
        print(f"  Pulling {name} ({series_id})...", end=" ")
        try:
            s = fred.get_series(series_id, observation_start="2005-01-01")
            s.name = name
            frames[name] = s
            print(f"{len(s)} observations")
        except Exception as e:
            print(f"Error: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # resample to quarterly
    df_quarterly = df.resample("Q").mean()
    df_quarterly = df_quarterly.reset_index()

    df_quarterly.to_parquet(OUT_PATH, index=False)

    print(f"\nShape: {df_quarterly.shape}")
    print(f"Date range: {df_quarterly['date'].min()} → {df_quarterly['date'].max()}")
    print(f"Saved to {OUT_PATH}")
    print(f"\nSample:")
    print(df_quarterly.tail(6).to_string())


if __name__ == "__main__":
    pull_fred()