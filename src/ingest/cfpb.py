import pandas as pd
import os
import urllib.request

# CFPB publishes the full complaints database as a direct download
# This is cleaner and more reliable than the search API
CFPB_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
RAW_PATH = "data/raw/cfpb_complaints_full.csv.zip"
OUT_PATH = "data/raw/cfpb_complaints.parquet"

TRUIST_NAMES = [
    "truist", "suntrust", "bb&t", "branch banking"
]

PEER_NAMES = [
    "bank of america", "wells fargo", "jpmorgan chase",
    "u.s. bank", "regions bank", "citibank"
]

COLS_NEEDED = [
    "Date received",
    "Company",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company response to consumer",
    "State",
    "ZIP code",
    "Submitted via",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID"
]


def download_raw():
    os.makedirs("data/raw", exist_ok=True)
    if os.path.exists(RAW_PATH):
        print(f"Already downloaded: {RAW_PATH} — skipping download.")
        return
    print("Downloading CFPB full complaint database (~250MB)...")
    print("This will take a few minutes depending on your connection.")

    def progress(count, block_size, total_size):
        mb_done = count * block_size / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        print(f"\r  {mb_done:.1f} MB / {mb_total:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(CFPB_URL, RAW_PATH, reporthook=progress)
    print(f"\nDownloaded to {RAW_PATH}")


def process():
    print("\nReading CSV (this takes ~1 min for 5M rows)...")
    df = pd.read_csv(
        RAW_PATH,
        usecols=COLS_NEEDED,
        dtype=str,
        low_memory=False
    )

    # normalize
    df.columns = [c.lower().replace(" ", "_").replace("?", "").replace("-", "_") for c in df.columns]
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["company_lower"] = df["company"].str.lower().fillna("")

    # flag truist and peers
    df["is_truist"] = df["company_lower"].apply(
        lambda x: any(t in x for t in TRUIST_NAMES)
    )
    df["is_peer"] = df["company_lower"].apply(
        lambda x: any(p in x for p in PEER_NAMES)
    )

    # keep only truist + peers to stay manageable
    df_filtered = df[df["is_truist"] | df["is_peer"]].copy()
    df_filtered = df_filtered.drop(columns=["company_lower"])

    print(f"\nTotal complaints in database: {len(df):,}")
    print(f"Truist-related: {df['is_truist'].sum():,}")
    print(f"Peer banks: {df['is_peer'].sum():,}")
    print(f"Filtered dataset shape: {df_filtered.shape}")

    df_filtered.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")

    print("\nTruist complaints by year:")
    truist = df_filtered[df_filtered["is_truist"]]
    print(truist.groupby(truist["date_received"].dt.year).size())

    print("\nComplaints by company (top 15):")
    print(df_filtered["company"].value_counts().head(15))


if __name__ == "__main__":
    download_raw()
    process()