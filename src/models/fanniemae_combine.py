"""
Fannie Mae Combine — run after all 8 extractions are complete
==============================================================
Combines the 8 quarterly parquets into one clean analysis file.

Run after:
  python src/models/fanniemae_extract_one.py data/raw/fanniemae/2021Q1.csv
  python src/models/fanniemae_extract_one.py data/raw/fanniemae/2021Q2.csv
  ... (all 8)

Then:
  python src/models/fanniemae_combine.py
"""

import pandas as pd
import numpy as np
import os
import glob

EXTRACTED_DIR = "data/raw/fanniemae/extracted"
OUT_PATH      = "data/processed/fanniemae_2021_2022.parquet"

EXPECTED = [
    "fanniemae_2021Q1.parquet",
    "fanniemae_2021Q2.parquet",
    "fanniemae_2021Q3.parquet",
    "fanniemae_2021Q4.parquet",
    "fanniemae_2022Q1.parquet",
    "fanniemae_2022Q2.parquet",
    "fanniemae_2022Q3.parquet",
    "fanniemae_2022Q4.parquet",
]

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    files = sorted(glob.glob(os.path.join(EXTRACTED_DIR, "fanniemae_*.parquet")))

    print("Fannie Mae Combine")
    print("="*50)
    print(f"\nFound {len(files)} extracted files:")
    for f in files:
        mb = os.path.getsize(f) / 1e6
        print(f"  {os.path.basename(f):40s}  {mb:.0f} MB")

    missing = [e for e in EXPECTED
               if not os.path.exists(os.path.join(EXTRACTED_DIR, e))]
    if missing:
        print(f"\nMissing files ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
        print("\nCombining what is available. Re-run after remaining files are extracted.")

    if not files:
        print("No extracted files found. Run fanniemae_extract_one.py first.")
        exit(1)

    # Load and combine
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df)
        print(f"  Loaded {os.path.basename(f)}: {len(df):,} loans")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal before dedup: {len(combined):,}")

    # Cross-quarter dedup: a loan originated in Q1 may appear in all subsequent
    # quarterly files. Keep first occurrence (earliest quarter = origination).
    combined = combined.sort_values("source_file")  # sorts chronologically
    combined = combined.drop_duplicates(subset=["loan_id"], keep="first")
    print(f"After cross-quarter dedup: {len(combined):,} unique loans")

    # Save
    combined.to_parquet(OUT_PATH, index=False)
    out_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\nSaved: {OUT_PATH}  ({out_mb:.0f} MB)")

    # Summary
    print(f"\nCoverage summary:")
    if "year" in combined.columns:
        yr = combined.groupby("year").size()
        for y, n in yr.items():
            print(f"  {y}: {n:,} loans")

    if "is_truist" in combined.columns:
        print(f"\nTruist loans: {combined['is_truist'].sum():,}")

    if "fico" in combined.columns:
        clean = combined[combined["fico"].notna() &
                         (combined["fico"] >= 300) & (combined["fico"] <= 850)]
        print(f"\nFICO statistics (N={len(clean):,}):")
        print(f"  Mean:   {clean['fico'].mean():.0f}")
        print(f"  Median: {clean['fico'].median():.0f}")
        print(f"  P25:    {clean['fico'].quantile(0.25):.0f}")
        print(f"  P75:    {clean['fico'].quantile(0.75):.0f}")

        # Loan size quartile FICO distribution
        clean2 = clean[clean["original_upb"].notna() & (clean["original_upb"] > 0)]
        clean2 = clean2.copy()
        clean2["loan_q"] = pd.qcut(
            clean2["original_upb"], q=4,
            labels=["Q1 (small)", "Q2", "Q3", "Q4 (large)"]
        )
        print(f"\nFICO by loan size quartile:")
        print(f"  {'Quartile':<15} {'N':>8} {'Mean FICO':>10} {'Median':>8}")
        print(f"  {'-'*45}")
        for q in ["Q1 (small)", "Q2", "Q3", "Q4 (large)"]:
            s = clean2[clean2["loan_q"] == q]["fico"]
            print(f"  {q:<15} {len(s):>8,} {s.mean():>10.1f} {s.median():>8.0f}")

        q1 = clean2[clean2["loan_q"] == "Q1 (small)"]["fico"].median()
        q4 = clean2[clean2["loan_q"] == "Q4 (large)"]["fico"].median()
        print(f"\n  Q4 vs Q1 FICO gap: {q4-q1:.0f} points")
        if q4 - q1 < 30:
            print(f"  Small gap: FICO alone cannot explain the HMDA racial approval gap.")
        else:
            print(f"  Larger gap: FICO is a partial explanation — quantified here.")

    print(f"\nNext step: run gse_credit_scores.py with DATA_PATH = '{OUT_PATH}'")