"""
Fannie Mae Single Quarter Extractor — Correct Column Positions
==============================================================
Column positions confirmed from diagnostic on 2021Q1.csv (110-column format).

Confirmed positions:
  Col  1: loan_id
  Col  2: report_period (YYYYMM)
  Col  4: seller_name
  Col  7: original_rate
  Col  9: original_upb
  Col 12: loan_term (180=15yr, 360=30yr)
  Col 13: first_payment_date
  Col 15: loan_age
  Col 19: original_cltv
  Col 20: original_ltv
  Col 22: original_dti
  Col 23: fico
  Col 26: loan_purpose (P=purchase, R=refi, C=cash-out)
  Col 29: occupancy
  Col 30: property_state

Usage:
  python src/models/fanniemae_extract_one.py data/raw/fanniemae/2021Q1.csv
  ... repeat for all 8 files, then run fanniemae_combine.py
"""

import pandas as pd
import numpy as np
import os, sys, gc, shutil

OUT_DIR     = "data/raw/fanniemae/extracted"
CHUNK_SIZE  = 200_000
FLUSH_EVERY = 50

KEEP_COLS = {
    1:  "loan_id",
    2:  "report_period",
    4:  "seller_name",
    7:  "original_rate",
    9:  "original_upb",
    12: "loan_term",
    13: "first_payment_date",
    15: "loan_age",
    19: "original_cltv",
    20: "original_ltv",
    22: "original_dti",
    23: "fico",
    26: "loan_purpose",
    29: "occupancy",
    30: "property_state",
}

TRUIST_NAMES = ["TRUIST", "BB&T", "SUNTRUST", "BRANCH BANKING"]


def extract(input_path, out_dir=OUT_DIR, chunk_size=CHUNK_SIZE):
    base     = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(out_dir, f"fanniemae_{base}.parquet")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        mb = os.path.getsize(out_path) / 1e6
        print(f"Already exists: {out_path} ({mb:.0f} MB) — skipping.")
        return

    size_gb = os.path.getsize(input_path) / 1e9
    print(f"\nInput:  {os.path.basename(input_path)} ({size_gb:.1f} GB)")
    print(f"Output: {out_path}")

    # Quick validation
    print("\nValidating on first 500 rows...")
    sample = pd.read_csv(
        input_path, sep="|", header=None, nrows=500,
        dtype=str, low_memory=False, usecols=list(KEEP_COLS.keys())
    ).rename(columns=KEEP_COLS)

    for field, col_check, lo, hi in [
        ("fico",         "fico",         600, 850),
        ("original_dti", "original_dti", 10,  65),
        ("original_upb", "original_upb", 50000, 3_000_000),
    ]:
        if field in sample.columns:
            vals  = pd.to_numeric(sample[field], errors="coerce").dropna()
            valid = vals[(vals >= lo) & (vals <= hi)]
            pct   = len(valid) / max(len(vals), 1) * 100
            print(f"  {field:<15} col {[k for k,v in KEEP_COLS.items() if v==field][0]:>2}: "
                  f"{pct:.0f}% valid  sample={vals.head(3).tolist()}")

    if "property_state" in sample.columns:
        print(f"  state col 30: {sample['property_state'].value_counts().head(5).to_dict()}")

    # ── Read and extract ──────────────────────────────────────────────────────
    col_indices  = sorted(KEEP_COLS.keys())
    seen_loans   = set()
    total_rows   = 0
    unique_total = 0
    chunk_count  = 0
    batch_frames = []
    temp_files   = []
    tmp_dir      = os.path.join(out_dir, f"_tmp_{base}")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"\nExtracting (flush every {FLUSH_EVERY} chunks)...")

    for chunk in pd.read_csv(
        input_path, sep="|", header=None,
        chunksize=chunk_size, dtype=str, low_memory=False,
        usecols=col_indices,
    ):
        chunk_count += 1
        total_rows  += len(chunk)
        chunk        = chunk.rename(columns=KEEP_COLS)

        new_mask = ~chunk["loan_id"].isin(seen_loans)
        new_rows = chunk[new_mask].copy()
        seen_loans.update(new_rows["loan_id"].dropna().tolist())
        batch_frames.append(new_rows)
        unique_total += len(new_rows)

        if chunk_count % 10 == 0:
            print(f"  Chunk {chunk_count:>4}  rows={total_rows:>12,}  unique={unique_total:>8,}")

        if chunk_count % FLUSH_EVERY == 0:
            tmp_p = os.path.join(tmp_dir, f"b{chunk_count:06d}.parquet")
            pd.concat(batch_frames, ignore_index=True).to_parquet(tmp_p, index=False)
            temp_files.append(tmp_p)
            batch_frames = []
            gc.collect()

    if batch_frames:
        tmp_p = os.path.join(tmp_dir, f"b{chunk_count:06d}_final.parquet")
        pd.concat(batch_frames, ignore_index=True).to_parquet(tmp_p, index=False)
        temp_files.append(tmp_p)
        del batch_frames
        gc.collect()

    print(f"\nRows read: {total_rows:,}  Unique loans: {unique_total:,}")

    # Combine temp files
    print(f"Combining {len(temp_files)} batch files...")
    df = pd.concat([pd.read_parquet(f) for f in temp_files], ignore_index=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    gc.collect()

    before = len(df)
    df = df.drop_duplicates(subset=["loan_id"], keep="first")
    if len(df) < before:
        print(f"Dedup: {before:,} -> {len(df):,}")

    # Clean numerics
    for col in ["fico","original_dti","original_ltv","original_cltv",
                "original_upb","original_rate","loan_age","loan_term"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "fico" in df.columns:
        df = df[df["fico"].isna() | ((df["fico"] >= 300) & (df["fico"] <= 850))].copy()

    # Truist flag
    if "seller_name" in df.columns:
        df["is_truist"] = df["seller_name"].str.upper().str.strip().apply(
            lambda x: int(any(k in str(x) for k in TRUIST_NAMES))
        )

    # Year — first_payment_date is MMYYYY format (e.g. 122021 = Dec 2021)
    # Take last 4 characters to get the year
    if "first_payment_date" in df.columns:
        df["year"] = pd.to_numeric(
            df["first_payment_date"].astype(str).str[-4:], errors="coerce"
        )

    if "original_upb" in df.columns:
        df["log_upb"] = np.log1p(df["original_upb"].fillna(0).clip(lower=0))

    if "loan_purpose" in df.columns:
        lp = df["loan_purpose"].astype(str).str.strip()
        df["is_purchase"] = (lp == "P").astype(int)
        df["is_cashout"]  = (lp == "C").astype(int)
        df["is_refi"]     = (lp == "R").astype(int)

    df["source_file"] = base

    # Save
    df.to_parquet(out_path, index=False)
    mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved: {out_path}  ({mb:.0f} MB)  shape={df.shape}")

    # Summary
    if "fico" in df.columns:
        clean = df[df["fico"].notna() & (df["fico"] >= 600)]
        print(f"\nFICO  N={len(clean):,}  mean={clean['fico'].mean():.0f}  "
              f"median={clean['fico'].median():.0f}  "
              f"P25={clean['fico'].quantile(.25):.0f}  P75={clean['fico'].quantile(.75):.0f}")

        # Key result for paper: FICO by loan size
        c2 = clean[clean["original_upb"].notna() & (clean["original_upb"] > 0)].copy()
        c2["loan_q"] = pd.qcut(c2["original_upb"], q=4,
                               labels=["Q1","Q2","Q3","Q4"])
        print(f"\nFICO by loan size quartile:")
        for q in ["Q1","Q2","Q3","Q4"]:
            s   = c2[c2["loan_q"]==q]["fico"]
            upb = c2[c2["loan_q"]==q]["original_upb"]
            print(f"  {q}  N={len(s):>7,}  FICO_median={s.median():.0f}  "
                  f"loan_median=${upb.median():>8,.0f}")
        q1 = c2[c2["loan_q"]=="Q1"]["fico"].median()
        q4 = c2[c2["loan_q"]=="Q4"]["fico"].median()
        print(f"\n  Q4-Q1 FICO gap: {q4-q1:.0f} points")
        if q4-q1 < 30:
            print("  Small gap: FICO alone cannot explain the HMDA racial approval gap.")
        else:
            print(f"  Gap of {q4-q1:.0f} pts is meaningful — partial explanation.")

    if "is_truist" in df.columns:
        t = df[df["is_truist"]==1]
        print(f"\nTruist loans: {len(t):,}")
        if len(t) and "fico" in t.columns:
            c = t[t["fico"].notna() & (t["fico"]>=600)]
            if len(c):
                print(f"Truist FICO: mean={c['fico'].mean():.0f}  median={c['fico'].median():.0f}")

    if "seller_name" in df.columns:
        print(f"\nTop sellers:")
        for name, n in df["seller_name"].value_counts().head(12).items():
            marker = " <-- TRUIST" if any(k in str(name).upper() for k in TRUIST_NAMES) else ""
            print(f"  {name:<45} {n:>8,}{marker}")

    if "year" in df.columns:
        print(f"\nYears: {sorted(df['year'].dropna().astype(int).unique())}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/models/fanniemae_extract_one.py <file>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Not found: {path}"); sys.exit(1)
    extract(path)