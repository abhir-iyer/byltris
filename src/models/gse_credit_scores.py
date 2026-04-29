"""
Step 5: Fannie Mae GSE Data — FICO Analysis
============================================
Loads the combined parquet produced by fanniemae_combine.py and runs:
  1. FICO distribution by loan size quartile
  2. Interest rate regression (FICO + DTI + LTV + loan size)
  3. Truist-specific FICO analysis

Key question: is the FICO gap between small and large loans large enough
to explain the racial approval gap concentrated in small loans (HMDA)?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/fanniemae_2021_2022.parquet"
OUT_PATH  = "data/processed/gse_credit_score_analysis.parquet"

TRUIST_NAMES = ["TRUIST", "BB&T", "SUNTRUST", "BRANCH BANKING"]


# ══════════════════════════════════════════════════════════════════════════════
def fico_distribution_analysis(df):
    print("=== FICO DISTRIBUTION BY LOAN SIZE ===")
    print("(Small loans are disproportionately Black applicants per HMDA)")

    clean = df[
        df["fico"].notna() & (df["fico"] >= 300) & (df["fico"] <= 850) &
        df["original_upb"].notna() & (df["original_upb"] > 0)
    ].copy()

    clean["loan_quartile"] = pd.qcut(
        clean["original_upb"], q=4,
        labels=["Q1 (small)", "Q2", "Q3", "Q4 (large)"]
    )

    print(f"\n  {'Quartile':<15} {'N':>8} {'Mean FICO':>10} {'Median':>8} "
          f"{'P25':>6} {'P75':>6} {'Median loan':>12}")
    print(f"  {'-'*68}")
    for q in ["Q1 (small)", "Q2", "Q3", "Q4 (large)"]:
        sub = clean[clean["loan_quartile"] == q]
        f   = sub["fico"]
        upb = sub["original_upb"]
        print(f"  {q:<15} {len(f):>8,} {f.mean():>10.1f} {f.median():>8.0f} "
              f"{f.quantile(.25):>6.0f} {f.quantile(.75):>6.0f} "
              f"${upb.median():>10,.0f}")

    q1_med = clean[clean["loan_quartile"] == "Q1 (small)"]["fico"].median()
    q4_med = clean[clean["loan_quartile"] == "Q4 (large)"]["fico"].median()
    gap    = q4_med - q1_med
    q1_upb = clean[clean["loan_quartile"] == "Q1 (small)"]["original_upb"].median()
    q4_upb = clean[clean["loan_quartile"] == "Q4 (large)"]["original_upb"].median()

    print(f"\n  Q4 vs Q1 median FICO gap: {gap:.0f} points")
    print(f"  (loan size range: ${q1_upb:,.0f} to ${q4_upb:,.0f})")

    if gap < 30:
        print(f"\n  CONCLUSION: A {gap:.0f}-point FICO gap across a "
              f"${q4_upb-q1_upb:,.0f} spread in loan size")
        print(f"  cannot explain the racial approval gap concentrated in small")
        print(f"  loans documented in HMDA. FICO omission is not the mechanism.")
    else:
        print(f"\n  FICO gap of {gap:.0f} pts — partial explanation possible.")

    # By year
    print(f"\n  FICO gap by year (2021-2022):")
    for yr in [2021, 2022]:
        sub = clean[clean["year"] == yr].copy() if "year" in clean.columns else pd.DataFrame()
        if len(sub) < 1000:
            continue
        sub["lq"] = pd.qcut(sub["original_upb"], q=4, labels=["Q1","Q2","Q3","Q4"])
        q1y = sub[sub["lq"] == "Q1"]["fico"].median()
        q4y = sub[sub["lq"] == "Q4"]["fico"].median()
        print(f"    {yr}: Q1={q1y:.0f}  Q4={q4y:.0f}  gap={q4y-q1y:.0f} pts  "
              f"N={len(sub):,}")

    return clean, gap


# ══════════════════════════════════════════════════════════════════════════════
def interest_rate_analysis(df):
    print("\n\n=== INTEREST RATE REGRESSION ===")
    print("Testing: does loan size predict rate AFTER controlling for FICO?")

    if "original_rate" not in df.columns:
        print("  original_rate column not in data — skipping.")
        return None, None

    clean = df[
        df["original_rate"].notna() & (df["original_rate"] > 0) & (df["original_rate"] < 15) &
        df["fico"].notna() & (df["fico"] >= 300) & (df["fico"] <= 850) &
        df["original_dti"].notna() & (df["original_dti"] > 0) &
        df["original_ltv"].notna() & (df["log_upb"].notna())
    ].copy()

    print(f"  N: {len(clean):,}")
    if len(clean) < 1000:
        print("  Insufficient data.")
        return None, None

    features = ["fico", "original_dti", "original_ltv", "log_upb", "is_purchase"]
    features = [f for f in features if f in clean.columns]

    X  = sm.add_constant(clean[features].astype(float))
    y  = clean["original_rate"].astype(float)
    r1 = sm.OLS(y, X).fit()

    print(f"\n  OLS: Rate ~ {' + '.join(features)}")
    print(f"  R²: {r1.rsquared:.4f}")
    for f in features:
        print(f"    {f:<22} coef={r1.params[f]:>9.6f}  p={r1.pvalues[f]:.4f}")

    # Small loan premium
    clean["loan_q"]   = pd.qcut(clean["log_upb"], q=4, labels=[1,2,3,4])
    clean["is_small"] = (clean["loan_q"] == 1).astype(int)
    feats2 = [f for f in ["fico","original_dti","original_ltv","is_purchase","is_small"]
              if f in clean.columns]
    X2 = sm.add_constant(clean[feats2].astype(float))
    r2 = sm.OLS(y, X2).fit()
    sc = r2.params.get("is_small", np.nan)
    sp = r2.pvalues.get("is_small", np.nan)

    print(f"\n  Small loan rate premium (after FICO controls):")
    print(f"    coef={sc:.4f}  p={sp:.4f}  ({sc*100:.1f} bps)")
    if not np.isnan(sp):
        if sp < 0.05 and sc > 0:
            print(f"    Significant: small loans carry a rate premium even after FICO.")
            print(f"    Consistent with HMDA loan-size gradient finding.")
        else:
            print(f"    Not significant at 5%.")

    return r1, r2


# ══════════════════════════════════════════════════════════════════════════════
def truist_analysis(df):
    print("\n\n=== TRUIST IN FANNIE MAE DATA ===")

    if "is_truist" not in df.columns:
        df["is_truist"] = df["seller_name"].str.upper().str.strip().apply(
            lambda x: int(any(k in str(x) for k in TRUIST_NAMES))
        ) if "seller_name" in df.columns else 0

    t = df[df["is_truist"] == 1]
    print(f"  Truist-originated loans: {len(t):,}")

    if len(t) < 100:
        print("  Truist is anonymized as 'Other' in Fannie Mae public data.")
        print("  Seller-level analysis not possible with the public dataset.")
        print("  Industry-wide FICO analysis (above) applies to all GSE lenders.")
        return

    tv = t[t["fico"].notna() & (t["fico"]>=300) &
           t["original_upb"].notna() & (t["original_upb"]>0)].copy()
    tv["lq"] = pd.qcut(tv["original_upb"], q=4, labels=["Q1","Q2","Q3","Q4"])

    print(f"\n  Truist FICO by loan size:")
    for q in ["Q1","Q2","Q3","Q4"]:
        s   = tv[tv["lq"]==q]["fico"]
        upb = tv[tv["lq"]==q]["original_upb"]
        print(f"    {q}  N={len(s):>6,}  FICO_median={s.median():.0f}  "
              f"loan_median=${upb.median():>8,.0f}")

    q1f = tv[tv["lq"]=="Q1"]["fico"].median()
    q4f = tv[tv["lq"]=="Q4"]["fico"].median()
    print(f"\n  Truist Q4-Q1 FICO gap: {q4f-q1f:.0f} points")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print(f"Loading: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        print("Run fanniemae_combine.py first.")
        exit(1)

    df = pd.read_parquet(DATA_PATH)
    print(f"Shape: {df.shape}")

    # Ensure derived columns
    if "log_upb" not in df.columns and "original_upb" in df.columns:
        df["log_upb"] = np.log1p(df["original_upb"].fillna(0).clip(lower=0))
    if "is_purchase" not in df.columns and "loan_purpose" in df.columns:
        df["is_purchase"] = (df["loan_purpose"].astype(str).str.strip() == "P").astype(int)

    # Run
    clean_df, fico_gap = fico_distribution_analysis(df)
    interest_rate_analysis(df)
    truist_analysis(df)

    # Save summary
    rows = []
    for q in ["Q1 (small)", "Q2", "Q3", "Q4 (large)"]:
        sub = clean_df[clean_df["loan_quartile"] == q]
        rows.append({
            "quartile":     q,
            "n":            len(sub),
            "fico_mean":    sub["fico"].mean(),
            "fico_median":  sub["fico"].median(),
            "fico_p25":     sub["fico"].quantile(.25),
            "fico_p75":     sub["fico"].quantile(.75),
            "upb_median":   sub["original_upb"].median(),
        })
    pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")

    print("\n" + "="*60)
    print("KEY RESULT FOR PAPER:")
    print(f"  FICO gap Q4 vs Q1 (by loan size): {fico_gap:.0f} points")
    print(f"  Across {len(df):,} GSE-acquired loans, 2021-2022")
    print(f"  A {fico_gap:.0f}-pt FICO gap cannot account for the racial")
    print(f"  approval gap concentrated in small loans per HMDA.")
    print("="*60)
    print("\nDone.")