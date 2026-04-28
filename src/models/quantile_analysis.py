import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_extended.parquet"

CREDIT_FEATURES = [
    "log_income", "log_loan_amount",
    "dti_mid", "purpose_purchase", "purpose_refi", "purpose_cashout"
]
RACE_FEATURES = ["is_black", "is_hispanic", "is_asian"]
ALL_FEATURES  = CREDIT_FEATURES + RACE_FEATURES


def prep(df, institution="Truist Bank"):
    sub = df[df["institution"] == institution].copy()
    sub["log_income"]      = np.log1p(pd.to_numeric(sub["income"], errors="coerce").clip(lower=0))
    sub["log_loan_amount"] = np.log1p(pd.to_numeric(sub["loan_amount"], errors="coerce").clip(lower=0))
    sub["action_taken"]    = pd.to_numeric(sub["action_taken"], errors="coerce")
    sub = sub[sub["action_taken"].isin([1, 3])].copy()
    sub["approved"] = (sub["action_taken"] == 1).astype(int)
    sub["activity_year"] = pd.to_numeric(sub["activity_year"], errors="coerce")
    sub["loan_amount_num"] = pd.to_numeric(sub["loan_amount"], errors="coerce")

    def dti_mid(val):
        try:
            if "-" in str(val):
                lo, hi = str(val).replace("%", "").split("-")
                return (float(lo) + float(hi)) / 2
            return float(str(val).replace("%","").replace("<","").replace(">","").strip())
        except:
            return np.nan
    sub["dti_mid"] = sub["debt_to_income_ratio"].apply(dti_mid)

    sub["is_black"]    = (sub["derived_race"] == "Black or African American").astype(int)
    sub["is_hispanic"] = sub["derived_ethnicity"].str.contains("Hispanic", na=False).astype(int)
    sub["is_asian"]    = (sub["derived_race"] == "Asian").astype(int)

    lp = sub["loan_purpose"].astype(str)
    sub["purpose_purchase"] = (lp == "1").astype(int)
    sub["purpose_refi"]     = (lp == "31").astype(int)
    sub["purpose_cashout"]  = (lp == "32").astype(int)
    return sub


# ── 1. Logit by loan amount quartile ─────────────────────────────────────────
def logit_by_loan_quartile(sub):
    print("=== BLACK OR BY LOAN AMOUNT QUARTILE ===\n")

    sub2 = sub.copy()
    sub2 = sub2[sub2["loan_amount_num"].notna() & (sub2["loan_amount_num"] > 0)]
    sub2["loan_quartile"] = pd.qcut(
        sub2["loan_amount_num"], q=4,
        labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"],
        duplicates="drop"
    )

    results = []
    for q in ["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"]:
        s = sub2[sub2["loan_quartile"] == q]
        clean = s[ALL_FEATURES + ["approved"]].dropna()
        if len(clean) < 300:
            print(f"  {q}: insufficient data")
            continue

        # drop purpose dummies if they cause collinearity within quartile
        feats = [f for f in ALL_FEATURES if f in clean.columns]
        try:
            X = sm.add_constant(clean[feats].astype(float))
            y = clean["approved"].astype(int)
            res = sm.Logit(y, X).fit(disp=0)
            b_or = np.exp(res.params.get("is_black", np.nan))
            b_p  = res.pvalues.get("is_black", np.nan)
            b_lo = np.exp(res.params["is_black"] - 1.96*res.bse["is_black"])
            b_hi = np.exp(res.params["is_black"] + 1.96*res.bse["is_black"])
            q_med = s["loan_amount_num"].median()
            print(f"  {q:<20} median=${q_med:>9,.0f}  N={len(clean):>7,}  "
                  f"Black OR={b_or:.4f}  [{b_lo:.4f},{b_hi:.4f}]  p={b_p:.4f}")
            results.append({"quartile": q, "median_loan": q_med,
                            "n": len(clean), "black_OR": b_or,
                            "black_CI_lo": b_lo, "black_CI_hi": b_hi})
        except Exception as e:
            print(f"  {q}: failed — {e}")

    return results


# ── 2. Logit by loan amount decile ───────────────────────────────────────────
def logit_by_loan_decile(sub):
    print("\n\n=== BLACK OR BY LOAN AMOUNT DECILE ===\n")

    sub2 = sub.copy()
    sub2 = sub2[sub2["loan_amount_num"].notna() & (sub2["loan_amount_num"] > 0)]
    sub2["loan_decile"] = pd.qcut(
        sub2["loan_amount_num"], q=10, labels=False, duplicates="drop"
    )

    results = []
    for d in range(10):
        s = sub2[sub2["loan_decile"] == d]
        clean = s[ALL_FEATURES + ["approved"]].dropna()
        if len(clean) < 200:
            continue
        try:
            X = sm.add_constant(clean[ALL_FEATURES].astype(float))
            y = clean["approved"].astype(int)
            res = sm.Logit(y, X).fit(disp=0)
            b_or = np.exp(res.params.get("is_black", np.nan))
            b_p  = res.pvalues.get("is_black", np.nan)
            q_lo = s["loan_amount_num"].quantile(0.1)
            q_hi = s["loan_amount_num"].quantile(0.9)
            print(f"  Decile {d+1:>2}  ${q_lo:>8,.0f}-${q_hi:>9,.0f}  "
                  f"N={len(clean):>6,}  Black OR={b_or:.4f}  p={b_p:.4f}")
            results.append({"decile": d+1, "q_lo": q_lo, "q_hi": q_hi,
                            "n": len(clean), "black_OR": b_or, "black_p": b_p})
        except Exception as e:
            print(f"  Decile {d+1}: failed — {e}")

    return results


# ── 3. Loan amount gap — Black vs White applicants ────────────────────────────
def loan_amount_distribution(sub):
    print("\n\n=== LOAN AMOUNT DISTRIBUTION BY RACE ===\n")

    sub2 = sub[sub["loan_amount_num"].notna() & (sub["loan_amount_num"] > 0)].copy()
    sub2["is_white"] = (sub2["derived_race"] == "White").astype(int)

    for race, mask in [
        ("Black or African American", sub2["derived_race"] == "Black or African American"),
        ("White", sub2["derived_race"] == "White"),
        ("Asian", sub2["derived_race"] == "Asian"),
    ]:
        s = sub2[mask]["loan_amount_num"]
        print(f"  {race:<35}  N={len(s):>7,}  "
              f"median=${s.median():>9,.0f}  mean=${s.mean():>9,.0f}  "
              f"p25=${s.quantile(0.25):>8,.0f}  p75=${s.quantile(0.75):>9,.0f}")


# ── 4. Peer quartile comparison ───────────────────────────────────────────────
def peer_quartile_comparison(df):
    print("\n\n=== PEER COMPARISON: Q1 vs Q4 LOAN AMOUNT ===\n")
    print("(Q1=smallest loans, Q4=largest — testing where gap is concentrated)\n")

    rows = []
    for inst in sorted(df["institution"].unique()):
        sub = prep(df, inst)
        sub2 = sub[sub["loan_amount_num"].notna() & (sub["loan_amount_num"] > 0)].copy()
        sub2["loan_quartile"] = pd.qcut(
            sub2["loan_amount_num"], q=4, labels=[1,2,3,4], duplicates="drop"
        )
        for q in [1, 4]:
            s = sub2[sub2["loan_quartile"] == q]
            clean = s[ALL_FEATURES + ["approved"]].dropna()
            if len(clean) < 300:
                continue
            try:
                X = sm.add_constant(clean[ALL_FEATURES].astype(float))
                y = clean["approved"].astype(int)
                res = sm.Logit(y, X).fit(disp=0)
                b_or = np.exp(res.params.get("is_black", np.nan))
                rows.append({"institution": inst, "quartile": f"Q{q}",
                             "n": len(clean), "black_OR": b_or})
            except:
                pass

    results_df = pd.DataFrame(rows)
    pivot = results_df.pivot(index="institution", columns="quartile", values="black_OR").round(4)
    if "Q1" in pivot.columns and "Q4" in pivot.columns:
        pivot["Q4-Q1"] = (pivot["Q4"] - pivot["Q1"]).round(4)
    print(pivot.to_string())

    results_df.to_parquet("data/processed/quantile_analysis.parquet", index=False)
    print(f"\nSaved to data/processed/quantile_analysis.parquet")
    return results_df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)
    print(f"Shape: {df.shape}\n")

    truist = prep(df, "Truist Bank")
    print(f"Truist sample: {len(truist):,} rows\n")

    quartile_results = logit_by_loan_quartile(truist)
    decile_results   = logit_by_loan_decile(truist)
    loan_amount_distribution(truist)
    peer_quartile_comparison(df)

    # save Truist quartile results
    if quartile_results:
        pd.DataFrame(quartile_results).to_parquet(
            "data/processed/truist_quartile_results.parquet", index=False)

    print("\nDone.")