import pandas as pd
import numpy as np
import statsmodels.api as sm
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


def run_logit(sub, label=""):
    clean = sub[ALL_FEATURES + ["approved"]].dropna()
    if len(clean) < 500:
        print(f"  {label}: insufficient data (n={len(clean)})")
        return None
    X = sm.add_constant(clean[ALL_FEATURES].astype(float))
    y = clean["approved"].astype(int)
    res = sm.Logit(y, X).fit(disp=0)
    b_or = np.exp(res.params.get("is_black", np.nan))
    b_p  = res.pvalues.get("is_black", np.nan)
    b_lo = np.exp(res.params["is_black"] - 1.96*res.bse["is_black"])
    b_hi = np.exp(res.params["is_black"] + 1.96*res.bse["is_black"])
    print(f"  {label:<42} N={len(clean):>7,}  Black OR={b_or:.4f}  [{b_lo:.4f},{b_hi:.4f}]  p={b_p:.4f}")
    return {"label": label, "n": len(clean), "black_OR": b_or,
            "black_CI_lo": b_lo, "black_CI_hi": b_hi, "black_p": b_p}


if __name__ == "__main__":
    print("Loading extended HMDA data...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)
    print(f"Shape: {df.shape}")
    print(f"Years: {sorted(df['activity_year'].dropna().astype(int).unique())}\n")

    results = []

    # ── 1. Truist by year (full 2018-2023 arc) ───────────────────────────────
    print("=== TRUIST: Black OR by year (2018-2023) ===")
    truist = prep(df, "Truist Bank")
    for year in sorted(truist["activity_year"].dropna().astype(int).unique()):
        r = run_logit(truist[truist["activity_year"] == year], f"Truist {year}")
        if r:
            results.append(r)

    # ── 2. Pre vs post merger ────────────────────────────────────────────────
    print("\n=== PRE vs POST MERGER ===")
    pre  = truist[truist["activity_year"].between(2018, 2020)]
    post = truist[truist["activity_year"].between(2021, 2023)]
    r_pre  = run_logit(pre,  "Truist PRE-merger  (2018-2020)")
    r_post = run_logit(post, "Truist POST-merger (2021-2023)")
    if r_pre and r_post:
        print(f"\n  OR change (post - pre): {r_post['black_OR'] - r_pre['black_OR']:+.4f}")
        print(f"  Merger {'widened' if r_post['black_OR'] < r_pre['black_OR'] else 'narrowed'} the gap")

    # ── 3. Peer comparison all years ─────────────────────────────────────────
    print("\n=== PEER COMPARISON (all available years) ===")
    for inst in sorted(df["institution"].unique()):
        sub = prep(df, inst)
        r = run_logit(sub, inst)
        if r:
            results.append(r)

    # ── 4. Triple DiD: did merger change the racial gap? ────────────────────
    print("\n=== DiD: DID THE MERGER CHANGE THE RACIAL GAP? ===")
    print("(Black x Truist x Post-2021 triple interaction)\n")

    peers = ["Wells Fargo", "Bank of America", "JPMorgan Chase",
             "Regions Bank", "PNC Bank", "U.S. Bank"]
    did_raw = df[df["institution"].isin(["Truist Bank"] + peers)].copy()

    did_raw["log_income"]      = np.log1p(pd.to_numeric(did_raw["income"], errors="coerce").clip(lower=0))
    did_raw["log_loan_amount"] = np.log1p(pd.to_numeric(did_raw["loan_amount"], errors="coerce").clip(lower=0))
    did_raw["action_taken"]    = pd.to_numeric(did_raw["action_taken"], errors="coerce")
    did_raw = did_raw[did_raw["action_taken"].isin([1, 3])].copy()
    did_raw["approved"] = (did_raw["action_taken"] == 1).astype(int)
    did_raw["activity_year"] = pd.to_numeric(did_raw["activity_year"], errors="coerce")

    def dti_mid(val):
        try:
            if "-" in str(val):
                lo, hi = str(val).replace("%", "").split("-")
                return (float(lo) + float(hi)) / 2
            return float(str(val).replace("%","").replace("<","").replace(">","").strip())
        except:
            return np.nan
    did_raw["dti_mid"] = did_raw["debt_to_income_ratio"].apply(dti_mid)
    did_raw["is_black"]    = (did_raw["derived_race"] == "Black or African American").astype(int)
    did_raw["is_hispanic"] = did_raw["derived_ethnicity"].str.contains("Hispanic", na=False).astype(int)
    did_raw["is_asian"]    = (did_raw["derived_race"] == "Asian").astype(int)
    lp = did_raw["loan_purpose"].astype(str)
    did_raw["purpose_purchase"] = (lp == "1").astype(int)
    did_raw["purpose_refi"]     = (lp == "31").astype(int)
    did_raw["purpose_cashout"]  = (lp == "32").astype(int)

    did_raw["treat"]           = (did_raw["institution"] == "Truist Bank").astype(int)
    did_raw["post"]            = (did_raw["activity_year"] >= 2021).astype(int)
    did_raw["black_treat"]     = did_raw["is_black"] * did_raw["treat"]
    did_raw["black_post"]      = did_raw["is_black"] * did_raw["post"]
    did_raw["treat_post"]      = did_raw["treat"] * did_raw["post"]
    did_raw["black_treat_post"]= did_raw["is_black"] * did_raw["treat"] * did_raw["post"]

    features_did = (ALL_FEATURES +
                    ["treat","post","treat_post",
                     "black_treat","black_post","black_treat_post"])

    clean_did = did_raw[features_did + ["approved"]].dropna()
    print(f"DiD sample: {len(clean_did):,} rows")
    X_did = sm.add_constant(clean_did[features_did].astype(float))
    y_did = clean_did["approved"].astype(int)

    try:
        res_did = sm.Logit(y_did, X_did).fit(disp=0)
        coef = res_did.params.get("black_treat_post", np.nan)
        p    = res_did.pvalues.get("black_treat_post", np.nan)
        lo   = np.exp(coef - 1.96*res_did.bse.get("black_treat_post", np.nan))
        hi   = np.exp(coef + 1.96*res_did.bse.get("black_treat_post", np.nan))
        print(f"\nTriple interaction (Black x Truist x Post-2021):")
        print(f"  Coef: {coef:.4f}  OR: {np.exp(coef):.4f}  [{lo:.4f},{hi:.4f}]  p={p:.4f}")
        if p < 0.05:
            direction = "widened" if coef < 0 else "narrowed"
            print(f"  Significant at 5%: merger {direction} the Black-White gap at Truist vs peers.")
        else:
            print(f"  Not significant: merger did not measurably change the gap vs peers.")
    except Exception as e:
        print(f"  DiD failed: {e}")

    results_df = pd.DataFrame([r for r in results if r])
    results_df.to_parquet("data/processed/premerger_disparity.parquet", index=False)
    print(f"\nSaved to data/processed/premerger_disparity.parquet")
    print("\nDone.")