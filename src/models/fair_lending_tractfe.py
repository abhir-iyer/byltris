import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_truist.parquet"

def prep(df):
    df = df[df["institution"] == "Truist Bank"].copy()
    df["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce")
    df = df[df["action_taken"].isin([1, 3])].copy()
    df["approved"] = (df["action_taken"] == 1).astype(int)

    df["is_black"]    = (df["derived_race"] == "Black or African American").astype(int)
    df["is_hispanic"] = df["derived_ethnicity"].str.contains("Hispanic", na=False).astype(int)
    df["is_asian"]    = (df["derived_race"] == "Asian").astype(int)

    df["log_income"]      = np.log1p(pd.to_numeric(df["income"], errors="coerce").clip(lower=0))
    df["log_loan_amount"] = np.log1p(pd.to_numeric(df["loan_amount"], errors="coerce").clip(lower=0))

    def dti_mid(val):
        try:
            if "-" in str(val):
                lo, hi = str(val).replace("%","").split("-")
                return (float(lo)+float(hi))/2
            return float(str(val).replace("%","").replace("<","").replace(">","").strip())
        except:
            return np.nan
    df["dti_mid"] = df["debt_to_income_ratio"].apply(dti_mid)

    lp = df["loan_purpose"].astype(str)
    df["purpose_purchase"] = (lp == "1").astype(int)
    df["purpose_refi"]     = (lp == "31").astype(int)
    df["purpose_cashout"]  = (lp == "32").astype(int)

    df["ltv"] = pd.to_numeric(df["loan_to_value_ratio"], errors="coerce")
    lt = pd.to_numeric(df["loan_type"], errors="coerce")
    df["is_fha"]  = (lt == 2).astype(int)
    df["is_va"]   = (lt == 3).astype(int)
    df["is_usda"] = (lt == 4).astype(int)

    aus = pd.to_numeric(df["aus-1"], errors="coerce")
    df["aus_du"]     = (aus == 1).astype(int)
    df["aus_lp"]     = (aus == 2).astype(int)
    df["aus_manual"] = (aus.isin([3,4,5,6,7])).astype(int)

    df["is_conforming"]  = (df["conforming_loan_limit"].astype(str).str.lower() == "c").astype(int)
    df["is_manufactured"]= (pd.to_numeric(df["construction_method"], errors="coerce") == 2).astype(int)
    oc = pd.to_numeric(df["occupancy_type"], errors="coerce")
    df["is_investment"]  = (oc == 3).astype(int)
    df["loan_term"]      = pd.to_numeric(df["loan_term"], errors="coerce")
    df["is_30yr"]        = (df["loan_term"].between(355, 365)).astype(int)

    df["ltv"]      = df["ltv"].fillna(df["ltv"].median())
    df["dti_mid"]  = df["dti_mid"].fillna(df["dti_mid"].median())

    # tract identifier
    df["tract"] = df["census_tract"].astype(str).str.strip()

    return df


CREDIT = [
    "log_income", "log_loan_amount", "dti_mid", "ltv",
    "purpose_purchase", "purpose_refi", "purpose_cashout",
    "is_fha", "is_va", "is_usda",
    "is_investment", "is_manufactured", "is_30yr", "is_conforming",
    "aus_du", "aus_lp", "aus_manual",
]
RACE = ["is_black", "is_hispanic", "is_asian"]


if __name__ == "__main__":
    print("Loading...")
    raw    = pd.read_parquet(DATA_PATH)
    truist = prep(raw)
    truist[CREDIT + RACE] = truist[CREDIT + RACE].fillna(0)
    print(f"N = {len(truist):,}")

    # ── Model A: no tract FE (baseline for comparison) ──────────────
    feats_A = [f for f in CREDIT + RACE if truist[f].std() > 0]
    truist[feats_A] = truist[feats_A].fillna(0)
    X_A = sm.add_constant(truist[feats_A].astype(float))
    y   = truist["approved"].astype(int)
    r_A = sm.Logit(y, X_A).fit(disp=0)
    or_A = np.exp(r_A.params["is_black"])
    ci_A = np.exp(r_A.params["is_black"] - 1.96*r_A.bse["is_black"]), \
           np.exp(r_A.params["is_black"] + 1.96*r_A.bse["is_black"])
    print(f"\nModel A (no tract FE):   Black OR={or_A:.4f}  CI=[{ci_A[0]:.4f},{ci_A[1]:.4f}]  p={r_A.pvalues['is_black']:.4f}")

    # ── Model B: state FE ────────────────────────────────────────────
    state_dummies = pd.get_dummies(truist["state_code"].astype(str), prefix="state", drop_first=True)
    feats_B = feats_A + list(state_dummies.columns)
    X_B = sm.add_constant(
        pd.concat([truist[feats_A].astype(float), state_dummies.astype(float)], axis=1)
    )
    r_B = sm.Logit(y, X_B).fit(disp=0)
    or_B = np.exp(r_B.params["is_black"])
    ci_B = np.exp(r_B.params["is_black"] - 1.96*r_B.bse["is_black"]), \
           np.exp(r_B.params["is_black"] + 1.96*r_B.bse["is_black"])
    print(f"Model B (state FE):      Black OR={or_B:.4f}  CI=[{ci_B[0]:.4f},{ci_B[1]:.4f}]  p={r_B.pvalues['is_black']:.4f}")

    county_counts = truist["county_code"].value_counts()
    keep_counties = county_counts[county_counts >= 50].index
    truist["county_fe"] = truist["county_code"].where(truist["county_code"].isin(keep_counties), other="other")
    # ── Model C: county FE ───────────────────────────────────────────
    # use county_code — more granular than state, less collinear than tract
    # ── Model C: county FE (within-county demeaning) ─────────────────
    # Demean all variables by county to absorb county FE
    # This is equivalent to county FE without creating dummy columns
    cols_to_demean = feats_A + ["approved"]
    county_means = truist.groupby("county_fe")[cols_to_demean].transform("mean")
    truist_within = truist[cols_to_demean] - county_means

    # LPM within estimator
    from sklearn.linear_model import LinearRegression
    X_C = truist_within[feats_A].values
    y_C = truist_within["approved"].values
    lpm = LinearRegression().fit(X_C, y_C)

    # get SE via OLS in statsmodels on demeaned data
    r_C = sm.OLS(y_C, sm.add_constant(X_C)).fit(cov_type="HC3")
    feat_idx = feats_A.index("is_black")
    or_C = r_C.params[feat_idx + 1]  # +1 for const
    se_C  = r_C.bse[feat_idx + 1]
    ci_C  = (or_C - 1.96*se_C, or_C + 1.96*se_C)
    pv_C  = r_C.pvalues[feat_idx + 1]
    print(f"Model C (county FE, LPM): Black coef={or_C:.4f}  CI=[{ci_C[0]:.4f},{ci_C[1]:.4f}]  p={pv_C:.4f}")
    print("  (LPM coefficient, not OR — interpret as percentage point change in approval probability)")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n=== SUMMARY: GEOGRAPHIC FE ROBUSTNESS ===")
    print(f"{'Specification':<30} {'Black OR/coef':>14} {'95% CI':>22} {'p':>8}")
    print("-"*78)
    print(f"{'No geographic FE':<30} {or_A:>14.4f} [{ci_A[0]:.4f},{ci_A[1]:.4f}]  {r_A.pvalues['is_black']:>8.4f}")
    print(f"{'State FE':<30} {or_B:>14.4f} [{ci_B[0]:.4f},{ci_B[1]:.4f}]  {r_B.pvalues['is_black']:>8.4f}")
    print(f"{'County FE (LPM coef)':<30} {or_C:>14.4f} [{ci_C[0]:.4f},{ci_C[1]:.4f}]  {pv_C:>8.4f}")
    print("\nNote: County FE uses within-county demeaned LPM; coefficient is pp change, not OR.")

    # save
    results = pd.DataFrame([
        {"spec": "no_geo_fe",  "black_OR": or_A, "ci_lo": ci_A[0], "ci_hi": ci_A[1], "p": r_A.pvalues["is_black"]},
        {"spec": "state_fe",   "black_OR": or_B, "ci_lo": ci_B[0], "ci_hi": ci_B[1], "p": r_B.pvalues["is_black"]},
        {"spec": "county_fe_lpm", "black_OR": or_C, "ci_lo": ci_C[0], "ci_hi": ci_C[1], "p": pv_C},
    ])
    results.to_parquet("data/processed/fair_lending_geo_fe.parquet")
    print("\nSaved. Done.")