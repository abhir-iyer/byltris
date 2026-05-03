import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_full_panel.parquet"

INSTITUTIONS = [
    "Truist Bank", "Bank of America", "Wells Fargo", "JPMorgan Chase",
    "Regions Bank", "PNC Bank", "U.S. Bank", "Fifth Third Bank",
    "Huntington Bank", "Citizens Bank"
]

CREDIT = [
    "log_income", "log_loan_amount", "dti_mid", "ltv",
    "purpose_purchase", "purpose_refi", "purpose_cashout",
    "is_fha", "is_va", "is_conforming", "is_manufactured",
    "aus_du", "aus_lp", "aus_manual",
]

def prep(df):
    df = df[df["institution"].isin(INSTITUTIONS)].copy()
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
    df["dti_mid"] = df["dti_mid"].fillna(df["dti_mid"].median())

    df["log_income"]      = df["log_income"].fillna(df["log_income"].median())
    df["log_loan_amount"] = df["log_loan_amount"].fillna(0)

    df["ltv"] = pd.to_numeric(df["loan_to_value_ratio"], errors="coerce")
    df["ltv"] = df["ltv"].fillna(df["ltv"].median())

    lt = pd.to_numeric(df["loan_type"], errors="coerce")
    df["is_fha"] = (lt == 2).astype(int)
    df["is_va"]  = (lt == 3).astype(int)

    aus = pd.to_numeric(df["aus-1"], errors="coerce")
    df["aus_du"]     = (aus == 1).astype(int)
    df["aus_lp"]     = (aus == 2).astype(int)
    df["aus_manual"] = (aus.isin([3,4,5,6,7])).astype(int)

    df["is_conforming"]   = (df["conforming_loan_limit"].astype(str).str.lower() == "c").astype(int)
    df["is_manufactured"] = (pd.to_numeric(df["construction_method"], errors="coerce") == 2).astype(int)

    lp = df["loan_purpose"].astype(str)
    df["purpose_purchase"] = (lp == "1").astype(int)
    df["purpose_refi"]     = (lp == "31").astype(int)
    df["purpose_cashout"]  = (lp == "32").astype(int)

    df["year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    return df


if __name__ == "__main__":
    print("Loading full panel (2018-2023)...")
    raw = pd.read_parquet(DATA_PATH)
    df  = prep(raw)
    df[CREDIT] = df[CREDIT].fillna(0)
    print(f"N = {len(df):,}")
    print(f"Institutions: {df['institution'].value_counts().to_dict()}")

    # ── Institution fixed effects ─────────────────────────────────────
    print("\n=== POOLED MODEL WITH INSTITUTION FE ===")
    inst_dummies = pd.get_dummies(df["institution"], prefix="inst", drop_first=True)
    year_dummies = pd.get_dummies(df["year"].astype(int), prefix="yr",  drop_first=True)

    feats = CREDIT + ["is_black","is_hispanic","is_asian"]
    feats = [f for f in feats if df[f].std() > 0]

    X = sm.add_constant(
        pd.concat([df[feats], inst_dummies, year_dummies], axis=1).astype(float)
    )
    X = X[[c for c in X.columns if X[c].std() > 0]]

    r_fe = sm.OLS(df["approved"].astype(float), X).fit(cov_type="HC3")

    b_or = r_fe.params.get("is_black", np.nan)
    b_se = r_fe.bse.get("is_black", np.nan)
    b_p  = r_fe.pvalues.get("is_black", np.nan)
    print(f"Black coef (LPM, inst+year FE): {b_or:.4f}  SE={b_se:.4f}  p={b_p:.4f}")
    print(f"= {b_or*100:.2f} pp lower approval probability for Black applicants")

    # ── Institution × Race interactions ───────────────────────────────
    print("\n=== INSTITUTION × RACE INTERACTIONS ===")
    print("(Does the racial gap vary significantly across institutions?)")

    results = []
    ref_inst = "Bank of America"

    for inst in INSTITUTIONS:
        sub = df[df["institution"] == inst].copy()
        if len(sub) < 1000:
            continue
        sub_feats = [f for f in CREDIT + ["is_black","is_hispanic","is_asian"]
                     if sub[f].std() > 0]
        yr_dum = pd.get_dummies(sub["year"].astype(int), prefix="yr", drop_first=True)
        X_sub  = sm.add_constant(
            pd.concat([sub[sub_feats], yr_dum], axis=1).astype(float)
        )
        X_sub = X_sub[[c for c in X_sub.columns if X_sub[c].std() > 0]]
        try:
            r = sm.OLS(sub["approved"].astype(float), X_sub).fit(cov_type="HC3")
            b  = r.params.get("is_black", np.nan)
            se = r.bse.get("is_black", np.nan)
            p  = r.pvalues.get("is_black", np.nan)
            n  = len(sub)
            results.append({
                "institution": inst, "N": n,
                "black_coef": b, "se": se, "p": p,
                "ci_lo": b - 1.96*se, "ci_hi": b + 1.96*se,
            })
        except Exception as e:
            print(f"  {inst}: Error — {e}")

    res_df = pd.DataFrame(results).sort_values("black_coef")
    print(f"\n{'Institution':<25} {'N':>8} {'Black coef':>12} {'95% CI':>22} {'p':>8}")
    print("-"*80)
    for _, row in res_df.iterrows():
        print(f"{row['institution']:<25} {int(row['N']):>8} {row['black_coef']:>12.4f} "
              f"[{row['ci_lo']:.4f},{row['ci_hi']:.4f}]  {row['p']:>8.4f}")

    # ── Test: does gap correlate with institution size? ───────────────
    print("\n=== INSTITUTION SIZE VS GAP CORRELATION ===")
    inst_size = df.groupby("institution")["approved"].count().reset_index()
    inst_size.columns = ["institution","N"]
    merged = res_df.merge(inst_size, on="institution")
    corr = merged[["N_x","black_coef"]].corr().iloc[0,1]
    print(f"Pearson correlation (institution size vs Black coef): {corr:.4f}")
    if corr > 0.3:
        print("Positive correlation — larger institutions show smaller gap (consistent with standardization)")
    elif corr < -0.3:
        print("Negative correlation — larger institutions show larger gap")
    else:
        print("No strong correlation between size and gap")

    # Save
    res_df.to_parquet("data/processed/pooled_cross_institution.parquet", index=False)
    print("\nSaved. Done.")