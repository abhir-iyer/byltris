import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_full_panel.parquet"

CONTROL_INSTITUTIONS = [
    "Wells Fargo", "Bank of America", "JPMorgan Chase",
    "Regions Bank", "PNC Bank", "U.S. Bank"
]

CREDIT = [
    "log_income","log_loan_amount","dti_mid","ltv",
    "purpose_purchase","purpose_refi","purpose_cashout",
    "is_fha","is_va","aus_du","aus_lp","aus_manual",
    "is_conforming","is_manufactured",
]

def prep(df):
    truist_terms = ["truist", "suntrust", "bb&t", "bbt"]
    mask = (
        df["institution"].str.lower().str.contains("|".join(truist_terms), na=False) |
        df["institution"].isin(CONTROL_INSTITUTIONS)
    )
    df = df[mask].copy()

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

    df["dti_mid"]     = df["debt_to_income_ratio"].apply(dti_mid)
    df["dti_mid"]     = df["dti_mid"].fillna(df["dti_mid"].median())
    df["log_income"]  = df["log_income"].fillna(df["log_income"].median())

    lp = df["loan_purpose"].astype(str)
    df["purpose_purchase"] = (lp == "1").astype(int)
    df["purpose_refi"]     = (lp == "31").astype(int)
    df["purpose_cashout"]  = (lp == "32").astype(int)

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

    df["year"]      = pd.to_numeric(df["activity_year"], errors="coerce")
    df["is_truist"] = df["institution"].str.lower().str.contains(
        "truist|suntrust|bb.t", na=False
    ).astype(int)

    return df


if __name__ == "__main__":
    print("Loading HMDA panel (2018-2023)...")
    raw = pd.read_parquet(DATA_PATH)
    df  = prep(raw)
    df[CREDIT] = df[CREDIT].fillna(0)

    print(f"Panel N = {len(df):,}")
    print(f"Years: {sorted(df['year'].dropna().unique().astype(int).tolist())}")
    print(f"\nInstitution counts:")
    print(df.groupby("institution")["approved"].count().sort_values(ascending=False).to_string())

    # ── Static triple DiD (LPM, clustered SEs) ───────────────────────
    print("\n=== STATIC DiD (LPM, clustered by institution) ===")
    df["post"] = (df["year"] >= 2021).astype(int)
    df["black_x_truist"]          = df["is_black"] * df["is_truist"]
    df["black_x_post"]            = df["is_black"] * df["post"]
    df["truist_x_post"]           = df["is_truist"] * df["post"]
    df["black_x_truist_x_post"]   = df["is_black"] * df["is_truist"] * df["post"]

    static_feats = CREDIT + [
        "is_black","is_truist","post",
        "black_x_truist","black_x_post","truist_x_post",
        "black_x_truist_x_post"
    ]
    static_feats = [f for f in static_feats if df[f].std() > 0]
    X_static = sm.add_constant(df[static_feats].astype(float))
    r_static  = sm.OLS(df["approved"].astype(float), X_static).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["institution"]}
    )
    coef = r_static.params.get("black_x_truist_x_post", np.nan)
    se   = r_static.bse.get("black_x_truist_x_post", np.nan)
    pv   = r_static.pvalues.get("black_x_truist_x_post", np.nan)
    ci   = (coef - 1.96*se, coef + 1.96*se)
    print(f"Triple interaction (Black x Truist x Post):")
    print(f"  Coef={coef:.4f}  SE={se:.4f}  p={pv:.4f}  CI=[{ci[0]:.4f},{ci[1]:.4f}]")
    print(f"  = {coef*100:.2f} pp change in Black-White approval gap at Truist post-merger")

    # ── Dynamic event study (year-by-year, reference=2018) ───────────
    print("\n=== DYNAMIC EVENT STUDY (year-by-year, reference=2018) ===")
    years = sorted([int(y) for y in df["year"].dropna().unique() if int(y) != 2018])

    for yr in years:
        df[f"yr{yr}"]                    = (df["year"] == yr).astype(int)
        df[f"black_x_truist_x_yr{yr}"]  = df["is_black"] * df["is_truist"] * df[f"yr{yr}"]

    yr_dummies  = [f"yr{y}" for y in years]
    yr_interact = [f"black_x_truist_x_yr{y}" for y in years]
    base        = ["is_black","is_truist","black_x_truist","black_x_post","truist_x_post"]

    dyn_feats = CREDIT + base + yr_dummies + yr_interact
    dyn_feats = [f for f in dyn_feats if df[f].std() > 0]

    X_dyn = sm.add_constant(df[dyn_feats].astype(float))
    r_dyn  = sm.OLS(df["approved"].astype(float), X_dyn).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["institution"]}
    )

    print(f"\n{'Year':<6} {'Coef':>8} {'SE':>8} {'CI lo':>8} {'CI hi':>8} {'p':>8} {'Sig':<6} {'Period'}")
    print("-"*68)
    event_results = []
    for yr in years:
        key = f"black_x_truist_x_yr{yr}"
        if key not in r_dyn.params.index:
            print(f"{yr:<6} (dropped — collinear)")
            continue
        c  = r_dyn.params[key]
        s  = r_dyn.bse[key]
        p  = r_dyn.pvalues[key]
        lo = c - 1.96*s
        hi = c + 1.96*s
        period = "PRE " if yr < 2021 else "POST"
        sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"{yr:<6} {c:>8.4f} {s:>8.4f} {lo:>8.4f} {hi:>8.4f} {p:>8.4f} {sig:<6} {period}")
        event_results.append({
            "year": yr, "coef": c, "se": s,
            "ci_lo": lo, "ci_hi": hi, "p": p,
            "period": period.strip(), "sig": sig
        })

    # Pre-trend test
    print("\nPre-trend test (years before 2021):")
    pre = [r for r in event_results if r["period"] == "PRE"]
    all_pass = all(r["p"] > 0.05 for r in pre)
    for r in pre:
        status = "PASSES" if r["p"] > 0.05 else "FAILS"
        print(f"  {r['year']}: coef={r['coef']:.4f}  p={r['p']:.4f}  {status}")
    print(f"  Overall pre-trend: {'PASSES' if all_pass else 'MIXED'}")

    # Save
    pd.DataFrame(event_results).to_parquet("data/processed/event_study_improved.parquet")
    print("\nSaved. Done.")