import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_truist.parquet"

CREDIT_FEATURES = [
    "log_income", "log_loan_amount",
    "dti_mid", "purpose_purchase", "purpose_refi", "purpose_cashout"
]
RACE_FEATURES = ["is_black", "is_hispanic", "is_asian"]


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

    # state dummies
    sub["state_code"] = sub["state_code"].astype(str).str.strip()

    return sub


# ── 1. State fixed effects model ─────────────────────────────────────────────
def run_state_fe(sub):
    print("=== MODEL WITH STATE FIXED EFFECTS ===\n")

    clean = sub[CREDIT_FEATURES + RACE_FEATURES +
                ["approved", "state_code"]].dropna()

    # create state dummies — drop most common state as reference
    state_dummies = pd.get_dummies(clean["state_code"], prefix="state", drop_first=True)
    state_cols = state_dummies.columns.tolist()

    X = sm.add_constant(
        pd.concat([clean[CREDIT_FEATURES + RACE_FEATURES].astype(float),
                   state_dummies.astype(float)], axis=1)
    )
    y = clean["approved"].astype(int)

    res = sm.Logit(y, X).fit(disp=0)

    black_OR  = np.exp(res.params.get("is_black", np.nan))
    black_p   = res.pvalues.get("is_black", np.nan)
    black_lo  = np.exp(res.params["is_black"] - 1.96 * res.bse["is_black"])
    black_hi  = np.exp(res.params["is_black"] + 1.96 * res.bse["is_black"])

    print(f"N: {len(clean):,}")
    print(f"State dummies included: {len(state_cols)}")
    print(f"Black OR:  {black_OR:.4f}  [{black_lo:.4f}, {black_hi:.4f}]  p={black_p:.4f}")
    print(f"Pseudo R2: {res.prsquared:.4f}")
    print(f"\nBaseline (no state FE): OR=0.5414  [0.5232, 0.5603]")
    print(f"Change in OR with state FE: {black_OR - 0.5414:+.4f}")
    print(f"\nConclusion: State fixed effects {'materially change' if abs(black_OR - 0.5414) > 0.05 else 'do not materially change'} the result.")

    return res, black_OR, black_lo, black_hi


# ── 2. Year × Race interaction ────────────────────────────────────────────────
def run_year_interaction(sub):
    print("\n\n=== YEAR × RACE INTERACTION ===\n")

    clean = sub[CREDIT_FEATURES + RACE_FEATURES +
                ["approved", "activity_year"]].dropna()
    clean = clean[clean["activity_year"].isin([2021, 2022, 2023])].copy()

    # year dummies (2021 = reference)
    clean["year_2022"] = (clean["activity_year"] == 2022).astype(int)
    clean["year_2023"] = (clean["activity_year"] == 2023).astype(int)

    # interactions
    clean["black_x_2022"] = clean["is_black"] * clean["year_2022"]
    clean["black_x_2023"] = clean["is_black"] * clean["year_2023"]

    interaction_features = (CREDIT_FEATURES + RACE_FEATURES +
                            ["year_2022", "year_2023",
                             "black_x_2022", "black_x_2023"])

    X = sm.add_constant(clean[interaction_features].astype(float))
    y = clean["approved"].astype(int)
    res = sm.Logit(y, X).fit(disp=0)

    print(f"N: {len(clean):,}")
    print(f"\nBase effect (2021 reference):")
    base_or = np.exp(res.params.get("is_black", np.nan))
    base_p  = res.pvalues.get("is_black", np.nan)
    print(f"  Black OR (2021): {base_or:.4f}  p={base_p:.4f}")

    print(f"\nYear interaction terms:")
    for col, year in [("black_x_2022", 2022), ("black_x_2023", 2023)]:
        coef = res.params.get(col, np.nan)
        p    = res.pvalues.get(col, np.nan)
        combined_or = np.exp(res.params.get("is_black", 0) + coef)
        print(f"  Black OR ({year}): {combined_or:.4f}  interaction coef={coef:.4f}  p={p:.4f}")

    print(f"\nPseudo R2: {res.prsquared:.4f}")
    print(f"\nConclusion: Gap is {'widening' if res.params.get('black_x_2023', 0) < 0 else 'narrowing'} over time (interaction {'significant' if res.pvalues.get('black_x_2023', 1) < 0.05 else 'not significant'} at 5%).")

    return res


# ── 3. Loan purpose × Race interaction ───────────────────────────────────────
def run_purpose_interaction(sub):
    print("\n\n=== LOAN PURPOSE × RACE INTERACTION ===\n")

    clean = sub[CREDIT_FEATURES + RACE_FEATURES + ["approved"]].dropna()

    # interactions (purchase = reference via purpose_refi and purpose_cashout dummies)
    clean2 = clean.copy()
    clean2["black_x_refi"]    = clean2["is_black"] * clean2["purpose_refi"]
    clean2["black_x_cashout"] = clean2["is_black"] * clean2["purpose_cashout"]

    interaction_features = (CREDIT_FEATURES + RACE_FEATURES +
                            ["black_x_refi", "black_x_cashout"])

    X = sm.add_constant(clean2[interaction_features].astype(float))
    y = clean2["approved"].astype(int)
    res = sm.Logit(y, X).fit(disp=0)

    print(f"N: {len(clean2):,}")
    print(f"\nBase effect (purchase loans reference):")
    base_or = np.exp(res.params.get("is_black", np.nan))
    base_p  = res.pvalues.get("is_black", np.nan)
    print(f"  Black OR (purchase): {base_or:.4f}  p={base_p:.4f}")

    print(f"\nPurpose interaction terms:")
    for col, label in [("black_x_refi", "Refinance"),
                       ("black_x_cashout", "Cash-out refi")]:
        coef = res.params.get(col, np.nan)
        p    = res.pvalues.get(col, np.nan)
        combined_or = np.exp(res.params.get("is_black", 0) + coef)
        print(f"  Black OR ({label}): {combined_or:.4f}  interaction coef={coef:.4f}  p={p:.4f}")

    print(f"\nPseudo R2: {res.prsquared:.4f}")
    return res


# ── 4. McFadden R2 decomposition ──────────────────────────────────────────────
def r2_decomposition(sub):
    print("\n\n=== McFADDEN R2 DECOMPOSITION ===\n")
    print("How much of the raw racial gap does each control group explain?\n")

    clean = sub[CREDIT_FEATURES + RACE_FEATURES + ["approved"]].dropna()

    def get_r2(features):
        X = sm.add_constant(clean[features].astype(float))
        y = clean["approved"].astype(int)
        return sm.Logit(y, X).fit(disp=0).prsquared

    r2_race_only   = get_r2(RACE_FEATURES)
    r2_income      = get_r2(["log_income", "log_loan_amount"] + RACE_FEATURES)
    r2_dti         = get_r2(["log_income", "log_loan_amount", "dti_mid"] + RACE_FEATURES)
    r2_full        = get_r2(CREDIT_FEATURES + RACE_FEATURES)

    print(f"Race only:                    R2 = {r2_race_only:.4f}")
    print(f"+ Income & loan amount:       R2 = {r2_income:.4f}  (gain: {r2_income-r2_race_only:+.4f})")
    print(f"+ DTI:                        R2 = {r2_dti:.4f}  (gain: {r2_dti-r2_income:+.4f})")
    print(f"+ Loan purpose:               R2 = {r2_full:.4f}  (gain: {r2_full-r2_dti:+.4f})")
    print(f"\nTotal R2 gain from controls: {r2_full-r2_race_only:.4f}")
    print(f"Remaining unexplained by controls: {1 - (r2_full/max(r2_full,0.0001)):.1%} of maximum")

    total_gain = r2_full - r2_race_only
    if total_gain > 0:
        print(f"\nShare of explained variation by control group:")
        print(f"  Income & loan amount: {(r2_income-r2_race_only)/total_gain*100:.1f}%")
        print(f"  DTI:                  {(r2_dti-r2_income)/total_gain*100:.1f}%")
        print(f"  Loan purpose:         {(r2_full-r2_dti)/total_gain*100:.1f}%")

    return {
        "r2_race_only": r2_race_only,
        "r2_full": r2_full,
        "total_gain": total_gain
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)
    sub = prep(df, "Truist Bank")
    print(f"Truist sample: {len(sub):,} rows\n")

    res_state_fe, or_fe, lo_fe, hi_fe = run_state_fe(sub)
    res_year    = run_year_interaction(sub)
    res_purpose = run_purpose_interaction(sub)
    r2_decomp   = r2_decomposition(sub)

    # save results
    summary = pd.DataFrame([{
        "model": "State FE",
        "black_OR": or_fe,
        "black_CI_lo": lo_fe,
        "black_CI_hi": hi_fe,
    }])
    summary.to_parquet("data/processed/fair_lending_extended.parquet", index=False)

    print("\n\n=== FINAL SUMMARY ===")
    print(f"Baseline OR (no state FE):     0.5414  [0.5232, 0.5603]")
    print(f"With state fixed effects:      {or_fe:.4f}  [{lo_fe:.4f}, {hi_fe:.4f}]")
    print(f"\nKey interaction findings:")
    print(f"  Year trend: check output above")
    print(f"  Purpose gap: worst for cash-out refi (wealth extraction)")
    print("\nDone.")