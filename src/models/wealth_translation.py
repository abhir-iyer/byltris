"""
Step 1: Economic Translation + Counterfactual Excess Denials
============================================================
For each Black applicant in the Truist 2021-2023 sample:
  - Predict P(approved | is_black=1)  -- actual
  - Predict P(approved | is_black=0)  -- counterfactual (race removed)
  - excess_denial_i = P_counter - P_actual

Aggregate excess denials × wealth accumulation model = dollars foregone.

Wealth model:
  Method A (conservative): median home value × FHFA 10-year nominal appreciation (48%)
  Method B (full equity):  home appreciation + mortgage principal built over
                           NAR median holding period (13 years) at 6.5% rate

Sources:
  - ACS 2022 median home values by state (census.gov/housing)
  - FHFA House Price Index, 10-year cumulative 2013-2023 (+48% nominal)
  - NAR 2023 Profile of Home Buyers and Sellers (median tenure = 13 years)
  - Freddie Mac Primary Mortgage Market Survey avg 2022-2023 (6.5%)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/raw/hmda_truist.parquet"
OUT_PATH   = "data/processed/wealth_impact.parquet"

# ── ACS 2022 median owner-occupied home values by state (dollars) ─────────────
# Source: census.gov American Community Survey 2022 1-year estimates, Table B25077
STATE_HOME_VALUES = {
    "FL": 295000, "NC": 235000, "VA": 339000, "GA": 239000,
    "TN": 229000, "MD": 368000, "SC": 193000, "WV": 145000,
    "KY": 174000, "IN": 183000, "OH": 192000, "NJ": 401000,
    "PA": 222000, "TX": 266000,
}
NATIONAL_MEDIAN = 244900   # ACS 2022 national median, fallback

# ── Wealth accumulation parameters ───────────────────────────────────────────
FHFA_10YR_NOMINAL   = 0.48    # FHFA HPI cumulative nominal gain 2013-2023
HOLDING_YEARS       = 13      # NAR 2023 median homeowner tenure
MORTGAGE_RATE       = 0.065   # Freddie Mac avg 30-yr fixed 2022-2023
DOWN_PAYMENT_SHARE  = 0.10    # 10% down (NAR first-time buyer median)
LOAN_TERM_YEARS     = 30
NOMINAL_APPRECIATION= 0.045   # 4.5% annual nominal appreciation (S&P CS 30-yr avg)

# ── Model features ────────────────────────────────────────────────────────────
CREDIT_FEATURES = [
    "log_income", "log_loan_amount", "dti_mid",
    "purpose_purchase", "purpose_refi", "purpose_cashout",
]
RACE_FEATURES = ["is_black", "is_hispanic", "is_asian"]
ALL_FEATURES  = CREDIT_FEATURES + RACE_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
def prep(df):
    sub = df[df["institution"] == "Truist Bank"].copy()
    sub["activity_year"]  = pd.to_numeric(sub["activity_year"],  errors="coerce")
    sub["action_taken"]   = pd.to_numeric(sub["action_taken"],   errors="coerce")
    sub["loan_amount"]    = pd.to_numeric(sub["loan_amount"],     errors="coerce")
    sub["income"]         = pd.to_numeric(sub["income"],          errors="coerce")
    sub["loan_purpose"]   = sub["loan_purpose"].astype(str)

    sub = sub[
        sub["action_taken"].isin([1, 3]) &
        sub["activity_year"].between(2021, 2023)
    ].copy()
    sub["approved"] = (sub["action_taken"] == 1).astype(int)

    sub["log_income"]      = np.log1p(sub["income"].clip(lower=0))
    sub["log_loan_amount"] = np.log1p(sub["loan_amount"].clip(lower=0))

    def dti_mid(val):
        try:
            v = str(val).replace("%", "").replace("<", "").replace(">", "").strip()
            if "-" in v:
                lo, hi = v.split("-")
                return (float(lo) + float(hi)) / 2
            return float(v)
        except:
            return np.nan
    sub["dti_mid"] = sub["debt_to_income_ratio"].apply(dti_mid)

    sub["is_black"]    = (sub["derived_race"] == "Black or African American").astype(int)
    sub["is_hispanic"] = sub["derived_ethnicity"].str.contains("Hispanic", na=False).astype(int)
    sub["is_asian"]    = (sub["derived_race"] == "Asian").astype(int)

    lp = sub["loan_purpose"]
    sub["purpose_purchase"] = (lp == "1").astype(int)
    sub["purpose_refi"]     = (lp == "31").astype(int)
    sub["purpose_cashout"]  = (lp == "32").astype(int)

    if "state_code" in sub.columns:
        sub["state_code"] = sub["state_code"].astype(str).str.strip().str.upper()
    else:
        sub["state_code"] = "XX"

    sub["home_value"] = sub["state_code"].map(STATE_HOME_VALUES).fillna(NATIONAL_MEDIAN)
    return sub


# ══════════════════════════════════════════════════════════════════════════════
def equity_after_n_years(home_value, n_years=HOLDING_YEARS,
                          rate=MORTGAGE_RATE, down=DOWN_PAYMENT_SHARE,
                          term=LOAN_TERM_YEARS, appr=NOMINAL_APPRECIATION):
    """
    Total equity after n_years of ownership.
    Equity = down payment recovered + principal paid + home appreciation.
    """
    final_value    = home_value * (1 + appr) ** n_years
    appreciation   = final_value - home_value
    down_amount    = home_value * down
    loan           = home_value - down_amount

    # Monthly payment (standard amortization formula)
    r = rate / 12
    N = term * 12
    if r > 0:
        payment = loan * r * (1 + r)**N / ((1 + r)**N - 1)
    else:
        payment = loan / N

    # Accumulate principal paid over n_years months
    balance         = loan
    principal_paid  = 0
    for _ in range(n_years * 12):
        interest        = balance * r
        principal       = min(payment - interest, balance)
        principal_paid += principal
        balance        -= principal
        if balance <= 0:
            break

    total_equity = down_amount + principal_paid + appreciation
    return {
        "total_equity":    total_equity,
        "appreciation":    appreciation,
        "principal_paid":  principal_paid,
        "down_recovered":  down_amount,
    }


# ══════════════════════════════════════════════════════════════════════════════
def run_counterfactual(sub, model):
    """
    For every Black applicant, compute:
        excess_denial_i = P(approved | race=0) - P(approved | race=1)
    Positive value = applicant more likely approved absent racial coefficient.
    """
    black_idx = sub[sub["is_black"] == 1].copy()
    clean     = black_idx[ALL_FEATURES + ["approved", "home_value"]].dropna()

    X_actual  = sm.add_constant(clean[ALL_FEATURES].astype(float), has_constant="add")
    X_counter = X_actual.copy()
    X_counter["is_black"] = 0.0

    p_actual  = model.predict(X_actual).values
    p_counter = model.predict(X_counter).values

    clean = clean.copy()
    clean["p_actual"]       = p_actual
    clean["p_counter"]      = p_counter
    clean["excess_denial"]  = p_counter - p_actual   # > 0 means unfairly denied
    return clean


# ══════════════════════════════════════════════════════════════════════════════
def compute_wealth_impact(cf_df, n_years_sample=3):
    """
    Translate excess denials into aggregate wealth impact.
    """
    n_black           = len(cf_df)
    excess_denials    = cf_df["excess_denial"].sum()          # fractional excess
    avg_excess_prob   = cf_df["excess_denial"].mean()
    annual_excess     = excess_denials / n_years_sample

    # Use the loan_amount column if available; else state median home value
    # We use the home_value we mapped at prep time as the relevant asset
    median_hv = cf_df["home_value"].median()
    mean_hv   = cf_df["home_value"].mean()

    # Method A: conservative — FHFA 10-year nominal appreciation only
    wealth_A_per_denial    = median_hv * FHFA_10YR_NOMINAL
    wealth_A_total_3yr     = excess_denials * wealth_A_per_denial
    wealth_A_annual        = annual_excess  * wealth_A_per_denial

    # Method B: full equity — appreciation + principal over 13-year tenure
    eq = equity_after_n_years(median_hv)
    wealth_B_per_denial    = eq["total_equity"]
    wealth_B_total_3yr     = excess_denials * wealth_B_per_denial
    wealth_B_annual        = annual_excess  * wealth_B_per_denial

    return {
        "n_black_applicants":              n_black,
        "excess_denials_3yr":              excess_denials,
        "excess_denials_annual":           annual_excess,
        "avg_excess_denial_prob":          avg_excess_prob,
        "median_home_value":               median_hv,
        "mean_home_value":                 mean_hv,
        # equity components (Method B)
        "equity_appreciation":             eq["appreciation"],
        "equity_principal":                eq["principal_paid"],
        "equity_down_recovered":           eq["down_recovered"],
        "equity_total_per_homeowner":      eq["total_equity"],
        # Method A
        "wealth_per_denial_conservative":  wealth_A_per_denial,
        "wealth_total_3yr_conservative":   wealth_A_total_3yr,
        "wealth_annual_conservative":      wealth_A_annual,
        # Method B
        "wealth_per_denial_full_equity":   wealth_B_per_denial,
        "wealth_total_3yr_full_equity":    wealth_B_total_3yr,
        "wealth_annual_full_equity":       wealth_B_annual,
    }


# ══════════════════════════════════════════════════════════════════════════════
def print_results(r, model_summary):
    W = 62
    print("\n" + "=" * W)
    print("ECONOMIC IMPACT OF RACIAL APPROVAL GAP")
    print("Truist Bank  |  2021-2023")
    print("=" * W)
    print(f"\n{'Model verification':}")
    print(f"  Black OR (logit):             {model_summary['black_or']:.4f}")
    print(f"  95% CI:                       [{model_summary['ci_lo']:.4f}, {model_summary['ci_hi']:.4f}]")
    print(f"  N (complete cases):           {model_summary['n']:,}")

    print(f"\n{'Counterfactual excess denials':}")
    print(f"  Black applicants in sample:   {r['n_black_applicants']:>8,.0f}")
    print(f"  Excess denials (3-yr total):  {r['excess_denials_3yr']:>8,.0f}")
    print(f"  Excess denials (annual avg):  {r['excess_denials_annual']:>8,.0f}")
    print(f"  Avg excess P(denial) per      ")
    print(f"    Black applicant:            {r['avg_excess_denial_prob']:>8.4f}  ({r['avg_excess_denial_prob']*100:.1f} pp)")

    print(f"\n{'Asset base':}")
    print(f"  Median home value             ")
    print(f"  (Truist states, ACS 2022):    ${r['median_home_value']:>9,.0f}")

    print(f"\n{'Wealth per homeowner (Method B, {HOLDING_YEARS}-yr tenure)':}")
    print(f"  Home appreciation (4.5%/yr):  ${r['equity_appreciation']:>9,.0f}")
    print(f"  Mortgage principal built:     ${r['equity_principal']:>9,.0f}")
    print(f"  Down payment recovered:       ${r['equity_down_recovered']:>9,.0f}")
    print(f"  Total equity after 13 years:  ${r['equity_total_per_homeowner']:>9,.0f}")

    print(f"\n{'Aggregate wealth foregone':}")
    print(f"  Method A (FHFA 10-yr appr.):  ")
    print(f"    Per denied homebuyer:        ${r['wealth_per_denial_conservative']:>9,.0f}")
    print(f"    3-year total:                ${r['wealth_total_3yr_conservative']/1e6:>9.1f}M")
    print(f"    Annual:                      ${r['wealth_annual_conservative']/1e6:>9.1f}M")
    print(f"  Method B (full equity model):")
    print(f"    Per denied homebuyer:        ${r['wealth_per_denial_full_equity']:>9,.0f}")
    print(f"    3-year total:                ${r['wealth_total_3yr_full_equity']/1e6:>9.1f}M")
    print(f"    Annual:                      ${r['wealth_annual_full_equity']/1e6:>9.1f}M")

    print(f"\n{'Implied homeownership rate impact':}")
    # Black homeownership in Truist states (Census 2022 approx): 44%
    # If excess_denials/year applied to all Black households in Truist MSAs:
    # This is illustrative, not a causal claim
    truist_black_hmda_per_yr = r['n_black_applicants'] / 3
    pct_additional = (r['excess_denials_annual'] / truist_black_hmda_per_yr) * 100
    print(f"  Additional approvals/yr as %  ")
    print(f"  of annual Black applicants:   {pct_additional:.1f}%")
    print(f"  (i.e., homeownership rate     ")
    print(f"   would be {pct_additional:.1f} pp higher absent")
    print(f"   the racial approval penalty)")
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)

    sub = prep(df)
    print(f"Truist 2021-2023: {len(sub):,} rows")
    print(f"Black applicants: {sub['is_black'].sum():,}")

    # ── Fit full logit ────────────────────────────────────────────────────────
    clean = sub[ALL_FEATURES + ["approved"]].dropna()
    X     = sm.add_constant(clean[ALL_FEATURES].astype(float), has_constant="add")
    y     = clean["approved"].astype(int)
    model = sm.Logit(y, X).fit(disp=0)

    coef   = model.params["is_black"]
    se     = model.bse["is_black"]
    black_or = np.exp(coef)
    ci_lo    = np.exp(coef - 1.96 * se)
    ci_hi    = np.exp(coef + 1.96 * se)
    model_summary = {"black_or": black_or, "ci_lo": ci_lo,
                     "ci_hi": ci_hi, "n": len(clean)}

    # ── Counterfactual ────────────────────────────────────────────────────────
    cf_df   = run_counterfactual(sub, model)
    results = compute_wealth_impact(cf_df, n_years_sample=3)

    print_results(results, model_summary)

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame([results]).to_parquet(OUT_PATH, index=False)
    cf_df[["p_actual", "p_counter", "excess_denial", "home_value"]].to_parquet(
        "data/processed/counterfactual_individual.parquet", index=False
    )
    print(f"\nSaved to {OUT_PATH}")
    print("Done.")