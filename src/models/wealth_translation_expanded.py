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
    df["dti_mid"] = df["dti_mid"].fillna(df["dti_mid"].median())
    df["log_income"] = df["log_income"].fillna(df["log_income"].median())

    lp = df["loan_purpose"].astype(str)
    df["purpose_purchase"] = (lp == "1").astype(int)
    df["purpose_refi"]     = (lp == "31").astype(int)
    df["purpose_cashout"]  = (lp == "32").astype(int)

    df["ltv"] = pd.to_numeric(df["loan_to_value_ratio"], errors="coerce")
    df["ltv"] = df["ltv"].fillna(df["ltv"].median())

    lt = pd.to_numeric(df["loan_type"], errors="coerce")
    df["is_fha"]  = (lt == 2).astype(int)
    df["is_va"]   = (lt == 3).astype(int)
    df["is_usda"] = (lt == 4).astype(int)

    oc = pd.to_numeric(df["occupancy_type"], errors="coerce")
    df["is_investment"]  = (oc == 3).astype(int)
    df["is_second_home"] = (oc == 2).astype(int)

    df["is_manufactured"] = (pd.to_numeric(df["construction_method"], errors="coerce") == 2).astype(int)
    df["is_conforming"]   = (df["conforming_loan_limit"].astype(str).str.lower() == "c").astype(int)

    aus = pd.to_numeric(df["aus-1"], errors="coerce")
    df["aus_du"]     = (aus == 1).astype(int)
    df["aus_lp"]     = (aus == 2).astype(int)
    df["aus_manual"] = (aus.isin([3,4,5,6,7])).astype(int)

    df["loan_term"] = pd.to_numeric(df["loan_term"], errors="coerce")
    df["is_30yr"]   = (df["loan_term"].between(355, 365)).astype(int)

    return df


CREDIT = [
    "log_income", "log_loan_amount", "dti_mid", "ltv",
    "purpose_purchase", "purpose_refi", "purpose_cashout",
    "is_fha", "is_va", "is_usda",
    "is_investment", "is_second_home", "is_manufactured",
    "is_30yr", "is_conforming",
    "aus_du", "aus_lp", "aus_manual",
]
RACE = ["is_black", "is_hispanic", "is_asian"]

# Wealth parameters
MEDIAN_HOME_VALUE   = 239_000   # ACS 2022, Truist states
FHFA_10YR_APPR      = 0.48      # 2013-2023 nominal cumulative
HOLDING_YEARS       = 13        # NAR 2023 median tenure
MORTGAGE_RATE       = 0.065     # Freddie Mac 2022-2023 average
HOME_APPR_ANNUAL    = 0.045     # annual appreciation
DOWN_PCT            = 0.10      # 10% down payment


if __name__ == "__main__":
    print("Loading Truist HMDA data...")
    raw    = pd.read_parquet(DATA_PATH)
    truist = prep(raw)
    truist[CREDIT + RACE] = truist[CREDIT + RACE].fillna(0)
    print(f"N = {len(truist):,}")

    # ── Fit expanded logit ────────────────────────────────────────────
    feats = [f for f in CREDIT + RACE if truist[f].std() > 0]
    X = sm.add_constant(truist[feats].astype(float))
    y = truist["approved"].astype(int)
    print("Fitting expanded logit...")
    result = sm.Logit(y, X).fit(disp=0)

    b_or = np.exp(result.params.get("is_black", np.nan))
    b_p  = result.pvalues.get("is_black", np.nan)
    print(f"Black OR (expanded controls): {b_or:.4f}  p={b_p:.4f}")

    # ── Counterfactual excess denials ─────────────────────────────────
    print("\nComputing counterfactual excess denials...")
    black_apps = truist[truist["is_black"] == 1].copy()
    # use exact columns from fitted model
    model_cols = result.model.exog_names

    black_apps_X = sm.add_constant(black_apps[feats].astype(float))
# align columns exactly
    black_apps_X = black_apps_X.reindex(columns=model_cols, fill_value=0)

    prob_actual         = result.predict(black_apps_X)
    X_counter           = black_apps_X.copy()
    X_counter["is_black"] = 0
    prob_counterfactual = result.predict(X_counter)

    excess_prob = prob_counterfactual - prob_actual
    total_excess_denials = excess_prob.sum()
    annual_excess        = total_excess_denials / 3
    avg_excess_pp        = excess_prob.mean()

    print(f"\n{'='*60}")
    print(f"EXCESS DENIALS (expanded controls)")
    print(f"{'='*60}")
    print(f"Black applicants in sample:    {len(black_apps):,}")
    print(f"Excess denials (3-yr total):   {total_excess_denials:,.0f}")
    print(f"Excess denials (annual avg):   {annual_excess:,.0f}")
    print(f"Avg excess denial prob/person: {avg_excess_pp*100:.2f} pp")

    # ── Wealth translation ────────────────────────────────────────────
    # Method A: FHFA 10yr appreciation only
    wealth_A = MEDIAN_HOME_VALUE * FHFA_10YR_APPR
    annual_wealth_A = annual_excess * wealth_A

    # Method B: Full equity model
    loan_amt     = MEDIAN_HOME_VALUE * (1 - DOWN_PCT)
    down_payment = MEDIAN_HOME_VALUE * DOWN_PCT
    monthly_rate = MORTGAGE_RATE / 12
    n_payments   = 30 * 12
    monthly_pmt  = loan_amt * (monthly_rate * (1+monthly_rate)**n_payments) / ((1+monthly_rate)**n_payments - 1)
    principal_paid = sum(
        monthly_pmt - (loan_amt - sum(
            monthly_pmt - (loan_amt * monthly_rate * (1+monthly_rate)**k / ((1+monthly_rate)**n_payments - 1))
            for k in range(j)
        )) * monthly_rate
        for j in range(int(HOLDING_YEARS * 12))
    ) if False else loan_amt - loan_amt * (1+monthly_rate)**(n_payments) / ((1+monthly_rate)**(n_payments)) * 0  # simplified below

    # Simplified: remaining balance after HOLDING_YEARS years
    n_held = HOLDING_YEARS * 12
    remaining = loan_amt * ((1+monthly_rate)**n_payments - (1+monthly_rate)**n_held) / ((1+monthly_rate)**n_payments - 1)
    principal_paid_B = loan_amt - remaining
    appreciation_B   = MEDIAN_HOME_VALUE * ((1 + HOME_APPR_ANNUAL)**HOLDING_YEARS - 1)
    wealth_B         = appreciation_B + principal_paid_B + down_payment
    annual_wealth_B  = annual_excess * wealth_B

    print(f"\n{'='*60}")
    print(f"WEALTH TRANSLATION")
    print(f"{'='*60}")
    print(f"Median home value (ACS 2022):  ${MEDIAN_HOME_VALUE:,}")
    print(f"Method A (FHFA 10yr appr):     ${wealth_A:,.0f} per homeowner")
    print(f"Method B (full equity model):  ${wealth_B:,.0f} per homeowner")
    print(f"  Appreciation:                ${appreciation_B:,.0f}")
    print(f"  Principal paid:              ${principal_paid_B:,.0f}")
    print(f"  Down payment recovered:      ${down_payment:,.0f}")
    print(f"\nAnnual wealth foregone:")
    print(f"  Method A: ${annual_wealth_A/1e6:.1f}M")
    print(f"  Method B: ${annual_wealth_B/1e6:.1f}M")

    # ── Sensitivity to conversion rate ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"POLICY COUNTERFACTUAL: CREDIT HISTORY EQUALIZATION")
    print(f"{'='*60}")
    # excess credit history citations from multinomial model
    excess_ch_citations = 4976  # from fair_lending_multinomial.py
    print(f"Excess credit history citations (3-yr): {excess_ch_citations:,}")
    print(f"\n{'Scenario':<20} {'Conv rate':>10} {'Add. owners/yr':>16} {'Wealth A/yr':>14} {'Wealth B/yr':>14}")
    print("-"*76)
    for label, conv in [("Conservative", 0.50), ("Mid-range", 0.65), ("Generous", 0.80)]:
        add_owners = excess_ch_citations * conv / 3
        w_A = add_owners * wealth_A / 1e6
        w_B = add_owners * wealth_B / 1e6
        print(f"{label:<20} {conv:>10.0%} {add_owners:>16,.0f} ${w_A:>12.1f}M ${w_B:>12.1f}M")

    # save
    out = pd.DataFrame([{
        "black_OR_expanded":       b_or,
        "black_p_expanded":        b_p,
        "excess_denials_3yr":      total_excess_denials,
        "excess_denials_annual":   annual_excess,
        "avg_excess_pp":           avg_excess_pp,
        "wealth_per_owner_A":      wealth_A,
        "wealth_per_owner_B":      wealth_B,
        "annual_wealth_foregone_A": annual_wealth_A,
        "annual_wealth_foregone_B": annual_wealth_B,
    }])
    out.to_parquet("data/processed/wealth_translation_expanded.parquet", index=False)
    print("\nSaved. Done.")