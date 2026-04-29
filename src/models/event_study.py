"""
Step 4: Dynamic DiD — Event Study for the BB&T/SunTrust Merger
===============================================================
Upgrades the static triple DiD in the paper to a full event study.

Instead of a single post-merger indicator, we estimate year-by-year
triple interactions (Black × Truist × year_t) with 2018 as the
reference year. This:

  1. Directly tests parallel trends: pre-merger coefficients (2019)
     should be close to zero if the assumption holds.
  2. Shows the dynamic treatment trajectory (2020-2023).
  3. Is the current standard in applied econometrics for DiD designs
     (Callaway & Sant'Anna 2021; Sun & Abraham 2021).

We also compute the institution-year panel of Black ORs separately,
which enables a visual synthetic control comparison.

Merger timeline:
  Feb 2019   - Merger announced (BB&T + SunTrust)
  Dec 2019   - Merger completed, Truist Bank formed
  2020-2023  - Post-merger integration period
  Reference  - 2018 (last full pre-announcement year)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_extended.parquet"
OUT_PATH  = "data/processed/event_study.parquet"

CREDIT_FEATURES = [
    "log_income", "log_loan_amount", "dti_mid",
    "purpose_purchase", "purpose_refi", "purpose_cashout",
]
RACE_FEATURES = ["is_black", "is_hispanic", "is_asian"]
ALL_FEATURES  = CREDIT_FEATURES + RACE_FEATURES

PEERS = [
    "Wells Fargo", "Bank of America", "JPMorgan Chase",
    "Regions Bank", "PNC Bank", "U.S. Bank",
]
TREATMENT = "Truist Bank"
REF_YEAR  = 2018
ALL_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


# ══════════════════════════════════════════════════════════════════════════════
def prep_row(df_inst):
    sub = df_inst.copy()
    sub["log_income"]      = np.log1p(pd.to_numeric(sub["income"], errors="coerce").clip(lower=0))
    sub["log_loan_amount"] = np.log1p(pd.to_numeric(sub["loan_amount"], errors="coerce").clip(lower=0))
    sub["action_taken"]    = pd.to_numeric(sub["action_taken"], errors="coerce")
    sub["activity_year"]   = pd.to_numeric(sub["activity_year"], errors="coerce")
    sub = sub[sub["action_taken"].isin([1, 3])].copy()
    sub["approved"] = (sub["action_taken"] == 1).astype(int)

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

    lp = sub["loan_purpose"].astype(str)
    sub["purpose_purchase"] = (lp == "1").astype(int)
    sub["purpose_refi"]     = (lp == "31").astype(int)
    sub["purpose_cashout"]  = (lp == "32").astype(int)
    return sub


def logit_or(sub, label=""):
    clean = sub[ALL_FEATURES + ["approved"]].dropna()
    if len(clean) < 300 or clean["is_black"].sum() < 30:
        return None
    X = sm.add_constant(clean[ALL_FEATURES].astype(float), has_constant="add")
    y = clean["approved"].astype(int)
    try:
        res = sm.Logit(y, X).fit(disp=0)
        coef = res.params.get("is_black", np.nan)
        se   = res.bse.get("is_black", np.nan)
        return {
            "label":    label,
            "n":        len(clean),
            "black_OR": np.exp(coef),
            "ci_lo":    np.exp(coef - 1.96 * se),
            "ci_hi":    np.exp(coef + 1.96 * se),
            "black_p":  res.pvalues.get("is_black", np.nan),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
def build_institution_year_panel(df):
    """
    Compute Black OR for every (institution, year) cell.
    This is the outcome panel for the visual synthetic control.
    """
    print("Building institution-year panel...")
    panel = []
    institutions = [TREATMENT] + PEERS
    for inst in institutions:
        sub_inst = prep_row(df[df["institution"] == inst])
        for yr in ALL_YEARS:
            sub_yr = sub_inst[sub_inst["activity_year"] == yr]
            r = logit_or(sub_yr, f"{inst} {yr}")
            if r:
                panel.append({
                    "institution": inst,
                    "year":        yr,
                    "is_truist":   int(inst == TREATMENT),
                    "black_OR":    r["black_OR"],
                    "ci_lo":       r["ci_lo"],
                    "ci_hi":       r["ci_hi"],
                    "n":           r["n"],
                })
    return pd.DataFrame(panel)


# ══════════════════════════════════════════════════════════════════════════════
def dynamic_did(df):
    """
    Triple dynamic DiD:
    Outcome: approved (binary)
    Model: logit with credit controls + year dummies + triple interactions

    Key interactions (Black × Truist × year_t) for t = 2019, 2020, 2021, 2022, 2023
    Reference: Black × Truist × 2018 (normalized to zero)

    Pre-merger: 2019 (should be ~0 if parallel trends hold)
    Post-merger: 2020, 2021, 2022, 2023
    """
    institutions = [TREATMENT] + PEERS
    did_raw = df[df["institution"].isin(institutions)].copy()
    did_raw = prep_row(did_raw)
    did_raw = did_raw[did_raw["activity_year"].isin(ALL_YEARS)].copy()

    did_raw["treat"] = (did_raw["institution"] == TREATMENT).astype(int)

    # Year dummies (2018 = reference)
    for yr in ALL_YEARS:
        did_raw[f"yr{yr}"] = (did_raw["activity_year"] == yr).astype(int)

    # Black × Truist × year_t interactions (triple)
    year_interaction_cols = []
    for yr in ALL_YEARS:
        if yr == REF_YEAR:
            continue
        col = f"black_truist_yr{yr}"
        did_raw[col] = did_raw["is_black"] * did_raw["treat"] * did_raw[f"yr{yr}"]
        year_interaction_cols.append((yr, col))

    # Black × year and Truist × year and Black × Truist (lower-order terms)
    lower_order = []
    for yr in ALL_YEARS:
        if yr == REF_YEAR:
            continue
        col_by = f"black_yr{yr}"
        col_ty = f"truist_yr{yr}"
        did_raw[col_by] = did_raw["is_black"] * did_raw[f"yr{yr}"]
        did_raw[col_ty] = did_raw["treat"]    * did_raw[f"yr{yr}"]
        lower_order += [col_by, col_ty]

    did_raw["black_truist"] = did_raw["is_black"] * did_raw["treat"]

    # Year dummies (excluding 2018)
    year_dummies = [f"yr{yr}" for yr in ALL_YEARS if yr != REF_YEAR]

    features = (ALL_FEATURES
                + ["treat"]
                + year_dummies
                + lower_order
                + ["black_truist"]
                + [col for _, col in year_interaction_cols])

    clean = did_raw[features + ["approved"]].dropna()
    print(f"Dynamic DiD sample: {len(clean):,} rows")

    X = sm.add_constant(clean[features].astype(float), has_constant="add")
    y = clean["approved"].astype(int)
    res = sm.Logit(y, X).fit(disp=0)

    results = []
    for yr, col in year_interaction_cols:
        coef = res.params.get(col, np.nan)
        se   = res.bse.get(col, np.nan)
        pval = res.pvalues.get(col, np.nan)
        results.append({
            "year":         yr,
            "coef":         coef,
            "OR":           np.exp(coef),
            "ci_lo":        np.exp(coef - 1.96 * se),
            "ci_hi":        np.exp(coef + 1.96 * se),
            "p":            pval,
            "period":       "pre" if yr < 2020 else "post",
            "merger_year":  2020,
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
def simple_synthetic_control(panel_df):
    """
    Approximate synthetic control using the institution-year panel.
    Finds donor weights that minimize the pre-merger gap between
    Truist's Black OR and the weighted average of donor institutions.
    Pre-merger: 2018-2019 (reference + 1 year before completion)
    """
    from scipy.optimize import minimize

    pivot = panel_df.pivot(index="year", columns="institution", values="black_OR").dropna()
    if TREATMENT not in pivot.columns:
        return None, None

    truist   = pivot[TREATMENT].values
    donors   = pivot[PEERS].values
    years    = pivot.index.tolist()

    pre_mask  = np.array([y < 2020 for y in years])
    post_mask = ~pre_mask

    if pre_mask.sum() < 1:
        return None, None

    def objective(w):
        synthetic = donors @ w
        return np.sum((truist[pre_mask] - synthetic[pre_mask])**2)

    n_donors  = donors.shape[1]
    w0        = np.ones(n_donors) / n_donors
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0, 1)] * n_donors

    result = minimize(objective, w0, method="SLSQP",
                     bounds=bounds, constraints=constraints,
                     options={"ftol": 1e-9, "maxiter": 1000})

    if not result.success:
        return None, None

    weights    = result.x
    synthetic  = donors @ weights

    donor_names = [c for c in PEERS if c in pivot.columns]
    weight_series = pd.Series(weights, index=donor_names).round(4)

    gap_df = pd.DataFrame({
        "year":      years,
        "truist_OR": truist,
        "synth_OR":  synthetic,
        "gap":       truist - synthetic,
        "pre":       pre_mask,
    })
    return gap_df, weight_series


# ══════════════════════════════════════════════════════════════════════════════
def print_results(event_df, panel_df, gap_df, weights):
    W = 70
    print("\n" + "=" * W)
    print("DYNAMIC DiD — EVENT STUDY")
    print(f"Truist Bank vs {len(PEERS)} peer institutions  |  2018-2023")
    print(f"Reference year: {REF_YEAR}  |  Merger completed: Dec 2019")
    print("=" * W)

    print(f"\nTriple interaction coefficients (Black × Truist × year_t):")
    print(f"  {'Year':<8} {'Period':<10} {'OR':>8} {'95% CI':>20} {'p':>8}  {'Interpretation'}")
    print(f"  {'-'*70}")
    for _, row in event_df.sort_values("year").iterrows():
        period_label = "PRE " if row["period"] == "pre" else "POST"
        sig = "***" if row["p"] < 0.001 else ("**" if row["p"] < 0.01 else
              ("*" if row["p"] < 0.05 else "n.s."))
        direction = ""
        if row["period"] == "post":
            direction = "gap narrowed" if row["OR"] > 1 else "gap widened"
        elif row["period"] == "pre":
            direction = "parallel trends OK" if abs(row["OR"] - 1) < 0.05 else "pre-trend!"
        print(f"  {int(row['year']):<8} {period_label:<10} {row['OR']:>8.4f} "
              f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] {row['p']:>8.4f} {sig}  {direction}")

    # Test for pre-trends
    pre = event_df[event_df["period"] == "pre"]
    print(f"\nPre-merger parallel trends test:")
    if len(pre) > 0:
        max_pre_dev = (pre["OR"] - 1).abs().max()
        all_ns = (pre["p"] > 0.05).all()
        print(f"  Max pre-merger deviation from OR=1: {max_pre_dev:.4f}")
        print(f"  All pre-merger interactions non-significant: {all_ns}")
        if all_ns and max_pre_dev < 0.05:
            print(f"  CONCLUSION: Parallel trends assumption is supported.")
        elif all_ns:
            print(f"  CONCLUSION: Statistically non-significant but some deviation — interpret with care.")
        else:
            print(f"  WARNING: Pre-merger interactions are significant — parallel trends may not hold.")

    print(f"\nInstitution-year Black OR panel:")
    print(f"  {'Institution':<25} " + "  ".join(f"{y}" for y in ALL_YEARS))
    print(f"  {'-'*70}")
    for inst in [TREATMENT] + PEERS:
        row_vals = []
        for yr in ALL_YEARS:
            val = panel_df.loc[
                (panel_df["institution"] == inst) &
                (panel_df["year"] == yr), "black_OR"
            ]
            row_vals.append(f"{val.values[0]:.3f}" if len(val) > 0 else " n.a.")
        marker = " <-- TREATED" if inst == TREATMENT else ""
        print(f"  {inst:<25} " + "  ".join(row_vals) + marker)

    if gap_df is not None and weights is not None:
        print(f"\nApproximate synthetic control:")
        print(f"  Donor weights:")
        for inst, w in weights.items():
            if w > 0.01:
                print(f"    {inst:<25} {w:.4f}")
        print(f"\n  {'Year':<8} {'Truist OR':>10} {'Synthetic OR':>14} {'Gap':>8} {'Period'}")
        print(f"  {'-'*50}")
        for _, row in gap_df.iterrows():
            period = "PRE" if row["pre"] else "POST"
            print(f"  {int(row['year']):<8} {row['truist_OR']:>10.4f} "
                  f"{row['synth_OR']:>14.4f} {row['gap']:>8.4f}  {period}")
        post_rows = gap_df[~gap_df["pre"]]
        if len(post_rows) > 0:
            avg_post_gap = post_rows["gap"].mean()
            print(f"\n  Average post-merger gap (Truist - Synthetic): {avg_post_gap:.4f}")
            if avg_post_gap > 0:
                print(f"  Truist OR is ABOVE synthetic control post-merger")
                print(f"  (i.e., gap narrowed MORE than synthetic counterfactual predicts)")
            else:
                print(f"  Truist OR is BELOW synthetic control post-merger")
                print(f"  (i.e., gap widened relative to synthetic counterfactual)")
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)

    print("Loading extended HMDA data (2018-2023)...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)
    print(f"Shape: {df.shape}")

    # Step 1: Institution-year panel
    panel_df = build_institution_year_panel(df)
    panel_df.to_parquet("data/processed/institution_year_panel.parquet", index=False)
    print(f"Panel: {len(panel_df)} cells ({panel_df['institution'].nunique()} institutions × {panel_df['year'].nunique()} years)")

    # Step 2: Dynamic DiD
    print("\nRunning dynamic DiD...")
    event_df = dynamic_did(df)

    # Step 3: Synthetic control
    print("\nRunning approximate synthetic control...")
    gap_df, weights = simple_synthetic_control(panel_df)

    # Print
    print_results(event_df, panel_df, gap_df, weights)

    # Save
    event_df.to_parquet(OUT_PATH, index=False)
    if gap_df is not None:
        gap_df.to_parquet("data/processed/synthetic_control_gap.parquet", index=False)
    print(f"\nSaved to {OUT_PATH}")
    print("Done.")