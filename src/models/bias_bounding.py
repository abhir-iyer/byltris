import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

"""
Oster (2019) / Altonji et al. (2005) bias bounding.

The core question: how strong would omitted variable bias need to be
to explain away the Black OR = 0.541 finding?

We use two approaches:
1. Coefficient stability test — how much does the Black coefficient
   move when we add more controls? If it barely moves, omitted bias
   is unlikely to be large.
2. Oster delta — what ratio of unobservable to observable selection
   would be needed to reduce the effect to zero?
"""

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


def run_model(sub, features, label):
    clean = sub[features + ["approved"]].dropna()
    X = sm.add_constant(clean[features].astype(float))
    y = clean["approved"].astype(int)
    res = sm.Logit(y, X).fit(disp=0)
    coef  = res.params.get("is_black", np.nan)
    r2    = res.prsquared
    return coef, r2, len(clean), res


# ── 1. Coefficient stability across nested models ────────────────────────────
def coefficient_stability(sub):
    print("=== COEFFICIENT STABILITY (nested models) ===\n")

    # Model 1: race only (no controls)
    c1, r1, n1, _ = run_model(sub, RACE_FEATURES, "Race only")
    or1 = np.exp(c1)
    print(f"Model 1 — Race only (no controls)")
    print(f"  Black coef: {c1:.4f}  OR: {or1:.4f}  Pseudo-R2: {r1:.4f}  N: {n1:,}\n")

    # Model 2: income + loan amount only
    c2, r2, n2, _ = run_model(sub, ["log_income","log_loan_amount"] + RACE_FEATURES, "Income + loan")
    or2 = np.exp(c2)
    print(f"Model 2 — Income + loan amount")
    print(f"  Black coef: {c2:.4f}  OR: {or2:.4f}  Pseudo-R2: {r2:.4f}  N: {n2:,}\n")

    # Model 3: add DTI
    c3, r3, n3, _ = run_model(sub, ["log_income","log_loan_amount","dti_mid"] + RACE_FEATURES, "+ DTI")
    or3 = np.exp(c3)
    print(f"Model 3 — + DTI")
    print(f"  Black coef: {c3:.4f}  OR: {or3:.4f}  Pseudo-R2: {r3:.4f}  N: {n3:,}\n")

    # Model 4: full controls (baseline)
    c4, r4, n4, _ = run_model(sub, CREDIT_FEATURES + RACE_FEATURES, "Full controls")
    or4 = np.exp(c4)
    print(f"Model 4 — Full controls (baseline)")
    print(f"  Black coef: {c4:.4f}  OR: {or4:.4f}  Pseudo-R2: {r4:.4f}  N: {n4:,}\n")

    # coefficient movement
    movement = abs(c1 - c4) / abs(c1) * 100
    print(f"Coefficient movement from Model 1 to Model 4: {movement:.1f}%")
    print(f"(If < 30%, omitted variable bias is unlikely to be large — Altonji et al. 2005)\n")

    return {
        "coefs": [c1, c2, c3, c4],
        "ors":   [or1, or2, or3, or4],
        "r2s":   [r1, r2, r3, r4],
        "movement_pct": movement
    }


# ── 2. Oster (2019) delta ─────────────────────────────────────────────────────
def oster_delta(sub):
    """
    Oster (2019): compute delta — the ratio of unobservable to observable
    selection on treatment needed to fully explain away the coefficient.

    If delta > 1, omitted variable bias would need to be STRONGER than
    the selection on observables to nullify the finding.
    Convention: delta > 1 is considered a robust result.

    Formula (linear approximation):
    delta = (beta_restricted * (R_max - R_full)) /
            ((beta_uncontrolled - beta_restricted) * (R_full - R_uncontrolled))

    We use R_max = 2 * R_full as a conservative upper bound (Oster 2019 recommends
    1.3 * R_full; we use 2 * R_full to be conservative).
    """
    print("=== OSTER (2019) DELTA ===\n")

    # uncontrolled: race only
    c_u, r_u, _, _ = run_model(sub, RACE_FEATURES, "uncontrolled")

    # controlled: full model
    c_c, r_c, _, _ = run_model(sub, CREDIT_FEATURES + RACE_FEATURES, "controlled")

    # R_max candidates
    r_max_1_3 = min(1.3 * r_c, 0.99)   # Oster recommended
    r_max_2_0 = min(2.0 * r_c, 0.99)   # conservative

    for r_max_label, r_max in [("1.3 x R_full (Oster recommended)", r_max_1_3),
                                 ("2.0 x R_full (conservative)",       r_max_2_0)]:
        numerator   = c_c * (r_max - r_c)
        denominator = (c_u - c_c) * (r_c - r_u)

        if abs(denominator) < 1e-10:
            print(f"R_max = {r_max_label}: denominator near zero, cannot compute")
            continue

        delta = numerator / denominator

        print(f"R_max = {r_max_label}")
        print(f"  Beta uncontrolled: {c_u:.4f}  (OR={np.exp(c_u):.4f})")
        print(f"  Beta controlled:   {c_c:.4f}  (OR={np.exp(c_c):.4f})")
        print(f"  R2 uncontrolled:   {r_u:.4f}")
        print(f"  R2 controlled:     {r_c:.4f}")
        print(f"  R_max:             {r_max:.4f}")
        print(f"  Delta:             {delta:.4f}")
        if delta > 1:
            print(f"  Interpretation: Unobservables would need to be {delta:.1f}x STRONGER")
            print(f"  than observables to nullify the finding. Result is ROBUST.")
        else:
            print(f"  Interpretation: Delta < 1 — omitted variable concern remains.")
        print()

    return delta


# ── 3. Bartlett et al. calibration ───────────────────────────────────────────
def bartlett_calibration():
    """
    Bartlett et al. (2022, RFS) show that FinTech lenders — who rely
    entirely on algorithmic credit scoring — show roughly 40% smaller
    racial gaps than face-to-face lenders using the same HMDA data.

    Interpretation: the portion of the HMDA gap attributable to
    unobserved credit quality is approximately 40% of the raw gap.

    We apply this to our credit-adjusted OR to produce a
    credit-score-adjusted lower bound.
    """
    print("=== BARTLETT ET AL. (2022) CALIBRATION ===\n")

    actual_coef  = -0.6135   # from our statsmodels Stage 2
    actual_or    = np.exp(actual_coef)

    # Bartlett: FinTech gap is ~40% smaller than face-to-face
    # implying ~40% of the gap is explained by unobserved credit quality
    reduction_pct = 0.40

    adjusted_coef = actual_coef * (1 - reduction_pct)
    adjusted_or   = np.exp(adjusted_coef)

    print(f"Actual credit-adjusted Black coefficient: {actual_coef:.4f}  (OR={actual_or:.4f})")
    print(f"Bartlett et al. reduction factor:         {reduction_pct*100:.0f}%")
    print(f"Implied credit-score-adjusted coefficient: {adjusted_coef:.4f}  (OR={adjusted_or:.4f})")
    print(f"\nInterpretation: Even after applying the Bartlett et al. (2022) calibration,")
    print(f"the implied Black OR is {adjusted_or:.3f}, still indicating substantially")
    print(f"lower approval odds for Black applicants after both observed and")
    print(f"estimated unobserved credit quality differences are accounted for.")

    return adjusted_or


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)
    sub = prep(df, "Truist Bank")
    print(f"Truist sample: {len(sub):,} rows\n")

    stability = coefficient_stability(sub)
    delta     = oster_delta(sub)
    adj_or    = bartlett_calibration()

    print("\n=== SUMMARY FOR PAPER ===")
    print(f"Black OR (no controls):    {stability['ors'][0]:.4f}")
    print(f"Black OR (full controls):  {stability['ors'][3]:.4f}")
    print(f"Coefficient movement:      {stability['movement_pct']:.1f}%")
    print(f"Oster delta:               {delta:.4f}")
    print(f"Bartlett-adjusted OR:      {adj_or:.4f}")
    print("\nDone.")