import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_truist.parquet"

def prep_expanded(df, institution=None):
    if institution:
        df = df[df["institution"] == institution].copy()
    else:
        df = df.copy()

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

    oc = pd.to_numeric(df["occupancy_type"], errors="coerce")
    df["is_investment"]  = (oc == 3).astype(int)
    df["is_second_home"] = (oc == 2).astype(int)

    df["is_subordinate"]  = (pd.to_numeric(df["lien_status"], errors="coerce") == 2).astype(int)
    df["is_manufactured"] = (pd.to_numeric(df["construction_method"], errors="coerce") == 2).astype(int)

    aus = pd.to_numeric(df["aus-1"], errors="coerce")
    df["aus_du"]     = (aus == 1).astype(int)
    df["aus_lp"]     = (aus == 2).astype(int)
    df["aus_manual"] = (aus.isin([3,4,5,6,7])).astype(int)

    df["loan_term"] = pd.to_numeric(df["loan_term"], errors="coerce")
    df["is_30yr"]   = (df["loan_term"].between(355, 365)).astype(int)
    df["is_conforming"] = (df["conforming_loan_limit"].astype(str).str.lower() == "c").astype(int)

    df["miss_ltv"]    = df["ltv"].isna().astype(int)
    df["miss_dti"]    = df["dti_mid"].isna().astype(int)
    df["miss_income"] = df["log_income"].isna().astype(int)
    df["ltv"]         = df["ltv"].fillna(df["ltv"].median())
    df["dti_mid"]     = df["dti_mid"].fillna(df["dti_mid"].median())
    df["log_income"]  = df["log_income"].fillna(df["log_income"].median())

    return df


CREDIT_FEATURES = [
    "log_income", "log_loan_amount", "dti_mid", "ltv",
    "purpose_purchase", "purpose_refi", "purpose_cashout",
    "is_fha", "is_va", "is_usda",
    "is_investment", "is_second_home", "is_subordinate",
    "is_manufactured", "is_30yr", "is_conforming",
    "aus_du", "aus_lp", "aus_manual",
    "miss_ltv", "miss_dti", "miss_income",
]
RACE_FEATURES = ["is_black", "is_hispanic", "is_asian"]


def run_model(df, features, label="Model"):
    all_cols = features + ["approved"]
    clean = df[all_cols].dropna(subset=["approved"])
    features = [f for f in features if clean[f].std() > 0]
    X = sm.add_constant(clean[features].astype(float))
    y = clean["approved"].astype(int)
    result = sm.Logit(y, X).fit(disp=0)

    print(f"\n{'='*65}")
    print(f"{label}  N={len(clean):,}  PseudoR2={result.prsquared:.4f}")
    print(f"{'='*65}")
    coef_tbl = pd.DataFrame({
        "OR":    np.exp(result.params),
        "CI_lo": np.exp(result.params - 1.96*result.bse),
        "CI_hi": np.exp(result.params + 1.96*result.bse),
        "z":     result.tvalues,
        "p":     result.pvalues,
    }).drop(index="const", errors="ignore")
    race_rows = coef_tbl.loc[[r for r in RACE_FEATURES if r in coef_tbl.index]]
    print("\nRACE COEFFICIENTS:")
    print(race_rows.round(4).to_string())
    return result


def missingness_audit(df):
    print("\n=== MISSINGNESS AUDIT ===")
    for v in ["ltv","dti_mid","log_income","log_loan_amount"]:
        miss  = df[v].isna().mean()
        black = df[df["is_black"]==1][v].isna().mean()
        white = df[df["derived_race"]=="White"][v].isna().mean()
        print(f"{v:20s}  overall={miss:.3f}  Black={black:.3f}  White={white:.3f}")


def aus_breakdown(truist):
    print("\n=== AUS BREAKDOWN ===")
    for label, mask in [
        ("DU (automated)",  truist["aus_du"]==1),
        ("LP (automated)",  truist["aus_lp"]==1),
        ("Manual/other",    truist["aus_manual"]==1),
    ]:
        sub = truist[mask].copy()
        if len(sub) < 500:
            print(f"  {label:25s}  N={len(sub):,}  (too small, skip)")
            continue
        feats = [f for f in CREDIT_FEATURES + RACE_FEATURES if sub[f].std() > 0]
        try:
            res = sm.Logit(
                sub["approved"].astype(int),
                sm.add_constant(sub[feats].astype(float))
            ).fit(disp=0)
            b_or = np.exp(res.params.get("is_black", np.nan))
            b_p  = res.pvalues.get("is_black", np.nan)
            print(f"  {label:25s}  N={len(sub):,}  Black OR={b_or:.4f}  p={b_p:.4f}")
        except Exception as e:
            print(f"  {label:25s}  N={len(sub):,}  Error: {e}")


def loan_size_gradient(truist):
    print("\n=== LOAN SIZE GRADIENT (expanded controls) ===")
    loan_amt = pd.to_numeric(truist["loan_amount"], errors="coerce")
    truist["loan_q"] = pd.qcut(loan_amt, 4, labels=["Q1","Q2","Q3","Q4"])
    for q in ["Q1","Q2","Q3","Q4"]:
        sub = truist[truist["loan_q"]==q].copy()
        feats = [f for f in CREDIT_FEATURES + RACE_FEATURES if sub[f].std() > 0]
        try:
            res = sm.Logit(
                sub["approved"].astype(int),
                sm.add_constant(sub[feats].astype(float))
            ).fit(disp=0)
            b_or = np.exp(res.params.get("is_black", np.nan))
            b_p  = res.pvalues.get("is_black", np.nan)
            med  = loan_amt[truist["loan_q"]==q].median()
            print(f"  {q}  median=${med:,.0f}  N={len(sub):,}  Black OR={b_or:.4f}  p={b_p:.4f}")
        except Exception as e:
            print(f"  {q}  Error: {e}")


if __name__ == "__main__":
    print("Loading HMDA (Truist 2021-2023)...")
    raw    = pd.read_parquet(DATA_PATH)
    truist = prep_expanded(raw, institution="Truist Bank")
    print(f"Rows after action filter: {len(truist):,}")

    missingness_audit(truist)

    r1 = run_model(truist, CREDIT_FEATURES,
                   "Stage 1 — credit only (expanded)")
    r2 = run_model(truist, CREDIT_FEATURES + RACE_FEATURES,
                   "Stage 2 — credit + race (expanded)")

    conventional = truist[
        (truist["is_fha"]==0) & (truist["is_va"]==0) &
        (truist["is_usda"]==0) & (truist["is_manufactured"]==0)
    ].copy()
    r3 = run_model(conventional, CREDIT_FEATURES + RACE_FEATURES,
                   "Stage 2 — conventional only, no manufactured housing")

    aus_breakdown(truist)
    loan_size_gradient(truist)

    pd.DataFrame({
        "OR":    np.exp(r2.params),
        "CI_lo": np.exp(r2.params - 1.96*r2.bse),
        "CI_hi": np.exp(r2.params + 1.96*r2.bse),
        "p":     r2.pvalues,
    }).to_parquet("data/processed/fair_lending_expanded.parquet")
    print("\nSaved. Done.")