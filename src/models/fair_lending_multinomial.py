import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

    df["ltv"] = pd.to_numeric(df["loan_to_value_ratio"], errors="coerce").fillna(
        pd.to_numeric(df["loan_to_value_ratio"], errors="coerce").median()
    )
    lt = pd.to_numeric(df["loan_type"], errors="coerce")
    df["is_fha"]  = (lt == 2).astype(int)
    df["is_va"]   = (lt == 3).astype(int)

    aus = pd.to_numeric(df["aus-1"], errors="coerce")
    df["aus_du"]     = (aus == 1).astype(int)
    df["aus_lp"]     = (aus == 2).astype(int)
    df["aus_manual"] = (aus.isin([3,4,5,6,7])).astype(int)

    # denial reasons — HMDA codes:
    # 1=DTI, 2=employment, 3=credit history, 4=collateral,
    # 5=insufficient cash, 6=unverifiable, 7=credit app incomplete,
    # 8=mortgage insurance, 9=other, 10=not applicable
    df["dr1"] = pd.to_numeric(df["denial_reason-1"], errors="coerce")

    # primary denial reason category
    def categorize_denial(row):
        if row["approved"] == 1:
            return "originated"
        dr = row["dr1"]
        if dr == 3:
            return "credit_history"
        elif dr == 1:
            return "dti"
        elif dr == 4:
            return "collateral"
        elif dr == 7:
            return "incomplete"
        elif pd.isna(dr) or dr == 10:
            return "other_denial"
        else:
            return "other_denial"
    df["outcome"] = df.apply(categorize_denial, axis=1)

    return df


CREDIT = [
    "log_income", "log_loan_amount", "dti_mid", "ltv",
    "purpose_purchase", "purpose_refi", "purpose_cashout",
    "is_fha", "is_va", "aus_du", "aus_lp", "aus_manual",
]
RACE = ["is_black", "is_hispanic", "is_asian"]


if __name__ == "__main__":
    print("Loading...")
    raw    = pd.read_parquet(DATA_PATH)
    truist = prep(raw)
    truist[CREDIT + RACE] = truist[CREDIT + RACE].fillna(0)
    print(f"N = {len(truist):,}")

    print("\n=== OUTCOME DISTRIBUTION ===")
    print(truist["outcome"].value_counts())

    # ── Approach: binary logit for each denial type vs originated ─────
    # This directly answers: conditional on financial controls,
    # are Black applicants more likely to receive each denial type?
    print("\n=== BINARY LOGIT: each denial category vs originated ===")
    print("(Sample: full Truist 2021-2023, reference = originated)")
    print()

    results = []
    feats = [f for f in CREDIT + RACE if truist[f].std() > 0]

    for denial_type in ["credit_history", "dti", "collateral", "incomplete", "other_denial"]:
        # binary: 1 = this denial type, 0 = originated
        sub = truist[truist["outcome"].isin(["originated", denial_type])].copy()
        sub["y"] = (sub["outcome"] == denial_type).astype(int)
        if sub["y"].sum() < 100:
            print(f"  {denial_type:20s}  N too small, skip")
            continue
        sub_feats = [f for f in feats if sub[f].std() > 0]
        try:
            res = sm.Logit(
                sub["y"].astype(int),
                sm.add_constant(sub[sub_feats].astype(float))
            ).fit(disp=0)
            b_or  = np.exp(res.params.get("is_black", np.nan))
            b_lo  = np.exp(res.params.get("is_black", np.nan) - 1.96*res.bse.get("is_black", np.nan))
            b_hi  = np.exp(res.params.get("is_black", np.nan) + 1.96*res.bse.get("is_black", np.nan))
            b_p   = res.pvalues.get("is_black", np.nan)
            n_pos = int(sub["y"].sum())
            print(f"  {denial_type:20s}  N_denied={n_pos:,}  Black OR={b_or:.4f}  [{b_lo:.4f},{b_hi:.4f}]  p={b_p:.4f}")
            results.append({
                "denial_type": denial_type,
                "N_denied": n_pos,
                "black_OR": b_or,
                "ci_lo": b_lo,
                "ci_hi": b_hi,
                "p": b_p,
            })
        except Exception as e:
            print(f"  {denial_type:20s}  Error: {e}")

    # ── Among denied only: relative denial reason ─────────────────────
    print("\n=== CONDITIONAL ON DENIAL: credit history vs DTI ===")
    print("(Sample: denied applicants only)")
    denied = truist[truist["approved"] == 0].copy()
    denied["y_ch"]  = (denied["dr1"] == 3).astype(int)  # credit history
    denied["y_dti"] = (denied["dr1"] == 1).astype(int)  # DTI

    for label, yvar in [("Credit history denial", "y_ch"), ("DTI denial", "y_dti")]:
        d_feats = [f for f in feats if denied[f].std() > 0]
        try:
            res = sm.Logit(
                denied[yvar].astype(int),
                sm.add_constant(denied[d_feats].astype(float))
            ).fit(disp=0)
            b_or = np.exp(res.params.get("is_black", np.nan))
            b_lo = np.exp(res.params.get("is_black", np.nan) - 1.96*res.bse.get("is_black", np.nan))
            b_hi = np.exp(res.params.get("is_black", np.nan) + 1.96*res.bse.get("is_black", np.nan))
            b_p  = res.pvalues.get("is_black", np.nan)
            print(f"  {label:30s}  N={len(denied):,}  Black OR={b_or:.4f}  [{b_lo:.4f},{b_hi:.4f}]  p={b_p:.4f}")
            results.append({
                "denial_type": label,
                "N_denied": len(denied),
                "black_OR": b_or,
                "ci_lo": b_lo,
                "ci_hi": b_hi,
                "p": b_p,
            })
        except Exception as e:
            print(f"  {label}  Error: {e}")

    pd.DataFrame(results).to_parquet("data/processed/fair_lending_multinomial.parquet")
    print("\nSaved. Done.")