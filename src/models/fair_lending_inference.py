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


def prep(df, institution):
    sub = df[df["institution"] == institution].copy()

    sub["log_income"]      = np.log1p(pd.to_numeric(sub["income"], errors="coerce").clip(lower=0))
    sub["log_loan_amount"] = np.log1p(pd.to_numeric(sub["loan_amount"], errors="coerce").clip(lower=0))
    sub["action_taken"]    = pd.to_numeric(sub["action_taken"], errors="coerce")
    sub = sub[sub["action_taken"].isin([1, 3])].copy()
    sub["approved"] = (sub["action_taken"] == 1).astype(int)

    def dti_mid(val):
        try:
            if "-" in str(val):
                lo, hi = str(val).replace("%","").split("-")
                return (float(lo) + float(hi)) / 2
            return float(str(val).replace("%","").replace("<","").replace(">","").strip())
        except:
            return np.nan
    sub["dti_mid"] = sub["debt_to_income_ratio"].apply(dti_mid)

    sub["is_black"]    = (sub["derived_race"] == "Black or African American").astype(int)
    sub["is_hispanic"] = sub["derived_ethnicity"].str.contains("Hispanic", na=False).astype(int)
    sub["is_asian"]    = (sub["derived_race"] == "Asian").astype(int)
    sub["is_white"]    = (sub["derived_race"] == "White").astype(int)

    if "loan_purpose" in sub.columns:
        lp = sub["loan_purpose"].astype(str)
        sub["purpose_purchase"] = (lp == "1").astype(int)
        sub["purpose_refi"]     = (lp == "31").astype(int)
        sub["purpose_cashout"]  = (lp == "32").astype(int)
    else:
        for c in ["purpose_purchase","purpose_refi","purpose_cashout"]:
            sub[c] = 0

    return sub


def run_stage(sub, features, label="Stage"):
    all_cols = features + ["approved"]
    clean = sub[all_cols].dropna()
    X = sm.add_constant(clean[features].astype(float))
    y = clean["approved"].astype(int)

    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    print(f"\n{'='*60}")
    print(f"{label}  (N={len(clean):,})")
    print(f"{'='*60}")
    coef_table = pd.DataFrame({
        "coef":    result.params,
        "OR":      np.exp(result.params),
        "se":      result.bse,
        "z":       result.tvalues,
        "p":       result.pvalues,
        "OR_lo95": np.exp(result.params - 1.96*result.bse),
        "OR_hi95": np.exp(result.params + 1.96*result.bse),
    }).drop(index="const", errors="ignore")
    print(coef_table.round(4).to_string())
    print(f"\nPseudo R2: {result.prsquared:.4f}")
    return result


def peer_comparison(df):
    all_features = CREDIT_FEATURES + RACE_FEATURES
    rows = []
    for inst in df["institution"].unique():
        sub = prep(df, inst)
        clean = sub[all_features + ["approved"]].dropna()
        if len(clean) < 500:
            continue
        X = sm.add_constant(clean[all_features].astype(float))
        y = clean["approved"].astype(int)
        res = sm.Logit(y, X).fit(disp=0)
        rows.append({
            "institution":  inst,
            "N":            len(clean),
            "black_OR":     np.exp(res.params.get("is_black", np.nan)),
            "black_p":      res.pvalues.get("is_black", np.nan),
            "black_CI_lo":  np.exp(res.params.get("is_black", np.nan) - 1.96*res.bse.get("is_black", np.nan)),
            "black_CI_hi":  np.exp(res.params.get("is_black", np.nan) + 1.96*res.bse.get("is_black", np.nan)),
        })
    table = pd.DataFrame(rows).sort_values("black_OR")
    print("\n\nPeer comparison with confidence intervals:")
    print(table.round(4).to_string(index=False))
    table.to_parquet("data/processed/fair_lending_inference.parquet", index=False)
    return table


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH)
    truist = prep(df, "Truist Bank")

    s1 = run_stage(truist, CREDIT_FEATURES, "Stage 1 — credit factors only [Truist]")
    s2 = run_stage(truist, CREDIT_FEATURES + RACE_FEATURES, "Stage 2 — credit + race [Truist]")
    peer_comparison(df)
    print("\nDone.")