import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_truist.parquet"

def prep(df):
    df = df.copy()
    df["log_income"]      = np.log1p(pd.to_numeric(df["income"], errors="coerce").clip(lower=0))
    df["log_loan_amount"] = np.log1p(pd.to_numeric(df["loan_amount"], errors="coerce").clip(lower=0))
    df["action_taken"]    = pd.to_numeric(df["action_taken"], errors="coerce")
    df = df[df["action_taken"].isin([1, 3])].copy()
    df["approved"] = (df["action_taken"] == 1).astype(int)

    def dti_mid(val):
        try:
            if "-" in str(val):
                lo, hi = str(val).replace("%","").split("-")
                return (float(lo) + float(hi)) / 2
            return float(str(val).replace("%","").replace("<","").replace(">","").strip())
        except:
            return np.nan
    df["dti_mid"] = df["debt_to_income_ratio"].apply(dti_mid)

    df["is_black"]    = (df["derived_race"] == "Black or African American").astype(int)
    df["is_hispanic"] = df["derived_ethnicity"].str.contains("Hispanic", na=False).astype(int)
    df["is_asian"]    = (df["derived_race"] == "Asian").astype(int)

    if "loan_purpose" in df.columns:
        lp = df["loan_purpose"].astype(str)
        df["purpose_purchase"] = (lp == "1").astype(int)
        df["purpose_refi"]     = (lp == "31").astype(int)
        df["purpose_cashout"]  = (lp == "32").astype(int)
    else:
        for c in ["purpose_purchase","purpose_refi","purpose_cashout"]:
            df[c] = 0

    # loan size quartile
    loan_amt = pd.to_numeric(df["loan_amount"], errors="coerce")
    df["loan_q1"] = (loan_amt <= loan_amt.quantile(0.25)).astype(int)
    df["loan_q4"] = (loan_amt >= loan_amt.quantile(0.75)).astype(int)

    return df

if __name__ == "__main__":
    print("Loading HMDA data...")
    df = pd.read_parquet(DATA_PATH)
    
    # filter to Truist only if institution column exists
    if "institution" in df.columns:
        df = df[df["institution"] == "Truist Bank"].copy()
    
    df = prep(df)

    FEATURES = [
        "log_income", "log_loan_amount", "dti_mid",
        "purpose_purchase", "purpose_refi", "purpose_cashout",
        "is_black", "is_hispanic", "is_asian",
        "loan_q1", "loan_q4"
    ]
    INTERACTION_FEATURES = FEATURES + ["is_black_x_loan_q1", "is_black_x_dti"]

    df["is_black_x_loan_q1"] = df["is_black"] * df["loan_q1"]
    df["is_black_x_dti"]     = df["is_black"] * df["dti_mid"]

    clean = df[INTERACTION_FEATURES + ["approved"]].dropna()
    print(f"Clean sample: {len(clean):,}")

    X = sm.add_constant(clean[INTERACTION_FEATURES].astype(float))
    y = clean["approved"].astype(int)

    print("\nFitting interaction model...")
    result = sm.Logit(y, X).fit(disp=0)

    # pull key coefficients
    coefs = pd.DataFrame({
        "coef":    result.params,
        "OR":      np.exp(result.params),
        "se":      result.bse,
        "z":       result.tvalues,
        "p":       result.pvalues,
        "OR_lo95": np.exp(result.params - 1.96*result.bse),
        "OR_hi95": np.exp(result.params + 1.96*result.bse),
    }).drop(index="const", errors="ignore")

    print("\n" + "="*65)
    print("INTERACTION MODEL RESULTS")
    print("="*65)
    print(coefs.round(4).to_string())
    print(f"\nPseudo R2: {result.prsquared:.4f}")

    print("\n" + "="*65)
    print("KEY INTERACTION TERMS")
    print("="*65)
    for key in ["is_black", "loan_q1", "is_black_x_loan_q1", "is_black_x_dti"]:
        if key in coefs.index:
            row = coefs.loc[key]
            print(f"\n{key}:")
            print(f"  OR = {row['OR']:.4f}  [{row['OR_lo95']:.4f}, {row['OR_hi95']:.4f}]  p = {row['p']:.4f}")

    print("\nInterpretation:")
    if "is_black_x_loan_q1" in coefs.index:
        or_val = coefs.loc["is_black_x_loan_q1","OR"]
        p_val  = coefs.loc["is_black_x_loan_q1","p"]
        if p_val < 0.05 and or_val < 1.0:
            print(f"  is_black x loan_q1 OR={or_val:.4f} (p={p_val:.4f})")
            print("  -> Racial penalty is SIGNIFICANTLY LARGER for small loans")
            print("     after controlling for DTI and all other financial variables.")
        elif p_val < 0.05 and or_val > 1.0:
            print(f"  is_black x loan_q1 OR={or_val:.4f} (p={p_val:.4f})")
            print("  -> Racial penalty is SMALLER for small loans (unexpected).")
        else:
            print(f"  is_black x loan_q1 OR={or_val:.4f} (p={p_val:.4f})")
            print("  -> Interaction not significant at conventional level.")

    if "is_black_x_dti" in coefs.index:
        or_val = coefs.loc["is_black_x_dti","OR"]
        p_val  = coefs.loc["is_black_x_dti","p"]
        print(f"\n  is_black x dti OR={or_val:.4f} (p={p_val:.4f})")
        if p_val < 0.05:
            direction = "LOWER" if or_val < 1.0 else "HIGHER"
            print(f"  -> DTI penalizes Black applicants {direction} than White applicants.")

    coefs.to_parquet("data/processed/interaction_model_results.parquet")
    print("\nSaved to data/processed/interaction_model_results.parquet")
    print("Done.")