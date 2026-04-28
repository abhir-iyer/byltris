import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import shap
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_truist.parquet"
OUT_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load & clean ──────────────────────────────────────────────────────────

def load_data():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded: {df.shape}")

    # keep only originated (1) and denied (3)
    df = df[df["action_taken"].isin([1, 3])].copy()
    df["approved"] = (df["action_taken"] == 1).astype(int)

    # numeric conversions
    for col in ["loan_amount", "income", "combined_loan_to_value_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # DTI is a string range in HMDA — convert to midpoint
    if "debt_to_income_ratio" in df.columns:
        def dti_midpoint(val):
            try:
                if "-" in str(val):
                    lo, hi = str(val).replace("%","").split("-")
                    return (float(lo) + float(hi)) / 2
                return float(str(val).replace("%","").replace("<","").replace(">","").strip())
            except:
                return np.nan
        df["dti_mid"] = df["debt_to_income_ratio"].apply(dti_midpoint)
    else:
        df["dti_mid"] = np.nan

    # log transforms for skewed financial variables
    df["log_income"]      = np.log1p(df["income"].clip(lower=0))
    df["log_loan_amount"] = np.log1p(df["loan_amount"].clip(lower=0))

    # race binary flags
    df["is_black"] = (df["derived_race"] == "Black or African American").astype(int)
    df["is_white"] = (df["derived_race"] == "White").astype(int)
    df["is_asian"] = (df["derived_race"] == "Asian").astype(int)
    df["is_hispanic"] = (df["derived_ethnicity"].str.contains("Hispanic", na=False)).astype(int)

    # loan purpose dummies
    if "loan_purpose" in df.columns:
        df["loan_purpose"] = df["loan_purpose"].astype(str)
        df["purpose_purchase"]   = (df["loan_purpose"] == "1").astype(int)
        df["purpose_refi"]       = (df["loan_purpose"] == "31").astype(int)
        df["purpose_cashout"]    = (df["loan_purpose"] == "32").astype(int)

    print(f"After filtering: {len(df):,} rows")
    print(f"Approval rate: {df['approved'].mean():.3%}")
    return df


# ── 2. Two-stage fair lending model ─────────────────────────────────────────

CREDIT_FEATURES = [
    "log_income", "log_loan_amount",
    "dti_mid", "purpose_purchase", "purpose_refi", "purpose_cashout"
]

RACE_FEATURES = ["is_black", "is_hispanic", "is_asian"]


def run_stage1(df, institution="Truist Bank"):
    """Stage 1: predict approval using credit-legitimate features only."""
    sub = df[df["institution"] == institution].copy()
    sub = sub[CREDIT_FEATURES + ["approved"]].dropna()

    X = sub[CREDIT_FEATURES]
    y = sub["approved"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    print(f"\nStage 1 [{institution}] — credit factors only")
    print(f"  AUC: {auc:.4f}")
    print(f"  N:   {len(sub):,}")

    coef = pd.DataFrame({
        "feature": CREDIT_FEATURES,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0])
    }).sort_values("odds_ratio", ascending=False)
    print(f"\n  Credit factor coefficients:")
    print(coef.to_string(index=False))

    return model, scaler, sub.index


def run_stage2(df, institution="Truist Bank"):
    """
    Stage 2: add race to the model.
    If race coefficient is significant after credit controls,
    it suggests unexplained disparity.
    """
    all_features = CREDIT_FEATURES + RACE_FEATURES
    sub = df[df["institution"] == institution].copy()
    sub = sub[all_features + ["approved"]].dropna()

    X = sub[all_features]
    y = sub["approved"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    print(f"\nStage 2 [{institution}] — credit factors + race")
    coef = pd.DataFrame({
        "feature": all_features,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0])
    }).sort_values("odds_ratio", ascending=False)
    print(coef.to_string(index=False))

    # race-specific odds ratios (key finding)
    print(f"\n  Race odds ratios after credit-factor adjustment:")
    for race in RACE_FEATURES:
        row = coef[coef["feature"] == race].iloc[0]
        direction = "LOWER" if row["odds_ratio"] < 1 else "HIGHER"
        print(f"  {race}: OR={row['odds_ratio']:.3f} ({direction} odds vs reference group)")

    return model, scaler


def peer_disparity_table(df):
    """Run stage 2 for all institutions and compare residual gaps."""
    results = []
    all_features = CREDIT_FEATURES + RACE_FEATURES

    for inst in df["institution"].unique():
        sub = df[df["institution"] == inst].copy()
        sub = sub[all_features + ["approved"]].dropna()

        if len(sub) < 500:
            continue

        X = sub[all_features]
        y = sub["approved"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        coef_dict = dict(zip(all_features, model.coef_[0]))
        results.append({
            "institution": inst,
            "n": len(sub),
            "black_coef": coef_dict.get("is_black", np.nan),
            "black_OR": np.exp(coef_dict.get("is_black", np.nan)),
            "hispanic_coef": coef_dict.get("is_hispanic", np.nan),
            "hispanic_OR": np.exp(coef_dict.get("is_hispanic", np.nan)),
        })

    results_df = pd.DataFrame(results).sort_values("black_OR")
    print("\n\nPeer comparison — Black applicant odds ratio after credit-factor adjustment:")
    print("(OR < 1.0 means lower approval odds for Black applicants vs White)")
    print(results_df[["institution", "n", "black_OR", "hispanic_OR"]].to_string(index=False))

    out_path = f"{OUT_DIR}/fair_lending_peer_disparity.parquet"
    results_df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return results_df


# ── 3. SHAP on stage 2 model ─────────────────────────────────────────────────

def run_shap_fairlending(model, scaler, df, institution="Truist Bank"):
    all_features = CREDIT_FEATURES + RACE_FEATURES
    sub = df[df["institution"] == institution][all_features + ["approved"]].dropna()
    X_scaled = scaler.transform(sub[all_features])

    # sample for speed
    sample = X_scaled[:2000]
    explainer = shap.LinearExplainer(model, sample, feature_perturbation="correlation_dependent")
    shap_vals = explainer.shap_values(sample)

    importance = pd.DataFrame({
        "feature": all_features,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\nSHAP feature importance [{institution} — Stage 2]:")
    print(importance.to_string(index=False))
    return shap_vals, importance


# ── 4. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    # Stage 1 — credit factors only
    s1_model, s1_scaler, _ = run_stage1(df, "Truist Bank")

    # Stage 2 — add race
    s2_model, s2_scaler = run_stage2(df, "Truist Bank")

    # Peer comparison
    peer_table = peer_disparity_table(df)

    # SHAP
    shap_vals, shap_imp = run_shap_fairlending(s2_model, s2_scaler, df, "Truist Bank")

    # save
    model_path = f"{MODEL_DIR}/fair_lending_logit.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "stage1_model": s1_model, "stage1_scaler": s1_scaler,
            "stage2_model": s2_model, "stage2_scaler": s2_scaler,
            "credit_features": CREDIT_FEATURES,
            "race_features": RACE_FEATURES
        }, f)
    print(f"\nModel saved to {model_path}")
    print("\nDone.")