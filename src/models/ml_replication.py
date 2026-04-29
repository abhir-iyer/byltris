"""
Step 2: ML Replication with SHAP Feature Importance
====================================================
Replicates the main fair lending result using three model families:
  1. Logistic regression (baseline — matches paper)
  2. Gradient boosted trees (GBT via sklearn)
  3. Random forest (RF via sklearn)

For each model:
  - ROC-AUC on held-out test set (stratified 80/20 split)
  - Feature importance via SHAP values (GBT only — exact, not sampled)

Key question: does is_black rank as a top predictor in black-box models
even after all financial controls are included?

If three completely different model families agree that race carries
independent predictive content, it is much harder to attribute the finding
to logistic regression specification.

Install if needed:
  pip install shap scikit-learn --break-system-packages
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import shap

DATA_PATH = "data/raw/hmda_truist.parquet"
OUT_PATH  = "data/processed/ml_replication.parquet"

CREDIT_FEATURES = [
    "log_income", "log_loan_amount", "dti_mid",
    "purpose_purchase", "purpose_refi", "purpose_cashout",
]
RACE_FEATURES = ["is_black", "is_hispanic", "is_asian"]
ALL_FEATURES  = CREDIT_FEATURES + RACE_FEATURES

RANDOM_STATE = 42


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

    return sub


# ══════════════════════════════════════════════════════════════════════════════
def fit_logit(X_train, y_train, X_test, y_test):
    X_sm = sm.add_constant(X_train, has_constant="add")
    model = sm.Logit(y_train, X_sm).fit(disp=0)
    X_test_sm = sm.add_constant(X_test, has_constant="add")
    y_pred = model.predict(X_test_sm)

    auc  = roc_auc_score(y_test, y_pred)
    aps  = average_precision_score(y_test, y_pred)

    coef    = model.params.get("is_black", np.nan)
    se      = model.bse.get("is_black", np.nan)
    black_or = np.exp(coef)
    ci_lo    = np.exp(coef - 1.96 * se)
    ci_hi    = np.exp(coef + 1.96 * se)
    pval     = model.pvalues.get("is_black", np.nan)

    return {
        "model":    "Logistic Regression",
        "auc":      auc,
        "aps":      aps,
        "black_or": black_or,
        "ci_lo":    ci_lo,
        "ci_hi":    ci_hi,
        "pval":     pval,
        "n_train":  len(X_train),
        "n_test":   len(X_test),
    }


def fit_gbt(X_train, y_train, X_test, y_test):
    """
    Gradient boosted trees.
    Hyperparameters are modest to avoid overfitting on a binary outcome.
    n_estimators=300, max_depth=4, learning_rate=0.05 is a standard
    conservative setting; subsample=0.8 reduces variance.
    """
    gbt = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
        n_iter_no_change=20,
        validation_fraction=0.1,
        tol=1e-4,
    )
    gbt.fit(X_train, y_train)
    y_pred = gbt.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred)
    aps = average_precision_score(y_test, y_pred)

    return gbt, {
        "model":    "Gradient Boosted Trees",
        "auc":      auc,
        "aps":      aps,
        "black_or": np.nan,   # ORs not defined for tree models
        "ci_lo":    np.nan,
        "ci_hi":    np.nan,
        "pval":     np.nan,
        "n_train":  len(X_train),
        "n_test":   len(X_test),
    }


def fit_rf(X_train, y_train, X_test, y_test):
    """
    Random forest.
    n_estimators=500, max_depth=10, min_samples_leaf=50 to avoid
    overfit on this large dataset.
    """
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=50,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred)
    aps = average_precision_score(y_test, y_pred)

    return rf, {
        "model":    "Random Forest",
        "auc":      auc,
        "aps":      aps,
        "black_or": np.nan,
        "ci_lo":    np.nan,
        "ci_hi":    np.nan,
        "pval":     np.nan,
        "n_train":  len(X_train),
        "n_test":   len(X_test),
    }


# ══════════════════════════════════════════════════════════════════════════════
def run_shap(gbt_model, X_train, X_test, feature_names):
    """
    SHAP values for GBT.
    Uses TreeExplainer (exact, not sampled).
    Returns mean absolute SHAP per feature (= mean importance).
    Also returns per-feature SHAP for is_black specifically.
    """
    print("  Computing SHAP values (this takes ~1-2 minutes)...")
    explainer = shap.TreeExplainer(gbt_model)

    # Use a sample of 10,000 test observations for speed
    sample = pd.DataFrame(X_test, columns=feature_names).sample(
        n=min(10000, len(X_test)), random_state=RANDOM_STATE
    )
    shap_values = explainer.shap_values(sample)

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)

    # is_black rank (1 = most important)
    black_rank = list(mean_abs_shap.index).index("is_black") + 1

    # Mean SHAP for is_black on Black applicants specifically
    black_mask = sample["is_black"] == 1
    black_shap_idx = list(feature_names).index("is_black")
    if black_mask.sum() > 0:
        black_avg_shap = shap_values[black_mask, black_shap_idx].mean()
    else:
        black_avg_shap = np.nan

    return mean_abs_shap, black_rank, black_avg_shap, shap_values, sample


# ══════════════════════════════════════════════════════════════════════════════
def shap_counterfactual(gbt_model, X_test_df, feature_names):
    """
    For GBT: set is_black=0 for all Black applicants and compare
    predicted approval probability.
    This mirrors the logit counterfactual but on a nonparametric model.
    """
    black_mask = X_test_df["is_black"] == 1
    black_df   = X_test_df[black_mask].copy()

    if len(black_df) == 0:
        return np.nan, np.nan

    X_actual  = black_df[feature_names].values
    X_counter = black_df[feature_names].copy()
    X_counter["is_black"] = 0
    X_counter = X_counter[feature_names].values

    p_actual  = gbt_model.predict_proba(X_actual)[:, 1]
    p_counter = gbt_model.predict_proba(X_counter)[:, 1]

    excess = p_counter - p_actual
    return excess.sum(), excess.mean()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)

    print("Loading and preparing data...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)

    sub   = prep(df)
    clean = sub[ALL_FEATURES + ["approved"]].dropna()
    X_all = clean[ALL_FEATURES].values
    y_all = clean["approved"].values
    print(f"Complete cases: {len(clean):,}  |  Approval rate: {y_all.mean():.3f}")

    # ── Stratified 80/20 split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
    )
    X_train_df = pd.DataFrame(X_train, columns=ALL_FEATURES)
    X_test_df  = pd.DataFrame(X_test,  columns=ALL_FEATURES)
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Model 1: Logistic regression ──────────────────────────────────────────
    print("\n[1/3] Logistic regression...")
    logit_results = fit_logit(
        pd.DataFrame(X_train, columns=ALL_FEATURES), y_train,
        pd.DataFrame(X_test,  columns=ALL_FEATURES), y_test
    )
    print(f"  AUC: {logit_results['auc']:.4f}  |  Black OR: {logit_results['black_or']:.4f}  "
          f"[{logit_results['ci_lo']:.4f}, {logit_results['ci_hi']:.4f}]  "
          f"p={logit_results['pval']:.4f}")

    # ── Model 2: Gradient boosted trees ──────────────────────────────────────
    print("\n[2/3] Gradient boosted trees...")
    gbt_model, gbt_results = fit_gbt(X_train, y_train, X_test, y_test)
    print(f"  AUC: {gbt_results['auc']:.4f}  |  n_estimators used: {gbt_model.n_estimators_}")

    # ── Model 3: Random forest ────────────────────────────────────────────────
    print("\n[3/3] Random forest...")
    rf_model, rf_results = fit_rf(X_train, y_train, X_test, y_test)
    print(f"  AUC: {rf_results['auc']:.4f}")

    # ── SHAP analysis on GBT ─────────────────────────────────────────────────
    print("\nSHAP analysis (GBT)...")
    mean_abs_shap, black_rank, black_avg_shap, shap_vals, shap_sample = run_shap(
        gbt_model, X_train, X_test, ALL_FEATURES
    )

    # ── GBT counterfactual ────────────────────────────────────────────────────
    gbt_excess_total, gbt_excess_avg = shap_counterfactual(gbt_model, X_test_df, ALL_FEATURES)

    # ── Print results ─────────────────────────────────────────────────────────
    W = 66
    print("\n" + "=" * W)
    print("ML REPLICATION RESULTS")
    print("Truist Bank  |  2021-2023  |  80/20 stratified split")
    print("=" * W)

    print(f"\n{'Model comparison (AUC, higher = better fit)':}")
    print(f"  {'Model':<30} {'AUC':>8}  {'Avg Precision':>14}  {'Black OR':>10}")
    print(f"  {'-'*64}")
    for r in [logit_results, gbt_results, rf_results]:
        or_str = f"{r['black_or']:.4f}" if not np.isnan(r['black_or']) else "  (n.a.)"
        print(f"  {r['model']:<30} {r['auc']:>8.4f}  {r['aps']:>14.4f}  {or_str:>10}")

    print(f"\n{'SHAP feature importance (GBT, mean |SHAP|)':}")
    print(f"  {'Rank':<6} {'Feature':<25} {'Mean |SHAP|':>12}")
    print(f"  {'-'*45}")
    for rank, (feat, val) in enumerate(mean_abs_shap.items(), 1):
        marker = " <-- RACE" if feat == "is_black" else (
                 " <-- RACE" if feat in ["is_hispanic", "is_asian"] else "")
        print(f"  {rank:<6} {feat:<25} {val:>12.6f}{marker}")

    print(f"\n{'is_black in GBT':}")
    print(f"  Rank among all features:   {black_rank} of {len(ALL_FEATURES)}")
    print(f"  Avg SHAP (Black appl.):    {black_avg_shap:.6f}  "
          f"({'negative = lower approval odds' if black_avg_shap < 0 else 'positive'})")

    print(f"\n{'GBT counterfactual (test set, Black applicants)':}")
    print(f"  Total excess denials:      {gbt_excess_total:.1f}  (fractional)")
    print(f"  Avg excess per applicant:  {gbt_excess_avg:.4f}  ({gbt_excess_avg*100:.1f} pp)")

    print(f"\n{'Consistency check':}")
    auc_range = max(logit_results['auc'], gbt_results['auc'], rf_results['auc']) - \
                min(logit_results['auc'], gbt_results['auc'], rf_results['auc'])
    print(f"  AUC range across models:   {auc_range:.4f}")
    if black_rank <= 4:
        print(f"  is_black ranks {black_rank} of {len(ALL_FEATURES)} in GBT SHAP — "
              f"race carries predictive content in nonparametric model")
    else:
        print(f"  is_black ranks {black_rank} of {len(ALL_FEATURES)} in GBT SHAP")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame([logit_results, gbt_results, rf_results])
    shap_df    = mean_abs_shap.reset_index()
    shap_df.columns = ["feature", "mean_abs_shap"]
    shap_df["rank"]          = range(1, len(shap_df) + 1)
    shap_df["is_race"]       = shap_df["feature"].isin(RACE_FEATURES)
    shap_df["black_avg_shap"] = np.where(shap_df["feature"] == "is_black", black_avg_shap, np.nan)

    results_df.to_parquet(OUT_PATH, index=False)
    shap_df.to_parquet("data/processed/shap_importance.parquet", index=False)
    print(f"\nSaved to {OUT_PATH}")
    print("Saved SHAP table to data/processed/shap_importance.parquet")
    print("\nDone.")