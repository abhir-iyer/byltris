import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/fdic_financials.parquet"
OUT_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load & feature engineer ──────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["CERT", "REPDTE"])

    # core CAMELS ratios
    df["texas_ratio"]    = df["NCLNLS"] / (df["EQ"] + df["LNLSNTV"]).replace(0, np.nan) * 100
    df["nim"]            = (df["INTINC"] - df["EINTEXP"]) / df["ASSET"].replace(0, np.nan)
    df["cre_ratio"]      = df["LNRE"] / df["ASSET"].replace(0, np.nan)
    df["tier1_leverage"] = df["RBCT1J"] / df["ASSET"].replace(0, np.nan)
    df["ltd_ratio"]      = df["LNLSNET"] / df["DEP"].replace(0, np.nan)
    df["roa"]            = df["NETINC"] / df["ASSET"].replace(0, np.nan)

    # quarter-over-quarter changes per bank
    for col in ["texas_ratio", "nim", "DEP", "ASSET"]:
        df[f"{col}_qoq"] = df.groupby("CERT")[col].pct_change()

    # log asset size (captures institution scale)
    df["log_asset"] = np.log1p(df["ASSET"])

    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distress label: Texas Ratio crosses 100 within the next 4 quarters.
    Proxy for FDIC enforcement action / failure.
    """
    df = df.sort_values(["CERT", "REPDTE"])
    df["distressed_now"] = (df["texas_ratio"] > 100).astype(int)

    # forward-looking: will this bank be distressed in the next 4 quarters?
    df["label"] = (
        df.groupby("CERT")["distressed_now"]
        .shift(-4)
        .fillna(0)
        .astype(int)
    )
    return df


# ── 2. Train / test split with temporal embargo ──────────────────────────────

FEATURES = [
    "texas_ratio", "nim", "cre_ratio", "tier1_leverage",
    "ltd_ratio", "roa", "log_asset",
    "texas_ratio_qoq", "nim_qoq", "DEP_qoq", "ASSET_qoq"
]

TRAIN_END  = pd.Timestamp("2019-12-31")
EMBARGO_END = pd.Timestamp("2020-06-30")   # 2-quarter gap — no leakage
TEST_START = pd.Timestamp("2020-09-30")


def split(df: pd.DataFrame):
    train = df[df["REPDTE"] <= TRAIN_END].copy()
    test  = df[df["REPDTE"] >= TEST_START].copy()
    print(f"Train: {len(train):,} rows | positive rate: {train['label'].mean():.3%}")
    print(f"Test:  {len(test):,} rows  | positive rate: {test['label'].mean():.3%}")
    return train, test


def prep_xy(df, features=FEATURES):
    sub = df[features + ["label"]].dropna()
    # clip extreme outliers
    sub[features] = sub[features].clip(-10, 10)
    X = sub[features]
    y = sub["label"]
    return X, y


# ── 3. Train models ──────────────────────────────────────────────────────────

def train_logit(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def train_xgb(X_train, y_train):
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Platt scaling for calibrated probabilities
    calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
    calibrated.fit(X_train, y_train)
    return calibrated


# ── 4. Evaluate ──────────────────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test, scaler=None):
    X = scaler.transform(X_test) if scaler else X_test
    proba = model.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    ap    = average_precision_score(y_test, proba)

    # precision at top 50 flagged banks
    top50_idx = np.argsort(proba)[-50:]
    p_at_50   = y_test.iloc[top50_idx].mean()

    print(f"\n{name}")
    print(f"  AUC:            {auc:.4f}")
    print(f"  Avg Precision:  {ap:.4f}")
    print(f"  Precision@50:   {p_at_50:.4f}")
    return proba


# ── 5. SHAP explainability ───────────────────────────────────────────────────

def run_shap(model, X_test, feature_names):
    print("\nComputing SHAP values...")
    # extract base XGB estimator from Platt calibration wrapper
    base = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base)
    shap_vals = explainer.shap_values(X_test)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nTop feature importances (SHAP):")
    print(importance.to_string(index=False))
    return shap_vals, importance


# ── 6. Watchlist ─────────────────────────────────────────────────────────────

def generate_watchlist(df, model, features=FEATURES, top_n=25):
    latest = df[df["REPDTE"] == df["REPDTE"].max()].copy()
    sub = latest[features].dropna()
    sub_clipped = sub.clip(-10, 10)

    proba = model.predict_proba(sub_clipped)[:, 1]
    latest_sub = latest.loc[sub.index].copy()
    latest_sub["distress_prob"] = proba

    watchlist = (
        latest_sub[["CERT", "STNAME", "CITY", "distress_prob", "texas_ratio", "tier1_leverage", "cre_ratio"]]
        .sort_values("distress_prob", ascending=False)
        .head(top_n)
    )

    print(f"\nTop {top_n} flagged banks (latest quarter):")
    print(watchlist.to_string(index=False))

    out_path = f"{OUT_DIR}/watchlist.parquet"
    watchlist.to_parquet(out_path, index=False)
    print(f"\nWatchlist saved to {out_path}")
    return watchlist


# ── 7. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading FDIC data...")
    df_raw = pd.read_parquet(DATA_PATH)

    print("Building features...")
    df = build_features(df_raw)
    df = build_labels(df)

    print(f"\nDataset: {len(df):,} rows | label rate: {df['label'].mean():.3%}")

    train, test = split(df)
    X_train, y_train = prep_xy(train)
    X_test,  y_test  = prep_xy(test)

    print("\nTraining logistic regression baseline...")
    logit, scaler = train_logit(X_train, y_train)
    logit_proba = evaluate("Logistic Regression", logit, X_test, y_test, scaler)

    print("\nTraining XGBoost + Platt calibration...")
    xgb_model = train_xgb(X_train, y_train)
    xgb_proba = evaluate("XGBoost (calibrated)", xgb_model, X_test, y_test)

    # SHAP on XGBoost
    shap_vals, shap_importance = run_shap(xgb_model, X_test.clip(-10, 10), FEATURES)

    # save model
    model_path = f"{MODEL_DIR}/early_warning_xgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": xgb_model, "features": FEATURES}, f)
    print(f"\nModel saved to {model_path}")

    # generate watchlist
    watchlist = generate_watchlist(df, xgb_model)

    print("\nDone.")