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

DENIAL_REASON_MAP = {
    "1": "Debt-to-income ratio",
    "2": "Employment history",
    "3": "Credit history",
    "4": "Collateral",
    "5": "Insufficient cash",
    "6": "Unverifiable information",
    "7": "Credit application incomplete",
    "8": "Mortgage insurance denied",
    "9": "Other",
    "10": "Other"
}


def prep(df, institution="Truist Bank"):
    sub = df[df["institution"] == institution].copy()
    sub["log_income"]      = np.log1p(pd.to_numeric(sub["income"], errors="coerce").clip(lower=0))
    sub["log_loan_amount"] = np.log1p(pd.to_numeric(sub["loan_amount"], errors="coerce").clip(lower=0))
    sub["action_taken"]    = pd.to_numeric(sub["action_taken"], errors="coerce")
    sub["approved"] = (sub["action_taken"] == 1).astype(int)
    sub["denied"]   = (sub["action_taken"] == 3).astype(int)

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


# ── 1. Denial reason frequency by race ───────────────────────────────────────
def denial_reason_by_race(sub):
    print("=== DENIAL REASON FREQUENCY BY RACE ===\n")

    denied = sub[sub["denied"] == 1].copy()

    # HMDA has up to 4 denial reasons
    reason_cols = [c for c in denied.columns if "denial_reason" in c]
    print(f"Denial reason columns found: {reason_cols}")

    if not reason_cols:
        print("No denial reason columns found in dataset.")
        return None

    # melt all reason columns into one
    id_cols = ["is_black", "is_white", "is_hispanic", "is_asian", "approved", "denied"]
    id_cols = [c for c in id_cols if c in denied.columns]

    denied["is_white"] = (denied["derived_race"] == "White").astype(int)

    long = pd.melt(
        denied,
        id_vars=["is_black", "is_white"],
        value_vars=reason_cols,
        var_name="reason_num",
        value_name="reason_code"
    )
    long = long[long["reason_code"].notna() &
                (long["reason_code"].astype(str).str.strip() != "") &
                (long["reason_code"].astype(str).str.strip() != "nan")].copy()
    long["reason_code"] = long["reason_code"].astype(str).str.strip()
    long["reason_label"] = long["reason_code"].map(DENIAL_REASON_MAP).fillna("Other")

    # share of each denial reason by race
    black_denials = long[long["is_black"] == 1]
    white_denials = long[long["is_white"] == 1]

    black_shares = (black_denials["reason_label"].value_counts(normalize=True) * 100).rename("Black %")
    white_shares = (white_denials["reason_label"].value_counts(normalize=True) * 100).rename("White %")

    comparison = pd.concat([black_shares, white_shares], axis=1).fillna(0)
    comparison["gap (Black - White)"] = comparison["Black %"] - comparison["White %"]
    comparison = comparison.sort_values("gap (Black - White)", ascending=False)

    print(f"\nDenied Black applicants: {len(black_denials):,}")
    print(f"Denied White applicants: {len(white_denials):,}")
    print(f"\nDenial reason share comparison:")
    print(comparison.round(1).to_string())

    comparison.to_parquet("data/processed/denial_reasons_by_race.parquet")
    print(f"\nSaved to data/processed/denial_reasons_by_race.parquet")
    return comparison


# ── 2. Conditional denial reason model ───────────────────────────────────────
def conditional_denial_model(sub):
    """
    Among denied applicants with similar credit profiles,
    are Black applicants more likely to receive 'credit history'
    as a denial reason than White applicants?
    """
    print("\n\n=== CONDITIONAL DENIAL REASON MODEL ===\n")

    denied = sub[sub["denied"] == 1].copy()
    reason_cols = [c for c in denied.columns if "denial_reason" in c]
    if not reason_cols:
        print("No denial reason columns.")
        return

    denied["is_white"] = (denied["derived_race"] == "White").astype(int)
    denied["credit_history_denial"] = denied[reason_cols].apply(
        lambda row: int(any(str(v).strip() == "3" for v in row)), axis=1
    )
    denied["dti_denial"] = denied[reason_cols].apply(
        lambda row: int(any(str(v).strip() == "1" for v in row)), axis=1
    )
    denied["collateral_denial"] = denied[reason_cols].apply(
        lambda row: int(any(str(v).strip() == "4" for v in row)), axis=1
    )

    features = CREDIT_FEATURES + RACE_FEATURES
    for outcome, label in [
        ("credit_history_denial", "Credit history denial"),
        ("dti_denial", "DTI denial"),
        ("collateral_denial", "Collateral denial"),
    ]:
        clean = denied[features + [outcome]].dropna()
        if len(clean) < 200 or clean[outcome].sum() < 50:
            print(f"{label}: insufficient positive cases")
            continue
        X = sm.add_constant(clean[features].astype(float))
        y = clean[outcome].astype(int)
        try:
            res = sm.Logit(y, X).fit(disp=0)
            black_OR = np.exp(res.params.get("is_black", np.nan))
            black_p  = res.pvalues.get("is_black", np.nan)
            black_lo = np.exp(res.params["is_black"] - 1.96 * res.bse["is_black"])
            black_hi = np.exp(res.params["is_black"] + 1.96 * res.bse["is_black"])
            print(f"{label:<30}  N={len(clean):>6,}  "
                  f"Black OR={black_OR:.4f}  [{black_lo:.4f}, {black_hi:.4f}]  p={black_p:.4f}")
        except Exception as e:
            print(f"{label}: model failed — {e}")


# ── 3. Application vs origination pipeline ───────────────────────────────────
def application_vs_origination(df, institution="Truist Bank"):
    """
    Test whether disparity differs between application stage
    and origination stage (action_taken codes 1-8).
    """
    print("\n\n=== APPLICATION vs ORIGINATION PIPELINE ===\n")

    sub = df[df["institution"] == institution].copy()
    sub["action_taken"] = pd.to_numeric(sub["action_taken"], errors="coerce")
    sub["log_income"]   = np.log1p(pd.to_numeric(sub["income"], errors="coerce").clip(lower=0))
    sub["log_loan_amount"] = np.log1p(pd.to_numeric(sub["loan_amount"], errors="coerce").clip(lower=0))

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

    features = CREDIT_FEATURES + RACE_FEATURES

    # approved = originated (1) vs denied (3) — standard
    for label, approved_codes, denied_codes in [
        ("Originated vs Denied",   [1], [3]),
        ("Originated vs All Other", [1], [2, 3, 4, 5]),
    ]:
        keep = sub[sub["action_taken"].isin(approved_codes + denied_codes)].copy()
        keep["approved"] = keep["action_taken"].isin(approved_codes).astype(int)
        clean = keep[features + ["approved"]].dropna()
        if len(clean) < 200:
            continue
        X = sm.add_constant(clean[features].astype(float))
        y = clean["approved"].astype(int)
        try:
            res = sm.Logit(y, X).fit(disp=0)
            black_OR = np.exp(res.params.get("is_black", np.nan))
            black_p  = res.pvalues.get("is_black", np.nan)
            black_lo = np.exp(res.params["is_black"] - 1.96 * res.bse["is_black"])
            black_hi = np.exp(res.params["is_black"] + 1.96 * res.bse["is_black"])
            print(f"{label:<30}  N={len(clean):>7,}  "
                  f"Black OR={black_OR:.4f}  [{black_lo:.4f}, {black_hi:.4f}]  p={black_p:.4f}")
        except Exception as e:
            print(f"{label}: failed — {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)

    sub = prep(df, "Truist Bank")
    print(f"Truist sample: {len(sub):,} rows\n")

    comparison = denial_reason_by_race(sub)
    conditional_denial_model(sub)
    application_vs_origination(df, "Truist Bank")

    print("\nDone.")