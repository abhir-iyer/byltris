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
ALL_FEATURES  = CREDIT_FEATURES + RACE_FEATURES


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


def run_logit(sub, label=""):
    clean = sub[ALL_FEATURES + ["approved"]].dropna()
    if len(clean) < 200:
        print(f"  {label}: insufficient data (n={len(clean)})")
        return None
    X = sm.add_constant(clean[ALL_FEATURES].astype(float))
    y = clean["approved"].astype(int)
    res = sm.Logit(y, X).fit(disp=0)
    black_OR = np.exp(res.params.get("is_black", np.nan))
    black_p  = res.pvalues.get("is_black", np.nan)
    black_lo = np.exp(res.params.get("is_black", np.nan) - 1.96 * res.bse.get("is_black", np.nan))
    black_hi = np.exp(res.params.get("is_black", np.nan) + 1.96 * res.bse.get("is_black", np.nan))
    print(f"  {label:<35} N={len(clean):>7,}  Black OR={black_OR:.4f}  [{black_lo:.4f}, {black_hi:.4f}]  p={black_p:.4f}")
    return {"label": label, "n": len(clean), "black_OR": black_OR,
            "black_CI_lo": black_lo, "black_CI_hi": black_hi, "black_p": black_p}


# ── 1. Baseline ───────────────────────────────────────────────────────────────
def run_baseline(sub):
    print("\n=== BASELINE ===")
    return run_logit(sub, "Full sample")


# ── 2. By year ────────────────────────────────────────────────────────────────
def run_by_year(sub):
    print("\n=== BY YEAR ===")
    results = []
    for year in sorted(sub["activity_year"].dropna().unique()):
        s = sub[sub["activity_year"] == year]
        r = run_logit(s, f"Year {int(year)}")
        if r:
            results.append(r)
    return results


# ── 3. By loan purpose ────────────────────────────────────────────────────────
def run_by_purpose(sub):
    print("\n=== BY LOAN PURPOSE ===")
    results = []
    purpose_map = {"1": "Purchase", "31": "Refinance", "32": "Cash-out refi"}
    # drop purpose dummies — they are constant within each purpose subset
    features_no_purpose = [f for f in ALL_FEATURES if not f.startswith("purpose_")]
    
    for code, name in purpose_map.items():
        s = sub[sub["loan_purpose"].astype(str) == code].copy()
        clean = s[features_no_purpose + ["approved"]].dropna()
        if len(clean) < 200:
            print(f"  Purpose: {name}: insufficient data (n={len(clean)})")
            continue
        X = sm.add_constant(clean[features_no_purpose].astype(float))
        y = clean["approved"].astype(int)
        try:
            res = sm.Logit(y, X).fit(disp=0)
            black_OR = np.exp(res.params.get("is_black", np.nan))
            black_p  = res.pvalues.get("is_black", np.nan)
            black_lo = np.exp(res.params.get("is_black", np.nan) - 1.96 * res.bse.get("is_black", np.nan))
            black_hi = np.exp(res.params.get("is_black", np.nan) + 1.96 * res.bse.get("is_black", np.nan))
            print(f"  Purpose: {name:<28} N={len(clean):>7,}  Black OR={black_OR:.4f}  [{black_lo:.4f}, {black_hi:.4f}]  p={black_p:.4f}")
            results.append({"label": f"Purpose: {name}", "n": len(clean),
                            "black_OR": black_OR, "black_CI_lo": black_lo,
                            "black_CI_hi": black_hi, "black_p": black_p})
        except Exception as e:
            print(f"  Purpose: {name}: failed — {e}")
    return results


# ── 4. By geography (top states) ─────────────────────────────────────────────
def run_by_state(sub):
    print("\n=== BY STATE (top 8 by volume) ===")
    results = []
    top_states = sub["state_code"].value_counts().head(8).index.tolist()
    for state in top_states:
        s = sub[sub["state_code"] == state]
        r = run_logit(s, f"State: {state}")
        if r:
            results.append(r)
    return results


# ── 5. By income tercile ──────────────────────────────────────────────────────
def run_by_income(sub):
    print("\n=== BY INCOME TERCILE ===")
    results = []
    sub2 = sub.copy()
    sub2["income_num"] = pd.to_numeric(sub2["income"], errors="coerce")
    sub2["income_tercile"] = pd.qcut(sub2["income_num"], q=3,
                                      labels=["Low income", "Mid income", "High income"],
                                      duplicates="drop")
    for tercile in ["Low income", "Mid income", "High income"]:
        s = sub2[sub2["income_tercile"] == tercile]
        r = run_logit(s, tercile)
        if r:
            results.append(r)
    return results


# ── 6. Placebo test ───────────────────────────────────────────────────────────
def run_placebo(sub, n_draws=500):
    print(f"\n=== PLACEBO TEST (n={n_draws} random draws) ===")
    np.random.seed(42)
    clean = sub[ALL_FEATURES + ["approved"]].dropna()
    placebo_ors = []

    for _ in range(n_draws):
        clean2 = clean.copy()
        clean2["is_black"] = np.random.permutation(clean2["is_black"].values)
        X = sm.add_constant(clean2[ALL_FEATURES].astype(float))
        y = clean2["approved"].astype(int)
        try:
            res = sm.Logit(y, X).fit(disp=0, method="bfgs")
            placebo_ors.append(np.exp(res.params.get("is_black", np.nan)))
        except:
            pass

    placebo_arr = np.array([x for x in placebo_ors if not np.isnan(x)])
    print(f"  Placebo Black OR distribution ({len(placebo_arr)} draws):")
    print(f"  Mean:   {placebo_arr.mean():.4f}")
    print(f"  Std:    {placebo_arr.std():.4f}")
    print(f"  95% CI: [{np.percentile(placebo_arr, 2.5):.4f}, {np.percentile(placebo_arr, 97.5):.4f}]")
    print(f"  Actual Black OR: 0.5414")
    print(f"  P-value (share of placebo ORs <= actual): {(placebo_arr <= 0.5414).mean():.4f}")
    return placebo_arr


# ── 7. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)
    sub = prep(df, "Truist Bank")
    print(f"Truist sample: {len(sub):,} rows")

    baseline  = run_baseline(sub)
    by_year   = run_by_year(sub)
    by_purpose = run_by_purpose(sub)
    by_state  = run_by_state(sub)
    by_income = run_by_income(sub)
    placebo   = run_placebo(sub, n_draws=500)

    # save all results
    all_results = [baseline] + by_year + by_purpose + by_state + by_income
    all_results = [r for r in all_results if r is not None]
    results_df  = pd.DataFrame(all_results)
    results_df.to_parquet("data/processed/fair_lending_robustness.parquet", index=False)
    np.save("data/processed/placebo_ors.npy", placebo)

    print(f"\n\nRobustness table summary:")
    print(results_df[["label","n","black_OR","black_CI_lo","black_CI_hi"]].to_string(index=False))
    print("\nDone.")