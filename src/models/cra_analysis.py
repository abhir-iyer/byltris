import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/hmda_extended.parquet"

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
    sub["activity_year"] = pd.to_numeric(sub["activity_year"], errors="coerce")

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

    sub["state_code"] = sub["state_code"].astype(str).str.strip().str.upper()
    return sub


def run_logit(sub, label=""):
    clean = sub[ALL_FEATURES + ["approved"]].dropna()
    if len(clean) < 300:
        print(f"  {label}: insufficient data (n={len(clean)})")
        return None
    X = sm.add_constant(clean[ALL_FEATURES].astype(float))
    y = clean["approved"].astype(int)
    res = sm.Logit(y, X).fit(disp=0)
    b_or = np.exp(res.params.get("is_black", np.nan))
    b_p  = res.pvalues.get("is_black", np.nan)
    b_lo = np.exp(res.params["is_black"] - 1.96*res.bse["is_black"])
    b_hi = np.exp(res.params["is_black"] + 1.96*res.bse["is_black"])
    print(f"  {label:<45} N={len(clean):>7,}  Black OR={b_or:.4f}  [{b_lo:.4f},{b_hi:.4f}]  p={b_p:.4f}")
    return {"label": label, "n": len(clean), "black_OR": b_or,
            "black_CI_lo": b_lo, "black_CI_hi": b_hi, "black_p": b_p}


# ── 1. Fetch Truist CRA assessment area states from FFIEC ────────────────────
def get_cra_states():
    """
    Truist's CRA assessment areas are publicly available from FFIEC.
    We approximate using their primary operating states based on
    branch concentration — publicly documented in their CRA Public File.
    Source: Truist Bank CRA Public File (ffiec.gov)
    """
    # Truist's primary CRA assessment area states
    # from their most recent CRA public evaluation (2022)
    TRUIST_CRA_STATES = {
        "NC", "VA", "GA", "FL", "TN", "MD", "SC",
        "WV", "KY", "IN", "OH", "NJ", "PA", "TX"
    }
    return TRUIST_CRA_STATES


# ── 2. CRA inside vs outside comparison ─────────────────────────────────────
def cra_inside_outside(sub, cra_states):
    print("=== CRA ASSESSMENT AREA: INSIDE vs OUTSIDE ===\n")

    sub = sub.copy()
    sub["in_cra"] = sub["state_code"].isin(cra_states).astype(int)

    inside  = sub[sub["in_cra"] == 1]
    outside = sub[sub["in_cra"] == 0]

    print(f"Applications inside CRA states:  {len(inside):,}")
    print(f"Applications outside CRA states: {len(outside):,}")
    print(f"CRA states covered: {sorted(cra_states)}\n")

    r_in  = run_logit(inside,  "Inside CRA assessment area")
    r_out = run_logit(outside, "Outside CRA assessment area")

    if r_in and r_out:
        diff = r_out["black_OR"] - r_in["black_OR"]
        print(f"\n  Gap difference (outside - inside): {diff:+.4f}")
        if diff < -0.03:
            print(f"  Disparity is LARGER outside CRA areas — CRA may be constraining discrimination")
        elif diff > 0.03:
            print(f"  Disparity is LARGER inside CRA areas — CRA oversight does not eliminate gap")
        else:
            print(f"  Gap is similar inside and outside CRA areas")

    return r_in, r_out


# ── 3. CRA interaction model ─────────────────────────────────────────────────
def cra_interaction_model(sub, cra_states):
    print("\n\n=== CRA INTERACTION MODEL ===\n")
    print("Testing: is_black × in_cra interaction\n")

    sub = sub.copy()
    sub["in_cra"] = sub["state_code"].isin(cra_states).astype(int)
    sub["black_x_cra"] = sub["is_black"] * sub["in_cra"]

    features = ALL_FEATURES + ["in_cra", "black_x_cra"]
    clean = sub[features + ["approved"]].dropna()

    if len(clean) < 500:
        print("Insufficient data.")
        return

    X = sm.add_constant(clean[features].astype(float))
    y = clean["approved"].astype(int)
    res = sm.Logit(y, X).fit(disp=0)

    black_or  = np.exp(res.params.get("is_black", np.nan))
    black_p   = res.pvalues.get("is_black", np.nan)
    inter_or  = np.exp(res.params.get("black_x_cra", np.nan))
    inter_p   = res.pvalues.get("black_x_cra", np.nan)
    inter_lo  = np.exp(res.params["black_x_cra"] - 1.96*res.bse["black_x_cra"])
    inter_hi  = np.exp(res.params["black_x_cra"] + 1.96*res.bse["black_x_cra"])

    print(f"N: {len(clean):,}")
    print(f"Black OR (outside CRA, reference):  {black_or:.4f}  p={black_p:.4f}")
    print(f"Black x CRA interaction OR:         {inter_or:.4f}  [{inter_lo:.4f},{inter_hi:.4f}]  p={inter_p:.4f}")

    combined_or = np.exp(res.params.get("is_black",0) + res.params.get("black_x_cra",0))
    print(f"Implied Black OR inside CRA:        {combined_or:.4f}")

    if inter_p < 0.05:
        direction = "smaller" if inter_or > 1 else "larger"
        print(f"\n  Significant at 5%: The Black-White gap is {direction} inside CRA assessment areas.")
    else:
        print(f"\n  Not significant: CRA designation does not measurably affect the gap.")

    print(f"Pseudo R2: {res.prsquared:.4f}")
    return res


# ── 4. Peer CRA comparison ───────────────────────────────────────────────────
def peer_cra_comparison(df, cra_states):
    print("\n\n=== PEER CRA COMPARISON ===\n")
    print("Black OR by institution: inside vs outside CRA states\n")

    results = []
    for inst in sorted(df["institution"].unique()):
        sub = prep(df, inst)
        sub["in_cra"] = sub["state_code"].isin(cra_states).astype(int)

        for label, mask in [("inside", sub["in_cra"]==1), ("outside", sub["in_cra"]==0)]:
            s = sub[mask]
            clean = s[ALL_FEATURES + ["approved"]].dropna()
            if len(clean) < 300:
                continue
            X = sm.add_constant(clean[ALL_FEATURES].astype(float))
            y = clean["approved"].astype(int)
            try:
                res = sm.Logit(y, X).fit(disp=0)
                b_or = np.exp(res.params.get("is_black", np.nan))
                b_p  = res.pvalues.get("is_black", np.nan)
                results.append({
                    "institution": inst, "cra": label,
                    "n": len(clean), "black_OR": b_or, "black_p": b_p
                })
            except:
                pass

    results_df = pd.DataFrame(results)
    pivot = results_df.pivot(index="institution", columns="cra", values="black_OR").round(4)
    pivot["gap (in - out)"] = (pivot.get("inside", np.nan) - pivot.get("outside", np.nan)).round(4)
    print(pivot.to_string())

    results_df.to_parquet("data/processed/cra_peer_comparison.parquet", index=False)
    print(f"\nSaved to data/processed/cra_peer_comparison.parquet")
    return results_df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
    df["loan_purpose"]  = df["loan_purpose"].astype(str)
    print(f"Shape: {df.shape}\n")

    cra_states = get_cra_states()
    truist = prep(df, "Truist Bank")

    r_in, r_out = cra_inside_outside(truist, cra_states)
    cra_interaction_model(truist, cra_states)
    peer_cra_comparison(df, cra_states)

    print("\nDone.")