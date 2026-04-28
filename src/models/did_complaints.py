import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/cfpb_complaints.parquet"

# Treatment: Truist-family entities
# Control: peer banks in our dataset
# Post: Q1 2019 onwards (merger announcement Feb 7, 2019)

def normalize(name):
    n = str(name).lower()
    if any(x in n for x in ["truist", "suntrust", "bb&t", "bbt"]): return "Truist"
    if "bank of america" in n: return "Bank of America"
    if "wells fargo" in n: return "Wells Fargo"
    if "jpmorgan" in n or "chase" in n: return "JPMorgan Chase"
    if "citibank" in n or "citi" in n: return "Citibank"
    return None


def build_panel():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["institution"] = df["company"].apply(normalize)
    df = df[df["institution"].notna()].copy()
    df["quarter"] = df["date"].dt.to_period("Q")
    df["year"] = df["date"].dt.year

    # restrict to 2014-2025 for sufficient pre-period
    df = df[df["year"].between(2014, 2025)]

    # quarterly complaint counts per institution
    panel = (
        df.groupby(["institution", "quarter"])
        .size()
        .reset_index(name="complaints")
    )
    panel["quarter_str"] = panel["quarter"].astype(str)
    panel["quarter_dt"]  = panel["quarter"].apply(lambda q: q.start_time)

    # treatment and post indicators
    panel["treat"] = (panel["institution"] == "Truist").astype(int)
    panel["post"]  = (panel["quarter_dt"] >= pd.Timestamp("2019-02-01")).astype(int)
    panel["log_complaints"] = np.log1p(panel["complaints"])

    return panel


def run_did(panel):
    # pre-period: 2014-2018 for parallel trends check
    pre = panel[panel["quarter_dt"] < pd.Timestamp("2019-01-01")]
    pre_growth = pre.groupby(["institution"])["complaints"].apply(
        lambda x: x.pct_change().mean() * 100
    ).reset_index(name="avg_qoq_pct")
    print("Pre-period average quarterly growth rates:")
    print(pre_growth.to_string(index=False))

    # DiD: two-way fixed effects (institution + quarter)
    print("\nRunning two-way fixed effects DiD...")
    result = smf.ols(
        "log_complaints ~ treat:post + C(institution) + C(quarter_str)",
        data=panel
    ).fit(cov_type="HC3")  # heteroskedasticity-robust SEs

    did_coef = result.params.get("treat:post", np.nan)
    did_se   = result.bse.get("treat:post", np.nan)
    did_p    = result.pvalues.get("treat:post", np.nan)
    did_ci   = result.conf_int().loc["treat:post"] if "treat:post" in result.params.index else [np.nan, np.nan]

    print(f"\nDiD estimate (treat x post):")
    print(f"  Coefficient (log scale): {did_coef:.4f}")
    print(f"  Implied % change:        {(np.exp(did_coef)-1)*100:.1f}%")
    print(f"  SE (HC3):                {did_se:.4f}")
    print(f"  p-value:                 {did_p:.4f}")
    print(f"  95% CI:                  [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
    print(f"\n  Interpretation: The merger is associated with a "
          f"{(np.exp(did_coef)-1)*100:.1f}% change in log complaint volume "
          f"at Truist relative to control banks post-announcement.")

    # product breakdown for the treatment period
    return result, did_coef, did_se, did_p


def product_breakdown():
    df = pd.read_parquet(DATA_PATH)
    df["institution"] = df["company"].apply(normalize)
    df["date"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["post"] = df["year"] >= 2019

    truist = df[df["institution"] == "Truist"].copy()
    pre_vol  = truist[~truist["post"]].groupby("product").size()
    post_vol = truist[truist["post"]].groupby("product").size()
    breakdown = pd.DataFrame({"pre": pre_vol, "post": post_vol}).fillna(0)
    breakdown["change"] = breakdown["post"] - breakdown["pre"]
    breakdown["pct_of_change"] = breakdown["change"] / breakdown["change"].sum() * 100
    breakdown = breakdown.sort_values("change", ascending=False)

    print("\nProduct-level complaint breakdown (Truist, pre vs post 2019):")
    print(breakdown.round(1).to_string())
    return breakdown


if __name__ == "__main__":
    print("Building panel...")
    panel = build_panel()
    print(f"Panel shape: {panel.shape}")
    print(f"\nQuarterly observations per institution:")
    print(panel.groupby("institution")["complaints"].describe().round(1))

    result, coef, se, p = run_did(panel)
    breakdown = product_breakdown()

    panel.to_parquet("data/processed/did_panel.parquet", index=False)
    print("\nDone.")