import pandas as pd
import numpy as np
import requests
from rapidfuzz import process, fuzz
import warnings
warnings.filterwarnings("ignore")

CFPB_PATH  = "data/raw/cfpb_complaints.parquet"
FDIC_PATH  = "data/raw/fdic_financials.parquet"
OUT_MATCH  = "data/processed/cfpb_fdic_fullmatch.parquet"
OUT_PANEL  = "data/processed/complaint_panel.parquet"

# ── 1. Fetch ALL institution names from FDIC API (paginated) ──────────────────
print("Fetching institution names from FDIC API (paginated)...")
rows = []
limit  = 10000
offset = 0
while True:
    url = (
        "https://banks.data.fdic.gov/api/institutions"
        f"?fields=CERT,NAME,CITY,STALP"
        f"&limit={limit}&offset={offset}&sort_by=CERT&sort_order=ASC"
    )
    try:
        r = requests.get(url, timeout=60)
        payload = r.json()
        batch = payload.get("data", [])
        if not batch:
            break
        for rec in batch:
            d = rec.get("data", rec)
            rows.append(d)
        print(f"  Fetched {len(rows):,} institutions so far...")
        if len(batch) < limit:
            break
        offset += limit
    except Exception as e:
        print(f"  Error at offset {offset}: {e}")
        break

inst_df = pd.DataFrame(rows)
print(f"Total institution records fetched: {len(inst_df):,}")
print(f"Columns: {inst_df.columns.tolist()}")

# ── 2. Load our FDIC panel to know which CERTs we care about ─────────────────
fdic_fin = pd.read_parquet(FDIC_PATH)
our_certs = set(fdic_fin["CERT"].unique())
inst_df = inst_df[inst_df["CERT"].isin(our_certs)].copy()
print(f"After filtering to our {len(our_certs):,} CERTs: {len(inst_df):,} rows")
inst_df.to_parquet("data/processed/fdic_institution_names.parquet", index=False)

# ── 3. Load CFPB ──────────────────────────────────────────────────────────────
print("\nLoading CFPB...")
cfpb = pd.read_parquet(CFPB_PATH)
cfpb["date"]    = pd.to_datetime(cfpb["date_received"], errors="coerce")
cfpb["year"]    = cfpb["date"].dt.year
cfpb["quarter"] = cfpb["date"].dt.to_period("Q")
cfpb = cfpb[cfpb["year"] >= 2011].copy()
print(f"  CFPB rows: {len(cfpb):,}")

# ── 4. Clean names for matching ───────────────────────────────────────────────
def clean(s):
    return (str(s).lower()
            .replace("national bank", "")
            .replace("federal savings bank", "")
            .replace("state bank", "")
            .replace("savings bank", "")
            .replace("community bank", "")
            .replace(",", "").replace(".", "")
            .strip())

inst_df["name_clean"] = inst_df["NAME"].apply(clean)
fdic_names = inst_df["name_clean"].tolist()
fdic_certs  = inst_df["CERT"].tolist()

cfpb["company_clean"] = cfpb["company"].apply(clean)
unique_companies = cfpb["company_clean"].dropna().unique()
print(f"\nUnique CFPB companies: {len(unique_companies):,}")
print(f"FDIC institutions to match against: {len(fdic_names):,}")

# ── 5. Fuzzy match ────────────────────────────────────────────────────────────
print("\nRunning fuzzy match (5-15 minutes)...")
results = []
batch_size = 100
for i in range(0, len(unique_companies), batch_size):
    chunk = unique_companies[i:i+batch_size]
    for company in chunk:
        match = process.extractOne(
            company,
            fdic_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=88
        )
        if match:
            matched_name, score, idx = match
            results.append({
                "company_clean": company,
                "fdic_name":     matched_name,
                "CERT":          fdic_certs[idx],
                "match_score":   score
            })
    if (i // batch_size) % 20 == 0:
        print(f"  {i:,} / {len(unique_companies):,} processed, {len(results):,} matched so far...")

match_df = pd.DataFrame(results)
print(f"\nMatched {len(match_df):,} / {len(unique_companies):,} CFPB company names to FDIC CERTs")
if len(match_df) > 0:
    print("\nSample low-score matches (review for false positives):")
    print(match_df.nsmallest(15, "match_score")[["company_clean","fdic_name","CERT","match_score"]].to_string(index=False))

# ── 6. Join to complaints ─────────────────────────────────────────────────────
cfpb = cfpb.merge(match_df[["company_clean","CERT","match_score"]],
                   on="company_clean", how="left")
matched = cfpb["CERT"].notna()
print(f"\nComplaints matched to a CERT: {matched.sum():,} ({matched.mean()*100:.1f}%)")
cfpb_matched = cfpb[matched].copy()
cfpb_matched.to_parquet(OUT_MATCH, index=False)

# ── 7. Aggregate to CERT×quarter panel ───────────────────────────────────────
print("\nBuilding CERT×quarter panel...")
panel = (
    cfpb_matched
    .groupby(["CERT","quarter"])
    .agg(
        complaint_count = ("date","count"),
        timely_response = ("timely_response", lambda x: (x=="Yes").mean()),
    )
    .reset_index()
)
panel["quarter_str"] = panel["quarter"].astype(str)
panel = panel.sort_values(["CERT","quarter"])
panel["complaint_lag1"]   = panel.groupby("CERT")["complaint_count"].shift(1)
panel["complaint_lag4"]   = panel.groupby("CERT")["complaint_count"].shift(4)
panel["complaint_growth"] = (
    (panel["complaint_count"] - panel["complaint_lag1"])
    / (panel["complaint_lag1"] + 1)
)
panel.to_parquet(OUT_PANEL, index=False)
print(f"Panel: {panel.shape[0]:,} rows, {panel['CERT'].nunique():,} unique CERTs")
print(f"Quarter range: {panel['quarter_str'].min()} to {panel['quarter_str'].max()}")

# ── 8. CRITICAL DIAGNOSTIC ───────────────────────────────────────────────────
print("\n=== CRITICAL DIAGNOSTIC ===")
fdic_fin["REPDTE"]  = pd.to_datetime(fdic_fin["REPDTE"])
fdic_fin["quarter"] = fdic_fin["REPDTE"].dt.to_period("Q")
fdic_fin["texas_ratio"] = (
    fdic_fin["NCLNLS"] /
    (fdic_fin["EQ"] + fdic_fin["LNLSNTV"].replace(0, np.nan))
) * 100

distressed_bq = (
    fdic_fin[fdic_fin["texas_ratio"] > 100][["CERT","quarter"]]
    .drop_duplicates()
)
print(f"Total distressed bank-quarters in FDIC panel: {len(distressed_bq):,}")
print(f"Unique distressed CERTs: {distressed_bq['CERT'].nunique():,}")

panel["quarter_p"] = pd.PeriodIndex(panel["quarter_str"], freq="Q")
overlap = panel.merge(
    distressed_bq.rename(columns={"quarter":"quarter_p"}),
    on=["CERT","quarter_p"], how="left"
)
overlap["distressed"] = overlap["distressed"].fillna(0) if "distressed" in overlap.columns else 0

# re-merge properly
distressed_bq["distressed"] = 1
panel["quarter_p"] = pd.PeriodIndex(panel["quarter_str"], freq="Q")
overlap2 = panel.merge(
    distressed_bq.rename(columns={"quarter":"quarter_p"}),
    on=["CERT","quarter_p"], how="left"
)
overlap2["distressed"] = overlap2["distressed"].fillna(0)

n_dist   = int(overlap2["distressed"].sum())
n_certs  = overlap2[overlap2["distressed"]==1]["CERT"].nunique()
total_bq = len(distressed_bq)

print(f"\nDistressed bank-quarters with CFPB complaint data: {n_dist:,} / {total_bq:,}")
print(f"Unique distressed CERTs covered: {n_certs:,}")
print(f"Coverage: {n_dist/total_bq*100:.1f}%")
print(f"\nVerdict: {'VIABLE — enough signal to test the hypothesis' if n_dist > 300 else 'LOW COVERAGE — paper viability at risk, discuss before proceeding'}")
print("\nDone.")