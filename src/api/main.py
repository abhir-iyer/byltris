"""
Byltris API — FastAPI backend
Serves pre-computed model outputs + live CERT lookup
Deploy on Railway
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
from functools import lru_cache

app = FastAPI(title="Byltris API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.environ.get("DATA_DIR", "data/processed")
RAW_DIR  = os.environ.get("RAW_DIR",  "data/raw")

# ── Data loaders (cached) ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_watchlist():
    path = os.path.join(DATA_DIR, "watchlist.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df

@lru_cache(maxsize=1)
def load_fair_lending():
    path = os.path.join(DATA_DIR, "fair_lending_inference.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)

@lru_cache(maxsize=1)
def load_did_panel():
    path = os.path.join(DATA_DIR, "did_panel.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)

@lru_cache(maxsize=1)
def load_fdic():
    path = os.path.join(RAW_DIR, "fdic_financials.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    df["texas_ratio"] = (
        df["NCLNLS"] / (df["EQ"] + df["LNLSNTV"].replace(0, np.nan))
    ) * 100
    return df

@lru_cache(maxsize=1)
def load_fdic_names():
    path = os.path.join(DATA_DIR, "fdic_institution_names.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)

def safe_val(v):
    """Convert numpy types to Python natives for JSON serialisation."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v

# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "project": "Byltris", "version": "1.0.0"}

@app.get("/api/stats")
def get_stats():
    """Top-level headline numbers shown on the landing page."""
    return {
        "banks_monitored": 9820,
        "bank_quarters":   550404,
        "early_warning_auc": 0.804,
        "precision_at_50_lift": 100,
        "cfpb_complaints": 677037,
        "merger_complaint_increase_pct": 11.8,
        "merger_p_value": 0.014,
        "truist_black_or": 0.541,
        "truist_black_or_ci_lo": 0.523,
        "truist_black_or_ci_hi": 0.560,
        "excess_denials_per_year": 1130,
        "annual_wealth_foregone_lo_m": 130,
        "annual_wealth_foregone_hi_m": 289,
        "hmda_applications": 4490000,
        "peer_institutions": 10,
    }

@app.get("/api/watchlist")
def get_watchlist(limit: int = 50):
    """Return top N distressed bank-quarters by predicted probability."""
    df = load_watchlist()
    if df.empty:
        raise HTTPException(status_code=503, detail="Watchlist data not available")
    cols = [c for c in ["CERT", "REPDTE", "distress_prob", "texas_ratio",
                         "roa", "tier1_leverage", "STNAME", "CITY"] if c in df.columns]
    top = df.sort_values("distress_prob", ascending=False).head(limit)[cols]
    records = []
    for _, row in top.iterrows():
        records.append({k: safe_val(v) for k, v in row.items()})
    return {"count": len(records), "data": records}

@app.get("/api/watchlist/{cert}")
def get_bank_watchlist(cert: int):
    """Return distress history for a specific CERT."""
    df = load_watchlist()
    if df.empty:
        raise HTTPException(status_code=503, detail="Watchlist data not available")
    bank = df[df["CERT"] == cert]
    if bank.empty:
        raise HTTPException(status_code=404, detail=f"CERT {cert} not found")
    cols = [c for c in ["CERT", "REPDTE", "distress_prob", "texas_ratio",
                         "roa", "tier1_leverage"] if c in bank.columns]
    records = [{k: safe_val(v) for k, v in row.items()} for _, row in bank[cols].iterrows()]
    return {"cert": cert, "data": records}

@app.get("/api/fairlending")
def get_fair_lending():
    """Return peer fair lending comparison table."""
    df = load_fair_lending()
    if df.empty:
        raise HTTPException(status_code=503, detail="Fair lending data not available")
    records = [{k: safe_val(v) for k, v in row.items()} for _, row in df.iterrows()]
    return {"count": len(records), "data": records}

@app.get("/api/complaints")
def get_complaints():
    """Return DiD panel summary and pre/post complaint volumes."""
    df = load_did_panel()
    if df.empty:
        raise HTTPException(status_code=503, detail="Complaints data not available")

    summary = (
        df.groupby(["institution", "post"])["complaints"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    records = [{k: safe_val(v) for k, v in row.items()} for _, row in summary.iterrows()]

    return {
        "did_estimate": 0.111,
        "did_se": 0.046,
        "did_p": 0.014,
        "did_ci_lo": 0.022,
        "did_ci_hi": 0.201,
        "implied_pct_increase": 11.8,
        "data": records,
    }

@app.get("/api/bank/{cert}")
def get_bank(cert: int):
    """
    Live CERT lookup. Returns the bank's full financial history
    from FDIC call reports including Texas Ratio trajectory.
    """
    fdic = load_fdic()
    if fdic.empty:
        raise HTTPException(status_code=503, detail="FDIC data not available")

    bank = fdic[fdic["CERT"] == cert].sort_values("REPDTE")
    if bank.empty:
        raise HTTPException(status_code=404, detail=f"CERT {cert} not in dataset")

    # institution name lookup
    names = load_fdic_names()
    name = "Unknown"
    city = ""
    state = ""
    if not names.empty and "CERT" in names.columns:
        match = names[names["CERT"] == cert]
        if not match.empty:
            name  = match.iloc[0].get("NAME", "Unknown")
            city  = match.iloc[0].get("CITY", "")
            state = match.iloc[0].get("STALP", "")

    latest = bank.iloc[-1]
    history = []
    for _, row in bank.iterrows():
        history.append({
            "date":          row["REPDTE"].strftime("%Y-%m-%d"),
            "texas_ratio":   safe_val(row.get("texas_ratio")),
            "roa":           safe_val(row.get("NETINC") / row.get("ASSET") * 100
                                      if row.get("ASSET") else None),
            "assets_m":      safe_val(row.get("ASSET") / 1000
                                      if row.get("ASSET") else None),
            "tier1_leverage": safe_val(row.get("RBCT1J") / row.get("ASSET") * 100
                                       if row.get("ASSET") else None),
        })

    return {
        "cert":         cert,
        "name":         name,
        "city":         city,
        "state":        state,
        "latest_date":  latest["REPDTE"].strftime("%Y-%m-%d"),
        "texas_ratio":  safe_val(latest.get("texas_ratio")),
        "assets_m":     safe_val(latest.get("ASSET") / 1000 if latest.get("ASSET") else None),
        "history":      history,
    }
