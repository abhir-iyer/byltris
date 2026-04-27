import requests
import pandas as pd
import os
from io import StringIO

OUT_PATH = 'data/raw/hmda_truist.parquet'
os.makedirs('data/raw', exist_ok=True)

BASE = 'https://ffiec.cfpb.gov/v2/data-browser-api/view/csv'

INSTITUTIONS = {
    # Primary subject
    "Truist Bank":      "JJKC32MCHWDI71265Z06",
    # Mega-bank peers
    "Wells Fargo":      "KB1H1DSPRFMYMCUFXT09",
    "Bank of America":  "B4TYDEB6GKMZO031MB27",
    "JPMorgan Chase":   "7H6GLXDRUGQFU57RNE97",
    # Regional peers — most analytically relevant for the paper
    # Same Southeast/Mid-Atlantic footprint as Truist
    "Regions Bank":     "EQTWLK1G7ODGC2MGLV11",
    "PNC Bank":         "AD6GFRVSDT01YPT1CS68",
    "Fifth Third Bank": "QFROUN1UWUYU0DVIWD51",
    "Huntington Bank":  "2WHM8VNJH63UN14OL754",
    "Citizens Bank":    "DRMSV1Q0EKMEXLAU1P80",
    "U.S. Bank":        "6BYL5QZYBDK8S7L73M02",
}

YEARS = [2021, 2022, 2023]

def pull_one(name, lei, year):
    print(f'  {name} {year}...', end=' ', flush=True)
    try:
        resp = requests.get(BASE, params={'leis': lei, 'years': year}, timeout=120)
        if resp.status_code != 200:
            print(f'HTTP {resp.status_code}')
            return pd.DataFrame()
        if len(resp.text) < 100:
            print('empty')
            return pd.DataFrame()
        df = pd.read_csv(StringIO(resp.text), dtype=str, low_memory=False)
        df.columns = [c.lower().strip() for c in df.columns]
        df['institution'] = name
        df['activity_year'] = year
        print(f'{len(df):,} rows')
        return df
    except Exception as e:
        print(f'Error: {e}')
        return pd.DataFrame()

def lookup_leis():
    print('Searching HMDA reporter panel for Truist...')
    resp = requests.get(
        'https://ffiec.cfpb.gov/v2/data-browser-api/view/filers',
        params={'name': 'truist', 'years': 2023},
        timeout=30
    )
    data = resp.json()
    print(data)

def run_all():
    frames = []
    for name, lei in INSTITUTIONS.items():
        for year in YEARS:
            df = pull_one(name, lei, year)
            if not df.empty:
                frames.append(df)

    if not frames:
        print('No data pulled — running LEI lookup...')
        lookup_leis()
        return

    combined = pd.concat(frames, ignore_index=True)
    combined['action_taken'] = pd.to_numeric(combined.get('action_taken'), errors='coerce')
    combined['loan_amount'] = pd.to_numeric(combined.get('loan_amount'), errors='coerce')
    combined['income'] = pd.to_numeric(combined.get('income'), errors='coerce')
    combined['approved'] = (combined['action_taken'] == 1).astype(int)
    combined['denied'] = (combined['action_taken'] == 3).astype(int)

    combined.to_parquet(OUT_PATH, index=False)
    print(f'Total records: {len(combined):,}')
    print(f'Saved to {OUT_PATH}')
    print(combined['institution'].value_counts())

    if 'derived_race' in combined.columns:
        truist = combined[combined['institution'].isin(['Truist Bank','SunTrust Bank','BBT'])]
        print(truist.groupby('derived_race')['approved'].agg(['mean','count']).round(3))

if __name__ == '__main__':
    run_all()
