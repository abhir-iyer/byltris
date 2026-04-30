"""
Fannie Mae Column Diagnostic
=============================
Reads first 5 rows and prints every column with its index and sample values.
Run this to identify the correct column positions before extracting.

Usage:
  python src/models/fanniemae_diagnose.py data/raw/fanniemae/2021Q1.csv
"""

import pandas as pd
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python src/models/fanniemae_diagnose.py <path_to_file>")
    sys.exit(1)

path = sys.argv[1]
print(f"Reading first 10 rows of: {os.path.basename(path)}\n")

df = pd.read_csv(path, sep="|", header=None, nrows=10, dtype=str, low_memory=False)

print(f"Total columns: {df.shape[1]}\n")
print(f"{'Col':>4}  {'Sample values (first 5 rows)'}")
print("-" * 80)

for i in range(df.shape[1]):
    vals = df.iloc[:5, i].tolist()
    # Truncate long values
    display = [str(v)[:20] for v in vals]
    print(f"{i:>4}  {display}")

print("\n--- Now look for:")
print("  FICO:        integers like 720, 750, 680 (NOT 360)")
print("  DTI:         values like 36, 42, 28 (NOT 2.9)")
print("  LTV:         values like 80, 95, 75")
print("  UPB:         values like 320000, 450000, 175000")
print("  State:       two-letter codes like FL, NC, VA")
print("  Loan purpose: P, C, R, or U")
print("  Seller name: bank names")
print("  Loan ID:     alphanumeric string (unique per loan)")
print("  Interest rate: values like 2.875, 3.25, 3.0")