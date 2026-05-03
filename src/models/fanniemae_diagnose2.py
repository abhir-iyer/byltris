import pandas as pd
import os

# find any fanniemae csv file
fannie_dir = "data/raw/fanniemae"
files = [f for f in os.listdir(fannie_dir) if f.endswith(".csv")]
print(f"Files found: {files}")

if not files:
    print("No CSV files found. Need to re-download a quarterly file.")
else:
    f = os.path.join(fannie_dir, files[0])
    print(f"\nReading first 5 rows of {files[0]}...")
    df = pd.read_csv(f, header=None, nrows=5, sep="|")
    print(f"Total columns: {len(df.columns)}")
    for i, col in enumerate(df.iloc[0]):
        print(f"Col {i:3d}: {col}")