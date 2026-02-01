import pandas as pd
import numpy as np

FILE_PATH = 'task_data/cleaned_data.parquet'

def inspect_data_types():
    print(f"Reading {FILE_PATH}...")
    try:
        df = pd.read_parquet(FILE_PATH)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Shape: {df.shape}")
    
    print("\n" + "="*60)
    print(" 1. NON-NUMERIC COLUMNS (Potential Fixes Needed)")
    print("="*60)
    
    # Check for object columns that are not identifiers
    # We define identifiers/text fields we expect to be strings
    expected_text = ['gvkey', 'conm', 'tic', 'cusip', 'cik', 'indfmt', 'datafmt', 'popsrc', 'consol', 'curcd', 'costat', 'fic', 'exchg']
    
    object_cols = df.select_dtypes(include=['object']).columns
    
    problematic_cols = []
    
    for col in object_cols:
        if col not in expected_text and not col.endswith('date'): # rough check
            # Check a sample of values
            sample = df[col].dropna().unique()[:5]
            print(f"Column '{col}' is OBJECT. Sample values: {sample}")
            problematic_cols.append(col)
            
    if not problematic_cols:
        print("Good news! All other columns seem to be numeric or expected text.")
    else:
        print(f"\n[ACTION REQUIRED] The following columns need 'destringing' (conversion to float):")
        print(problematic_cols)

    print("\n" + "="*60)
    print(" 2. CURRENCY CHECK (Standard Practice)")
    print("="*60)
    if 'curcd' in df.columns:
        print(df['curcd'].value_counts())
        non_usd = df[df['curcd'] != 'USD']
        if len(non_usd) > 0:
            print(f"\n[WARNING] Found {len(non_usd)} rows with non-USD currency!")
            print("Standard Practice: Drop non-USD rows to ensure comparability, unless you perform currency conversion.")
    else:
        print("Column 'curcd' not found.")

    print("\n" + "="*60)
    print(" 3. MISSING DATA (Standard Practice Drops)")
    print("="*60)
    # Check for columns with > 90% missing
    missing = df.isnull().mean()
    high_missing = missing[missing > 0.9]
    if not high_missing.empty:
        print(f"Found {len(high_missing)} columns with > 90% missing values:")
        print(high_missing.index.tolist())
        print("Standard Practice: Consider dropping these unless they are sparse but critical features.")
    
    # Check rows with missing Total Assets (at) - usually a sign of bad data
    if 'at' in df.columns:
        missing_at = df['at'].isnull().sum()
        if missing_at > 0:
            print(f"\n[WARNING] Found {missing_at} rows with missing Total Assets ('at').")
            print("Standard Practice: Drop these rows as 'at' is the denominator for most financial ratios.")

if __name__ == "__main__":
    inspect_data_types()
