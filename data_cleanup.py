import pandas as pd
import numpy as np
import os

# Configuration
RAW_FILE_PATH = 'task_data/itaiif_compustat_data_24112025.csv'
OUTPUT_FILE_PATH = 'task_data/cleaned_data.parquet'

def print_step_summary(step_name, initial_count, final_count, details=None):
    """Helper to print a readable summary of a cleaning step."""
    dropped = initial_count - final_count
    print(f"\n[STEP] {step_name}")
    print(f"   Input Rows:  {initial_count}")
    print(f"   Output Rows: {final_count}")
    print(f"   Dropped:     {dropped} rows")
    if details:
        print(f"   Details:     {details}")
    print("-" * 50)

def clean_data():
    print("==========================================")
    print("      COMPUSTAT ANNUAL DATA CLEANUP       ")
    print("==========================================")
    
    # 1. Load Raw Data
    if not os.path.exists(RAW_FILE_PATH):
        print(f"Error: Raw file not found at {RAW_FILE_PATH}")
        return None
        
    print(f"Loading raw data: {RAW_FILE_PATH}...")
    df = pd.read_csv(RAW_FILE_PATH)
    initial_shape = df.shape
    print(f"Initial Shape: {initial_shape}")
    
    current_count = len(df)
    
    # 2. Standard Compustat Annual Filters
    # We apply these one by one to see exactly what gets dropped.
    
    # Filter A: Industry Format = INDL (Industrial)
    # FS (Financial Services) often have different accounting structures, 
    # but usually for general research we focus on INDL unless specified otherwise.
    # If the user wants ALL, we can skip. But usually duplicates arise from having both INDL and FS.
    # Let's check if 'indfmt' exists.
    if 'indfmt' in df.columns:
        df_indl = df[df['indfmt'] == 'INDL']
        print_step_summary("Filter: Industry Format (INDL)", current_count, len(df_indl), "Kept only 'INDL'")
        df = df_indl
        current_count = len(df)

    # Filter B: Data Format = STD (Standardized)
    if 'datafmt' in df.columns:
        df_std = df[df['datafmt'] == 'STD']
        print_step_summary("Filter: Data Format (STD)", current_count, len(df_std), "Kept only 'STD'")
        df = df_std
        current_count = len(df)

    # Filter C: Population Source = D (Domestic / US)
    # Often 'I' (International) is a duplicate or separate set.
    if 'popsrc' in df.columns:
        df_pop = df[df['popsrc'] == 'D']
        print_step_summary("Filter: Population Source (Domestic)", current_count, len(df_pop), "Kept only 'D'")
        df = df_pop
        current_count = len(df)

    # Filter D: Consolidation Level = C (Consolidated)
    # P (Parent) is usually a subset and causes duplicates.
    if 'consol' in df.columns:
        df_con = df[df['consol'] == 'C']
        print_step_summary("Filter: Consolidation (C)", current_count, len(df_con), "Kept only 'C'")
        df = df_con
        current_count = len(df)

    # Filter E: Missing or Zero Key Financials (Assets & Net Income)
    # We need these for the core analysis. 'at' must be > 0 for ratio denominators.
    # We KEEP negative niadj values as they are essential for prediction.
    cols_to_check = ['at', 'niadj']
    # Check if columns exist first
    existing_cols = [c for c in cols_to_check if c in df.columns]
    if len(existing_cols) == len(cols_to_check):
        # 1. Drop NaN in key columns
        df_clean = df.dropna(subset=cols_to_check)
        
        # 2. Drop rows where Assets are exactly 0 (cannot compute ratios)
        # Note: We keep negative niadj, only at must be > 0.
        df_clean = df_clean[df_clean['at'] > 0]
        
        print_step_summary("Filter: Valid Financials (at > 0, niadj exists)", current_count, len(df_clean), "Dropped NaN or at=0; Kept negative niadj")
        df = df_clean
        current_count = len(df)
    else:
        print(f"   WARNING: Could not filter for missing {cols_to_check} because one or more are missing from columns.")

    # 3. Explicit Duplicate Check (GVKEY + FYEAR)
    # After standard filters, we should have unique company-years.
    print("\n[CHECK] Checking for remaining duplicates on GVKEY + FYEAR...")
    
    # Sort first: recent datadate last (so we keep the latest update if multiple exist)
    if 'datadate' in df.columns:
        df['datadate'] = pd.to_datetime(df['datadate'], format='%Y-%m-%d', errors='coerce')
        df.sort_values(by=['gvkey', 'fyear', 'datadate'], inplace=True)
    else:
        df.sort_values(by=['gvkey', 'fyear'], inplace=True)

    duplicates = df[df.duplicated(subset=['gvkey', 'fyear'], keep=False)]
    num_duplicates = len(duplicates)
    
    if num_duplicates > 0:
        print(f"   WARNING: Found {num_duplicates} duplicate rows (sharing same gvkey+fyear).")
        print("   Sample of duplicates:")
        print(duplicates[['gvkey', 'fyear', 'datadate']].head())
        
        # Drop duplicates, keeping the LAST one (latest datadate due to sort)
        df.drop_duplicates(subset=['gvkey', 'fyear'], keep='last', inplace=True)
        print_step_summary("Drop Remaining Duplicates", current_count, len(df), "Kept last entry per GVKEY-FYEAR")
        current_count = len(df) # Update count
    else:
        print("   PASSED: No remaining duplicates found.")

    # 4. Fix Mixed Data Types (Text -> Numeric)
    # Columns like 'prcc_c' (Price) sometimes have comma decimals (e.g. '0,0001') which crash Parquet
    print("\n[FIX] Standardizing numeric columns...")
    cols_to_fix = ['prcc_c', 'prcc_f'] # Known problematic columns
    
    # Also check other object columns that might be numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            # Heuristic: if column name implies numeric (e.g., ends in 'c' or 'f' or is known financial)
            # For now, stick to the known culprits to be safe, plus loop if specifically requested.
            pass
            
    for col in cols_to_fix:
        if col in df.columns and df[col].dtype == 'object':
            print(f"   Fixing mixed types in '{col}'...")
            # Replace ',' with '.' and convert
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. Save as Parquet
    print(f"\nSaving to {OUTPUT_FILE_PATH}...")
    # Ensure no columns are dropped (we only filtered rows)
    df.to_parquet(OUTPUT_FILE_PATH, index=False)
    
    print("==========================================")
    print(f" FINAL DATASET: {len(df)} rows")
    print("==========================================")
    return df

if __name__ == "__main__":
    clean_data()
