import pandas as pd
import numpy as np
import os

INPUT_FILE = 'task_data/cleaned_data.parquet'
OUTPUT_FILE = 'task_data/features.parquet'

def construct_features():
    print("==========================================")
    print("        FEATURE ENGINEERING START         ")
    print("==========================================")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")

    # 1. Sort for Time Series operations
    df.sort_values(by=['gvkey', 'fyear'], inplace=True)

    # 2. Construct Ratios
    # Note: We handle division by zero by allowing inf temporarily, then replacing or letting Winsorization handle it later.
    # For now, we just create the raw ratios.
    
    print("Constructing Financial Ratios...")
    
    # Profitability
    df['roa'] = df['niadj'] / df['at']  # Return on Assets
    df['roe'] = df['niadj'] / df['ceq'] # Return on Equity
    
    # Liquidity
    # Handle division by zero for liabilities
    df['current_ratio'] = df['act'] / df['lct']
    df['cash_ratio'] = df['che'] / df['lct']
    
    # Leverage / Solvency
    df['debt_to_assets'] = df['lt'] / df['at']
    df['debt_to_equity'] = df['lt'] / df['ceq']
    
    # Cash Flow
    df['cfo_to_assets'] = df['oancf'] / df['at']
    
    # Growth (Requires Lag)
    # Group by gvkey to ensure we don't calculate growth across different companies
    df['at_lag1'] = df.groupby('gvkey')['at'].shift(1)
    df['asset_growth'] = (df['at'] - df['at_lag1']) / df['at_lag1']
    
    # 3. Create Target Variable (t+1)
    print("Creating Target Variable (niadj t+1)...")
    
    # Shift niadj backwards by 1 (take next year's value)
    df['niadj_next'] = df.groupby('gvkey')['niadj'].shift(-1)
    
    # Create Binary Target: 1 if niadj_next > 0, else 0
    # Note: Rows where niadj_next is NaN (the last year) will be False in direct comparison if not handled, 
    # but strictly we should drop them because we don't KNOW the target.
    
    # Filter: Drop rows where we don't have next year's data
    initial_len = len(df)
    df.dropna(subset=['niadj_next'], inplace=True)
    dropped_rows = initial_len - len(df)
    print(f"Dropped {dropped_rows} rows (last year of data for each firm) to create valid targets.")
    
    df['target'] = (df['niadj_next'] > 0).astype(int)
    
    # Cleanup intermediate columns
    df.drop(columns=['at_lag1', 'niadj_next'], inplace=True)

    # 4. Save
    print(f"Saving {len(df)} rows to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    
    # Summary of generated features
    new_features = ['roa', 'roe', 'current_ratio', 'cash_ratio', 'debt_to_assets', 'debt_to_equity', 'cfo_to_assets', 'asset_growth', 'target']
    print("\nFeature Summary (Descriptive Stats):")
    print(df[new_features].describe().T[['count', 'mean', 'std', 'min', 'max']])
    
    print("==========================================")
    print("        FEATURE ENGINEERING DONE          ")
    print("==========================================")

if __name__ == "__main__":
    construct_features()
