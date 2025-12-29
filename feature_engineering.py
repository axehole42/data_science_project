import pandas as pd
import numpy as np
import os

INPUT_FILE = 'task_data/cleaned_data.parquet'
OUTPUT_FILE = 'task_data/features.parquet'

def construct_features():
    print("==========================================")
    print("      FEATURE ENGINEERING (ROA TARGET)    ")
    print("==========================================")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")

    # 1. Sort for Time Series operations
    df.sort_values(by=['gvkey', 'fyear'], inplace=True)

    print("Constructing Financial Ratios...")

    # --- A. CORE RATIOS (Year t) ---
    
    # 1. Profitability: Return on Assets (ROA)
    df['roa'] = df['niadj'] / df['at']

    # 2. Cash Flow: OCF to Assets
    df['ocf_to_assets'] = df['oancf'] / df['at']

    # 3. Earnings Quality: Accruals
    df['accruals'] = (df['niadj'] - df['oancf']) / df['at']

    # 4. Liquidity: Current Ratio
    df['current_ratio'] = df['act'] / df['lct']

    # 5. Liquidity: Cash Ratio
    df['cash_ratio'] = df['che'] / df['lct']

    # 6. Liquidity: Working Capital to Assets
    # Formula: (ACT - LCT) / AT
    df['working_cap_to_assets'] = (df['act'] - df['lct']) / df['at']

    # 7. Leverage: Financial Debt to Assets
    # Formula: (DLTT + DLC) / AT
    # DLTT = Long Term Debt, DLC = Debt in Current Liab
    # ASSUMPTION: We fill missing debt values with 0. 
    # Rationale: In Compustat, missing often implies 'Not Applicable' (no debt) rather than data loss.
    dltt_temp = df['dltt'].fillna(0)
    dlc_temp = df['dlc'].fillna(0)
    df['debt_to_assets'] = (dltt_temp + dlc_temp) / df['at']

    # 8. Leverage: Debt to Equity
    df['debt_to_equity'] = df['lt'] / df['seq']
    
    # 9. Backup Leverage: Total Liab to Assets
    df['total_liab_to_assets'] = df['lt'] / df['at']

    # 10. Size: Log Assets
    # Safer calculation: handle non-positive values just in case
    df['log_size'] = np.where(df['at'] > 0, np.log(df['at']), np.nan)

    # --- B. GROWTH & TREND FEATURES (Lags) ---
    # STRICT TIME LAGS: We must ensure lags are actually from t-1
    
    # Create lag columns
    df['fyear_lag1'] = df.groupby('gvkey')['fyear'].shift(1)
    df['at_lag1'] = df.groupby('gvkey')['at'].shift(1)
    df['roa_lag1'] = df.groupby('gvkey')['roa'].shift(1)
    df['debt_to_assets_lag1'] = df.groupby('gvkey')['debt_to_assets'].shift(1)
    df['current_ratio_lag1'] = df.groupby('gvkey')['current_ratio'].shift(1)
    
    # Check if lag is valid (fyear_lag1 == fyear - 1)
    # If not valid (gap in data), set lag feature to NaN
    lag_mask = (df['fyear_lag1'] == df['fyear'] - 1)
    
    # 11. Asset Growth
    df['asset_growth'] = np.where(lag_mask, (df['at'] - df['at_lag1']) / df['at_lag1'], np.nan)

    # 12. Delta ROA (Trend)
    df['delta_roa'] = np.where(lag_mask, df['roa'] - df['roa_lag1'], np.nan)

    # 13. Delta Leverage (Trend)
    df['delta_leverage'] = np.where(lag_mask, df['debt_to_assets'] - df['debt_to_assets_lag1'], np.nan)

    # 14. Delta Current Ratio (Trend)
    df['delta_curr_ratio'] = np.where(lag_mask, df['current_ratio'] - df['current_ratio_lag1'], np.nan)

    # --- HANDLING INFINITIES ---
    # Division by zero (e.g. LCT=0 in Current Ratio) creates inf/-inf.
    # We treat these as NaN (Missing) rather than extreme values to prevent model distortion.
    print("Handling Infinities (Replacing with NaN)...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)


    # --- C. TARGET CREATION (The Fix) ---
    print("Creating Target Variable (Strict t+1 Check)...")

    # Goal: Predict if ROA(t+1) > ROA(t)
    
    # 1. Get Next Year Data
    df['roa_next'] = df.groupby('gvkey')['roa'].shift(-1)
    df['fyear_next'] = df.groupby('gvkey')['fyear'].shift(-1)
    
    # 2. Strict Check: Is the next row actually the next year?
    # If fyear_next != fyear + 1, then we DO NOT have a valid target.
    valid_target_mask = (df['fyear_next'] == df['fyear'] + 1)
    
    # Filter: Keep only rows where we have a valid next year
    df_valid = df[valid_target_mask].copy()
    
    dropped_rows = len(df) - len(df_valid)
    print(f"Dropped {dropped_rows} rows (Last year of data OR gaps in time series).")

    # 3. Create Binary Target
    # y = 1 if ROA_next > ROA_current
    df_valid['target'] = (df_valid['roa_next'] > df_valid['roa']).astype(int)

    # --- D. CLEANUP & SAVE ---
    # Drop intermediate helper columns only
    cols_to_drop = ['at_lag1', 'roa_lag1', 'debt_to_assets_lag1', 'current_ratio_lag1', 'fyear_lag1', 'roa_next', 'fyear_next']
    df_final = df_valid.drop(columns=[c for c in cols_to_drop if c in df_valid.columns])
    
    # Reorder columns: Put Identifiers, Target, and Key Features at the very front
    important_cols = [
        'gvkey', 'fyear', 'datadate', 'conm', 'indfmt', 'datafmt', # Identifiers
        'target',                 # Y
        'roa', 'ocf_to_assets', 'accruals', # Profitability/Quality
        'current_ratio', 'cash_ratio', 'working_cap_to_assets', # Liquidity
        'debt_to_assets', 'debt_to_equity', 'total_liab_to_assets', # Leverage
        'asset_growth', 'log_size', # Growth/Size
        'delta_roa', 'delta_leverage', 'delta_curr_ratio' # Trends
    ]
    
    # Get all other columns that weren't in the 'important' list
    other_cols = [c for c in df_final.columns if c not in important_cols]
    
    # Reassemble the dataframe with important columns first
    df_final = df_final[important_cols + other_cols]

    print(f"Saving {len(df_final)} rows to {OUTPUT_FILE}...")
    df_final.to_parquet(OUTPUT_FILE, index=False)

    print("\nFeature Statistics (New Columns):")
    # Only show stats for the new features we created (columns 3 to 17 in important_cols)
    print(df_final[important_cols[3:]].describe().T[['count', 'mean', 'std', 'min', 'max']])

    print("\nMissing Values (NaN) Summary (New Columns):")
    print(df_final[important_cols[3:]].isna().sum())
    
    # Check Class Balance
    balance = df_final['target'].value_counts(normalize=True)
    print("\nClass Balance (Target):")
    print(balance)

    print("==========================================")
    print("        FEATURE ENGINEERING DONE          ")
    print("==========================================")

if __name__ == "__main__":
    construct_features()
