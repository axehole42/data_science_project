import pandas as pd
import numpy as np

FILE_PATH = 'task_data/itaiif_compustat_data_24112025.csv'

def clean_data():
    """
    Performs initial data cleaning:
    1. Loads data
    2. Sorts by Company (gvkey) and Year (fyear)
    3. Drops duplicate rows
    4. Formats dates
    """
    print("--- Starting Data Cleanup ---")
    
    # 1. Load
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"Loaded data: {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Could not find file at {FILE_PATH}")
        return None

    # 2. Sort
    # Critical for any time-series tasks (lags, leads)
    df.sort_values(by=['gvkey', 'fyear'], inplace=True)
    print("Data sorted by gvkey and fyear.")

    # 3. Drop Duplicates
    # Ensure one record per company per year
    initial_count = len(df)
    df.drop_duplicates(subset=['gvkey', 'fyear'], inplace=True)
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} duplicate rows.")
    else:
        print("No duplicates found.")

    # 4. Format Date
    if 'datadate' in df.columns:
        df['datadate'] = pd.to_datetime(df['datadate'], format='%Y-%m-%d', errors='coerce')
        print("Formatted 'datadate' to datetime objects.")

    print("--- Cleanup Complete (In-Memory) ---")
    print(df.head())
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_data()
