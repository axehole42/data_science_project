import pandas as pd

FILE_PATH = 'task_data/cleaned_data.parquet'

def inspect_structure():
    print(f"Reading {FILE_PATH}...")
    try:
        df = pd.read_parquet(FILE_PATH)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return
    
    print("\n" + "="*60)
    print(" 1. FREQUENCY CHECK (Annual vs. Quarterly)")
    print("="*60)
    
    # Check if a single company-year has multiple dates
    # We look at the top 5 companies that have the most rows
    top_companies = df['gvkey'].value_counts().head(5).index
    
    for gvkey in top_companies:
        subset = df[df['gvkey'] == gvkey].sort_values('datadate')
        years = subset['fyear'].unique()
        
        print(f"\nGVKEY: {gvkey}")
        print(f"Total Rows: {len(subset)}")
        print(f"Unique Years: {len(years)}")
        
        # Check duplicates per year
        dupes_per_year = subset.groupby('fyear').size()
        multi_entry_years = dupes_per_year[dupes_per_year > 1]
        
        if not multi_entry_years.empty:
            print(f"!! This company has multiple entries for these years: {list(multi_entry_years.index)}")
            print("Sample of multiple entries:")
            # Show columns that might differentiate rows
            cols_to_show = ['gvkey', 'fyear', 'datadate', 'indfmt', 'consol', 'datafmt', 'popsrc']
            # Only keep columns that actually exist in the dataframe
            cols_to_show = [c for c in cols_to_show if c in df.columns]
            
            print(subset[subset['fyear'].isin(multi_entry_years.index)][cols_to_show].head(10))
        else:
            print("Looks like 1 row per year (Annual Data).")

    print("\n" + "="*60)
    print(" 2. DUPLICATE ANALYSIS")
    print("="*60)
    
    # Check strict duplicates on gvkey + fyear
    dupes = df[df.duplicated(subset=['gvkey', 'fyear'], keep=False)]
    
    if len(dupes) > 0:
        print(f"Total rows sharing the same GVKEY + FYEAR: {len(dupes)}")
        print("This confirms we have multiple rows per year.")
        print("Check the 'Sample of multiple entries' above to see IF they differ by Quarter (datadate) or Format (indfmt/consol).")
    else:
        print("No duplicates found based on GVKEY + FYEAR.")

if __name__ == "__main__":
    inspect_structure()
