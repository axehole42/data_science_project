import pandas as pd

# Load only the relevant columns to save time/memory
df = pd.read_csv('task_data/itaiif_compustat_data_24112025.csv', usecols=['at', 'niadj'])

total_rows = len(df)
missing_at = df['at'].isna().sum()
zero_at = (df['at'] <= 0).sum()
missing_niadj = df['niadj'].isna().sum()

# Overlap check
# Rows that are dropped because AT is missing OR AT <= 0 OR NIADJ is missing
to_drop = df[df['at'].isna() | (df['at'] <= 0) | df['niadj'].isna()]

print(f"Total rows: {total_rows}")
print(f"Missing 'at' (NaN): {missing_at}")
print(f"Zero or Negative 'at' (<= 0): {zero_at}")
print(f"Missing 'niadj' (NaN): {missing_niadj}")
print(f"Total rows meeting drop criteria: {len(to_drop)}")
