import pandas as pd
import numpy as np

# Load the cleaned data
df = pd.read_parquet('task_data/cleaned_data.parquet')

# List of variables from the user's table
variables = [
    'at', 'niadj', 'lt', 'act', 'lct', 'che', 'rect', 
    'invt', 'ppent', 'dltt', 'dlc', 'seq', 'oancf', 'capx'
]

# Calculate statistics
stats = []
for var in variables:
    if var in df.columns:
        col_data = df[var]
        stats.append({
            'Variable': var,
            'N': col_data.count(),
            'Mean': col_data.mean(),
            'Std. Dev.': col_data.std(),
            'Median': col_data.median(),
            'Missing': col_data.isna().sum()
        })
    else:
        stats.append({
            'Variable': var,
            'N': 'N/A',
            'Mean': 'N/A',
            'Std. Dev.': 'N/A',
            'Median': 'N/A',
            'Missing': 'N/A'
        })

stats_df = pd.DataFrame(stats)

# Formatting for comparison
pd.options.display.float_format = '{:.2f}'.format
print(stats_df.to_string(index=False))

# Total row count
print(f"\nTotal rows in dataset: {len(df)}")
