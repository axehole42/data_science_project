import pandas as pd
import numpy as np
from scipy import stats
import json

# Load data and feature definitions
df = pd.read_parquet('task_data/features.parquet')

with open('task_data/feature_groups.json', 'r') as f:
    groups = json.load(f)

# Flatten the list to get ONLY engineered features
engineered_features = []
for group_features in groups.values():
    engineered_features.extend(group_features)

# Filter df to just these columns + log_assets/target for analysis
cols_to_check = [f for f in engineered_features if f in df.columns]
missing_counts = df[cols_to_check].isna().sum()

# Filter for those with actual missing values
missing_features = missing_counts[missing_counts > 0].index.tolist()

# Sort by number of missing values (descending)
missing_features = sorted(missing_features, key=lambda x: missing_counts[x], reverse=True)

# Anchors
anchors = ['log_assets', 'target'] 

print(f"{'Feature':<35} | {'% Miss':<8} | {'Size(Obs)':<9} | {'Size(Miss)':<9} | {'Tgt(Obs)':<8} | {'Tgt(Miss)':<8} | {'Conclusion'}")
print("-" * 115)

for feat in missing_features:
    # Skip if feature is one of our anchors
    if feat in anchors:
        continue
        
    is_missing = df[feat].isna()
    pct_missing = (is_missing.sum() / len(df)) * 100
    
    # Skip negligible missingness
    if pct_missing < 0.1:
        continue

    # 1. Check Size Bias
    size_miss = df.loc[is_missing, 'log_assets']
    size_obs = df.loc[~is_missing, 'log_assets']
    
    t_stat_size, p_val_size = stats.ttest_ind(size_obs, size_miss, equal_var=False)
    
    # 2. Check Target Bias
    tgt_miss = df.loc[is_missing, 'target']
    tgt_obs = df.loc[~is_missing, 'target']
    
    mean_tgt_miss = tgt_miss.mean()
    mean_tgt_obs = tgt_obs.mean()
    
    # Determine Status
    is_biased_size = p_val_size < 0.001
    size_diff = size_obs.mean() - size_miss.mean()
    
    conclusion = "MCAR (Likely)"
    if is_biased_size:
        if size_diff > 0:
            conclusion = "MAR (Small Firms)"
        else:
            conclusion = "MAR (Large Firms)"
            
    # Add to table
    print(f"{feat:<35} | {pct_missing:>6.2f}%  | {size_obs.mean():>9.2f} | {size_miss.mean():>9.2f} | {mean_tgt_obs:>8.2f} | {mean_tgt_miss:>8.2f} | {conclusion}")

print("-" * 115)
