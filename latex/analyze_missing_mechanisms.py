import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_parquet('task_data/features.parquet')

# Features to analyze (those with missing values)
missing_counts = df.isna().sum()
missing_features = missing_counts[missing_counts > 0].index.tolist()

# Sort by number of missing values (descending)
missing_features = sorted(missing_features, key=lambda x: missing_counts[x], reverse=True)

# We will test against these "Complete" anchors
anchors = ['log_assets', 'target'] 

results = []

print(f"{'Feature':<35} | {'% Miss':<8} | {'Size(Obs)':<9} | {'Size(Miss)':<9} | {'Tgt(Obs)':<8} | {'Tgt(Miss)':<8} | {'Conclusion'}")
print("-" * 115)

for feat in missing_features:
    # Skip if feature is one of our anchors (unlikely for target/log_assets but safe check)
    if feat in anchors:
        continue
        
    # Create mask
    is_missing = df[feat].isna()
    pct_missing = (is_missing.sum() / len(df)) * 100
    
    # If missingness is negligible (<0.1%), skip to avoid noise
    if pct_missing < 0.1:
        continue

    # 1. Check Size Bias (log_assets)
    size_miss = df.loc[is_missing, 'log_assets']
    size_obs = df.loc[~is_missing, 'log_assets']
    
    # Simple T-test for size difference
    t_stat_size, p_val_size = stats.ttest_ind(size_obs, size_miss, equal_var=False)
    
    # 2. Check Target Bias (target)
    # Target is binary, so we compare proportions
    tgt_miss = df.loc[is_missing, 'target']
    tgt_obs = df.loc[~is_missing, 'target']
    
    mean_tgt_miss = tgt_miss.mean()
    mean_tgt_obs = tgt_obs.mean()
    
    # Determine Status
    # We use a strict p-value because sample size is large
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
print("\n* 'MAR (Small Firms)' means missing values are significantly more common in smaller firms.")
print("* 'MAR (Large Firms)' means missing values are significantly more common in larger firms.")
