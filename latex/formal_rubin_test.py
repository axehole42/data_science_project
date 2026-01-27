import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

# Load data
df = pd.read_parquet('task_data/features.parquet')
with open('task_data/feature_groups.json', 'r') as f:
    groups = json.load(f)

engineered_features = []
for group_features in groups.values():
    engineered_features.extend(group_features)

# Selection for the formal test
missing_counts = df[engineered_features].isna().sum()
top_missing = missing_counts[missing_counts > 1000].sort_values(ascending=False).index.tolist()

results = []

print(f"{'Feature':<30} | {'Logit LLR p-value':<18} | {'Pseudo R2':<10} | {'Significant Predictors'}")
print("-" * 100)

for feat in top_missing:
    # Create missingness indicator
    y = df[feat].isna().astype(int)
    
    # Predictors: Size (log_assets) and Performance (target)
    # We add a constant for the intercept
    X = df[['log_assets', 'target']].copy()
    X = sm.add_constant(X)
    
    # Fit Logit model
    try:
        model = sm.Logit(y, X).fit(disp=0)
        p_llr = model.llr_pvalue
        pseudo_r2 = model.prsquared
        
        # Check which predictors are significant (p < 0.05)
        pvals = model.pvalues
        sig_preds = []
        if pvals['log_assets'] < 0.05: sig_preds.append('Size')
        if pvals['target'] < 0.05: sig_preds.append('Target')
        
        conclusion = ", ".join(sig_preds) if sig_preds else "None"
        
        print(f"{feat:<30} | {p_llr:<18.4e} | {pseudo_r2:<10.4f} | {conclusion}")
    except:
        continue

print("-" * 100)
print("Interpretation:")
print("1. If LLR p-value < 0.05, we reject MCAR in favor of MAR.")
print("2. Significant Predictors show WHICH observed variables determine the missingness.")
