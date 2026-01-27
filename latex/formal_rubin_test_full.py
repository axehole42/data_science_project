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

exclude_list = {'total_accruals'}
# Only features with at least 1 missing value
all_features = [f for f in engineered_features if f not in exclude_list and f in df.columns and df[f].isna().any()]
all_features.sort()

results = []

print(f"{'Feature':<35} | {'p-value':<12} | {'Pseudo R2':<10} | {'Predictors'}")
print("-" * 115)

for feat in all_features:
    y = df[feat].isna().astype(int)
    
    # Check if there are enough missing values for a Logit (at least a few)
    if y.sum() < 5:
        print(f"{feat:<35} | {'N/A (too few)':<12} | {'---':<10} | {'---'}")
        continue
        
    X = df[['log_assets', 'target']].copy()
    X = sm.add_constant(X)
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
        p_llr = model.llr_pvalue
        pseudo_r2 = model.prsquared
        
        # Check significance
        pvals = model.pvalues
        sig_preds = []
        if pvals['log_assets'] < 0.05: sig_preds.append('Size')
        if pvals['target'] < 0.05: sig_preds.append('Target')
        
        conclusion = ", ".join(sig_preds) if sig_preds else "None (MCAR)"
        
        # Determine overall mechanism
        if p_llr < 0.05:
            mech = f"MAR ({conclusion})"
        else:
            mech = "MCAR"
            
        print(f"{feat:<35} | {p_llr:<12.2e} | {pseudo_r2:<10.4f} | {mech}")
    except:
        print(f"{feat:<35} | {'ERROR':<12} | {'---':<10} | {'---'}")

print("-" * 115)
