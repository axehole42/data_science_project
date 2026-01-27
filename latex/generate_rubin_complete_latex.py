import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
from scipy import stats

# Load data and feature definitions
df = pd.read_parquet('task_data/features.parquet')
with open('task_data/feature_groups.json', 'r') as f:
    groups = json.load(f)

# Flatten list to get engineered features
engineered_features = []
for group_features in groups.values():
    engineered_features.extend(group_features)

exclude_list = {'total_accruals'}
all_features = [f for f in engineered_features if f in df.columns and f not in exclude_list and df[f].isna().any()]
all_features.sort()

total_rows = len(df)
stats_list = []

for feat in all_features:
    # 1. Basic Stats
    is_missing = df[feat].isna()
    count_missing = is_missing.sum()
    count_obs = total_rows - count_missing
    pct_missing = (count_missing / total_rows) * 100
    
    # Threshold for validity
    if count_missing < 30:
        stats_list.append({
            'Feature': feat.replace('_', '\\_'),
            'N': f"{count_obs:,}",
            'MissPct': f"{pct_missing:.3f}\\%",
            'Size_p': "---",
            'Nested_p': "---",
            'PR2': "---",
            'Mechanism': "n/a"
        })
        continue

    y = is_missing.astype(int)
    X0 = sm.add_constant(df['log_assets'])
    X1 = sm.add_constant(df[['log_assets', 'target']])
    
    try:
        res0 = sm.Logit(y, X0).fit(disp=0)
        res1 = sm.Logit(y, X1).fit(disp=0)
        
        # 1. Size significance (from Model 0 coefficient)
        p_size = res0.pvalues['log_assets']
        is_size_sig = p_size < 0.05
        
        # 2. Incremental LRT (Model 0 vs Model 1)
        lrt_stat = 2 * (res1.llf - res0.llf)
        p_lrt_target = stats.chi2.sf(lrt_stat, 1) # 1 d.f.
        is_target_sig = p_lrt_target < 0.05
        
        # Pseudo R2 from full model
        pr2 = res1.prsquared
        
        # Determine Mechanism
        if is_size_sig and is_target_sig:
            conclusion = "MAR (Size, Tgt)"
        elif is_size_sig:
            conclusion = "MAR (Size)"
        elif is_target_sig:
            conclusion = "MAR (Tgt)"
        else:
            conclusion = "MCAR"
            
        stats_list.append({
            'Feature': feat.replace('_', '\\_'),
            'N': f"{count_obs:,}",
            'MissPct': f"{pct_missing:.2f}\\%",
            'Size_p': f"{p_size:.2e}",
            'Nested_p': f"{p_lrt_target:.2e}", 
            'PR2': f"{pr2:.3f}",
            'Mechanism': conclusion
        })
    except:
        stats_list.append({'Feature': feat.replace('_', '\\_'), 'N': '---', 'MissPct': '---', 'Size_p': '---', 'Nested_p': '---', 'PR2': '---', 'Mechanism': 'Error'})

# LaTeX Generation
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\scriptsize")
print(r"\caption{Formal Rubin Test: Diagnosis via Baseline Size, Incremental Nested LRT, and Pseudo $R^2$}")
print(r"\label{tab:rubin_test_final_complete}")
print(r"\begin{tabular}{lrrcccccl}")
print(r"\toprule")
print(r"\textbf{Feature} & \textbf{N} & \textbf{Miss %} & \textbf{Size p-val} & \textbf{Nested LRT p-val} & \textbf{Pseudo $R^2$} & \textbf{Conclusion} \\")
print(r"\midrule")

for row in stats_list:
    line = f"{row['Feature']} & {row['N']} & {row['MissPct']} & {row['Size_p']} & {row['Nested_p']} & {row['PR2']} & {row['Mechanism']} \\\\"
    print(line)

print(r"\bottomrule")
print(r"\multicolumn{7}{l}{\footnotesize \textit{Note: Nested LRT p-val tests incremental Target power. Pseudo $R^2$ is McFadden's from the full model.}} \\")
print(r"\end{tabular}")
print(r"\end{table}")
