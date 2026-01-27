import pandas as pd
import numpy as np
from scipy import stats
import json

# Load data and feature definitions
df = pd.read_parquet('task_data/features.parquet')
with open('task_data/feature_groups.json', 'r') as f:
    groups = json.load(f)

engineered_features = []
for group_features in groups.values():
    engineered_features.extend(group_features)

cols_to_check = [f for f in engineered_features if f in df.columns]

# Selection of representative features for a clean table
# 1. High missingness Large Bias
# 2. High missingness Small Bias
# 3. Size and Target anchors
representative_feats = [
    'delta_current_ratio',
    'interest_coverage_ebitda',
    'quick_ratio',
    'ni_growth',
    'asset_growth',
    'roa_lag1',
    'fcf_to_debt',
    'capx_to_assets'
]

stats_list = []
for feat in representative_feats:
    is_missing = df[feat].isna()
    pct_missing = (is_missing.sum() / len(df)) * 100
    
    size_obs = df.loc[~is_missing, 'log_assets'].mean()
    size_miss = df.loc[is_missing, 'log_assets'].mean()
    
    tgt_obs = df.loc[~is_missing, 'target'].mean()
    tgt_miss = df.loc[is_missing, 'target'].mean()
    
    # Determine Status for Table
    t_stat_size, p_val_size = stats.ttest_ind(df.loc[~is_missing, 'log_assets'], df.loc[is_missing, 'log_assets'], equal_var=False)
    conclusion = "MCAR"
    if p_val_size < 0.001:
        conclusion = "MAR (Large)" if size_obs < size_miss else "MAR (Small)"

    stats_list.append({
        'Feature': feat.replace('_', '\\_'),
        'MissPct': f"{pct_missing:.1f}\\%",
        'SizeObs': f"{size_obs:.2f}",
        'SizeMiss': f"{size_miss:.2f}",
        'TgtObs': f"{tgt_obs:.2f}",
        'TgtMiss': f"{tgt_miss:.2f}",
        'Mechanism': conclusion
    })

# LaTeX Generation
print("\\begin{table}[h]")
print("\\centering")
print("\\small")
print("\\caption{Analysis of Missing Data Mechanisms (Representative Features)}")
print("\\label{tab:missing_mechanisms}")
print("\\begin{tabular}{lcccccl}")
print("\\toprule")
print("\\textbf{Feature} & \\textbf{\\% Miss} & \\textbf{Size(Obs)} & \\textbf{Size(Miss)} & \\textbf{Tgt(Obs)} & \\textbf{Tgt(Miss)} & \\textbf{Conclusion} \\ \\")
print("\\midrule")

for row in stats_list:
    line = f"{row['Feature']} & {row['MissPct']} & {row['SizeObs']} & {row['SizeMiss']} & {row['TgtObs']} & {row['TgtMiss']} & {row['Mechanism']} \\"
    print(line)

print("\\bottomrule")
print("\\multicolumn{7}{l}{\\footnotesize \\textit{Note: Size is measured as log\\_assets. Tgt is the mean ROA improvement rate.}} \\")
print("\\end{tabular}")
print("\\end{table}")
