import pandas as pd
import numpy as np
from scipy import stats
import json

# Load data and feature definitions
df = pd.read_parquet('task_data/features.parquet')
with open('task_data/feature_groups.json', 'r') as f:
    groups = json.load(f)

# Flatten list
engineered_features = []
for group_features in groups.values():
    engineered_features.extend(group_features)

exclude_list = {'total_accruals'}
cols_to_check = [f for f in engineered_features if f not in exclude_list and f in df.columns]
cols_to_check.sort()

stats_list = []
total_n = len(df)

for feat in cols_to_check:
    is_missing = df[feat].isna()
    count_missing = is_missing.sum()
    count_obs = total_n - count_missing
    
    if count_missing == 0: continue
    
    pct_missing = (count_missing / total_n) * 100
    
    size_obs = df.loc[~is_missing, 'log_assets'].mean()
    tgt_obs = df.loc[~is_missing, 'target'].mean()
    
    # Threshold for statistical validity
    if count_missing >= 30:
        size_miss_val = df.loc[is_missing, 'log_assets'].mean()
        size_miss = f"{size_miss_val:.2f}"
        tgt_miss = f"{df.loc[is_missing, 'target'].mean():.2f}"
        
        # Mechanism Determination
        t_stat_size, p_val_size = stats.ttest_ind(df.loc[~is_missing, 'log_assets'], df.loc[is_missing, 'log_assets'], equal_var=False)
        if p_val_size < 0.001:
            conclusion = "MAR (L)" if size_obs < size_miss_val else "MAR (S)"
        else:
            conclusion = "MCAR"
    else:
        size_miss = "---"
        tgt_miss = "---"
        conclusion = "n/a"

    stats_list.append({
        'Feature': feat.replace('_', '\\_'),
        'N': f"{count_obs:,}",
        'Miss': f"{count_missing:,}",
        'MissPct': f"{pct_missing:.2f}\"%",
        'SizeObs': f"{size_obs:.2f}",
        'SizeMiss': size_miss,
        'TgtObs': f"{tgt_obs:.2f}",
        'TgtMiss': tgt_miss,
        'Mechanism': conclusion
    })

# LaTeX Generation
print("\\begin{table}[h]")
print("\\centering")
print("\\scriptsize")
print("\\caption{Full Analysis of Missing Data Mechanisms for Engineered Ratios}")
print("\\label{tab:missing_mechanisms_full}")
print("\\begin{tabular}{lrrccccc}")
print("\\toprule")
print("\\textbf{Feature} & \\textbf{N} & \\textbf{Miss} & \\textbf{Miss \\%} & \\textbf{Size(O)} & \\textbf{Size(M)} & \\textbf{Tgt(O)} & \\textbf{Tgt(M)} & \\textbf{Conclusion} \\")
print("\\midrule")

for row in stats_list:
    line = f"{row['Feature']} & {row['N']} & {row['Miss']} & {row['MissPct']} & {row['SizeObs']} & {row['SizeMiss']} & {row['TgtObs']} & {row['TgtMiss']} & {row['Mechanism']} \\"
    print(line)

print("\\bottomrule")
print("\\multicolumn{9}{l}{\\footnotesize \\textit{Note: N = Observed count, Miss = Missing count, Size = log\\_assets. MAR(L)/(S) = Large/Small Bias. '---' < 30.}} \\")
print("\\end{tabular}")
print("\\end{table}")
