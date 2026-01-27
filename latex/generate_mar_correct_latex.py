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
for feat in cols_to_check:
    is_missing = df[feat].isna()
    count_missing = is_missing.sum()
    if count_missing == 0: continue
    
    pct_missing = (count_missing / len(df)) * 100
    
    size_obs = df.loc[~is_missing, 'log_assets'].mean()
    tgt_obs = df.loc[~is_missing, 'target'].mean()
    
    # Threshold for statistical validity
    if count_missing >= 30:
        size_miss = f"{df.loc[is_missing, 'log_assets'].mean():.2f}"
        tgt_miss = f"{df.loc[is_missing, 'target'].mean():.2f}"
        
        # Mechanism Determination
        t_stat_size, p_val_size = stats.ttest_ind(df.loc[~is_missing, 'log_assets'], df.loc[is_missing, 'log_assets'], equal_var=False)
        if p_val_size < 0.001:
            conclusion = "MAR (L)" if size_obs < float(size_miss) else "MAR (S)"
        else:
            conclusion = "MCAR"
    else:
        # Too small to be representative
        size_miss = "---"
        tgt_miss = "---"
        conclusion = "n/a"

    stats_list.append({
        'Feature': feat.replace('_', '\\_'),
        'MissPct': f"{pct_missing:.3f}\"%",
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
print("\\caption{Analysis of Missing Data Mechanisms for Engineered Ratios (Minimum 30 Observations for Miss Stats)}")
print("\\label{tab:missing_mechanisms_corrected}")
print("\\begin{tabular}{lcccccl}")
print("\\toprule")
print("\\textbf{Feature} & \\textbf{\\% Miss} & \\textbf{Size(Obs)} & \\textbf{Size(Miss)} & \\textbf{Tgt(Obs)} & \\textbf{Tgt(Miss)} & \\textbf{Conclusion} \\\\")
print("\\midrule")

for row in stats_list:
    line = f"{row['Feature']} & {row['MissPct']} & {row['SizeObs']} & {row['SizeMiss']} & {row['TgtObs']} & {row['TgtMiss']} & {row['Mechanism']} \\"
    print(line)

print("\\bottomrule")
print("\\multicolumn{7}{l}{\\footnotesize \\textit{Note: Size = log\\_assets. MAR(L) = Large Firm Bias, MAR(S) = Small Firm Bias. '---' denotes sample size < 30.}} \\\\")
print("\\end{tabular}")
print("\\end{table}")
