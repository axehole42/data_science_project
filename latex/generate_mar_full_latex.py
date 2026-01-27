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

# Exclude absolute values
exclude_list = {'total_accruals'}
cols_to_check = [f for f in engineered_features if f not in exclude_list and f in df.columns]
cols_to_check.sort()

stats_list = []
for feat in cols_to_check:
    is_missing = df[feat].isna()
    count_missing = is_missing.sum()
    if count_missing == 0: continue # Skip if no missing values to analyze
    
    pct_missing = (count_missing / len(df)) * 100
    
    size_obs = df.loc[~is_missing, 'log_assets'].mean()
    size_miss = df.loc[is_missing, 'log_assets'].mean()
    
    tgt_obs = df.loc[~is_missing, 'target'].mean()
    tgt_miss = df.loc[is_missing, 'target'].mean()
    
    # Mechanism Determination
    t_stat_size, p_val_size = stats.ttest_ind(df.loc[~is_missing, 'log_assets'], df.loc[is_missing, 'log_assets'], equal_var=False)
    conclusion = "MCAR"
    if p_val_size < 0.001:
        conclusion = "MAR (L)" if size_obs < size_miss else "MAR (S)"

    stats_list.append({
        'Feature': feat.replace('_', '\\_'),
        'MissPct': f"{pct_missing:.1f}\\%",
        'SizeObs': f"{size_obs:.2f}",
        'SizeMiss': f"{size_miss:.2f}",
        'TgtObs': f"{tgt_obs:.2f}",
        'TgtMiss': f"{tgt_miss:.2f}",
        'Mechanism': conclusion
    })

# LaTeX Generation (using longtable for length)
print("\\begin{table}[h]")
print("\\centering")
print("\\scriptsize")
print("\\caption{Full Analysis of Missing Data Mechanisms for Engineered Ratios}")
print("\\label{tab:missing_mechanisms_full}")
print("\\begin{tabular}{lcccccl}")
print("\\toprule")
print("\\textbf{Feature} & \\textbf{\\% Miss} & \\textbf{Size(Obs)} & \\textbf{Size(Miss)} & \\textbf{Tgt(Obs)} & \\textbf{Tgt(Miss)} & \\textbf{Conclusion} \\ \\")
print("\\midrule")

# Split into two columns if needed, but for now just a long list. 
# If it's too long for one page, recommend using \newpage or splitting into two tables.
for row in stats_list:
    line = f"{row['Feature']} & {row['MissPct']} & {row['SizeObs']} & {row['SizeMiss']} & {row['TgtObs']} & {row['TgtMiss']} & {row['Mechanism']} \\"
    print(line)

print("\\bottomrule")
print("\\multicolumn{7}{l}{\\\\footnotesize \\textit{Note: Size = log\\_assets. MAR(L) = Large Firm Bias, MAR(S) = Small Firm Bias.}} \\ \\")
print("\\end{tabular}")
print("\\end{table}")
