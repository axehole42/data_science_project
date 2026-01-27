import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

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
    
    # 2. Formal Logit Test (Rubin Framework)
    # y = 1 if missing, 0 if observed
    y = is_missing.astype(int)
    
    # If too few missing values, Logit won't converge reliably
    if count_missing < 30:
        stats_list.append({
            'Feature': feat.replace('_', '\\_'),
            'N': f"{count_obs:,}",
            'MissPct': f"{pct_missing:.3f}\"%",
            'LLR_p': "---",
            'PR2': "---",
            'Mechanism': "n/a (<30 miss)"
        })
        continue

    # Predictors: Size and Target
    X = df[['log_assets', 'target']].copy()
    X = sm.add_constant(X)
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
        p_llr = model.llr_pvalue
        pr2 = model.prsquared
        
        # Identify Significant Predictors (p < 0.05)
        sig_vars = []
        if model.pvalues['log_assets'] < 0.05: sig_vars.append('Size')
        if model.pvalues['target'] < 0.05: sig_vars.append('Tgt')
        
        if p_llr < 0.05:
            pred_str = f"({', '.join(sig_vars)})" if sig_vars else ""
            conclusion = f"MAR {pred_str}"
        else:
            conclusion = "MCAR"
            
        stats_list.append({
            'Feature': feat.replace('_', '\\_'),
            'N': f"{count_obs:,}",
            'MissPct': f"{pct_missing:.2f}\"%",
            'LLR_p': f"{p_llr:.2e}",
            'PR2': f"{pr2:.3f}",
            'Mechanism': conclusion
        })
    except:
        stats_list.append({
            'Feature': feat.replace('_', '\\_'),
            'N': f"{count_obs:,}",
            'MissPct': f"{pct_missing:.2f}\"%",
            'LLR_p': "err",
            'PR2': "---",
            'Mechanism': "Logit Error"
        })

# LaTeX Generation
print("\\begin{table}[h]")
print("\\centering")
print("\\scriptsize")
print("\\caption{Formal Rubin Test for Missingness: Logistic Regression of $P(Missing)$ on Size and Target}")
print("\\label{tab:rubin_test_full}")
print("\\begin{tabular}{lrrcccl}")
print("\\toprule")
print("\\textbf{Feature} & \\textbf{N} & \\textbf{Miss \\%} & \\textbf{LLR p-val} & \\textbf{Pseudo $R^2$} & \\textbf{Conclusion} \\ \\ ")
print("\\midrule")

for row in stats_list:
    line = f"{row['Feature']} & {row['N']} & {row['MissPct']} & {row['LLR_p']} & {row['PR2']} & {row['Mechanism']} \\"
    print(line)

print("\\bottomrule")
print("\\multicolumn{6}{l}{\\footnotesize \\textit{Note: LLR p-val < 0.05 rejects MCAR. Size = log\\_assets, Tgt = ROA Improvement label.}} \\ ")
print("\\end{tabular}")
print("\\end{table}")
