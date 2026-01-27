import pandas as pd
import json

# 1. Load Feature Groups
with open('task_data/feature_groups.json', 'r') as f:
    groups = json.load(f)

# Reverse mapping: feature -> group name
feat_to_group = {}
for g_name, feats in groups.items():
    for f in feats:
        feat_to_group[f] = g_name.replace('_', ' ')

# 2. Load Importance Data
df = pd.read_csv('task_data/models_optuna_tscv/feature_importance.csv')

# 3. Calculate Weight and Cumulative Weight
total_gain = df['importance'].sum()
df['Weight (%)'] = (df['importance'] / total_gain) * 100
df['Cumulative (%)'] = df['Weight (%)'].cumsum()

# 4. Cleanup Names and Assign Groups
def clean_name(name):
    clean = name.replace('_', ' ').title()
    clean = clean.replace('Roa', 'ROA').replace('Cfo', 'CFO').replace('Fcf', 'FCF').replace('Lt', 'LT').replace('St', 'ST').replace('Nwc', 'NWC').replace('Roe', 'ROE')
    return clean

df['Feature Name'] = df['feature'].apply(clean_name)
df['Group'] = df['feature'].apply(lambda x: feat_to_group.get(x, "Other"))

# LaTeX Generation
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\small")
print(r"\caption{Top 10 Feature Importance: Thematic Groups and Signal Contribution}")
print(r"\label{tab:feature_importance_top10}")
print(r"\begin{tabular}{llccc}")
print(r"\toprule")
print(r"\textbf{Feature} & \textbf{Thematic Group} & \textbf{Gain} & \textbf{Rel. Weight} & \textbf{Cumul.} \\")
print(r"\midrule")

# Top 10 for focused analysis
for _, row in df.head(10).iterrows():
    print(f"{row['Feature Name']} & {row['Group']} & {row['importance']:.2f} & {row['Weight (%)']:.1f}\% & {row['Cumulative (%)']:.1f}\% \\")

print(r"\bottomrule")
print(r"\multicolumn{5}{l}{\footnotesize \textit{Note: Rel. Weight is the feature's \% contribution to the total model gain.}} \\")
print(r"\end{tabular}")
print(r"\end{table}")
