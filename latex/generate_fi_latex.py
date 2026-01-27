import pandas as pd

# Load importance data
df = pd.read_csv('task_data/models_optuna_tscv/feature_importance.csv')

# Formatting names for LaTeX
def clean_name(name):
    clean = name.replace('_', ' ').title()
    clean = clean.replace('Roa', 'ROA').replace('Cfo', 'CFO').replace('Fcf', 'FCF').replace('Lt', 'LT').replace('St', 'ST').replace('Nwc', 'NWC')
    return clean

df['Feature Name'] = df['feature'].apply(clean_name)

# LaTeX Generation
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\small")
print(r"\caption{Feature Importance (Gain) for the Main ROA Improvement Model}")
print(r"\label{tab:feature_importance_main}")
print(r"\begin{tabular}{lr}")
print(r"\toprule")
print(r"\textbf{Feature} & \textbf{Gain Score} \\")
print(r"\midrule")

# Print top 20 for a clean table, or all if preferred. 
# Usually Top 15-20 is best for reports.
for _, row in df.head(20).iterrows():
    print(f"{row['Feature Name']} & {row['importance']:.2f} \\")

print(r"\bottomrule")
print(r"\multicolumn{2}{l}{\footnotesize \textit{Note: Gain represents the relative contribution of each feature to the model accuracy.}} \\")
print(r"\end{tabular}")
print(r"\end{table}")

