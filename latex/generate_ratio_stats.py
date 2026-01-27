import pandas as pd
import json

# Load data and feature definitions
df = pd.read_parquet('task_data/features.parquet')
with open('task_data/feature_groups.json', 'r') as f:
    groups = json.load(f)

# Flatten the list of features
engineered_features = []
for group_features in groups.values():
    engineered_features.extend(group_features)

# Exclude known absolute values or non-ratio metrics if any slip through
# 'total_accruals' is an absolute magnitude ($), so we remove it.
# 'log_assets' and 'log_book_equity' are retained as standard "Size" proxies in ratio analysis.
exclude_list = {'total_accruals'}
features_to_analyze = [f for f in engineered_features if f not in exclude_list and f in df.columns]

features_to_analyze.sort()

# Calculate stats
stats = []
for col in features_to_analyze:
    series = df[col]
    
    n = series.count()
    missing = series.isna().sum()
    mean = series.mean()
    std = series.std()
    p99 = series.quantile(0.99)
    
    # Format name: "debt_to_assets" -> "Debt To Assets"
    clean_name = col.replace('_', ' ').title()
    # Manual overrides for acronyms
    clean_name = clean_name.replace('Roa', 'ROA').replace('Roe', 'ROE').replace('Cfo', 'CFO').replace('Fcf', 'FCF').replace('Ebitda', 'EBITDA').replace('Lt', 'LT').replace('St', 'ST').replace('Nwc', 'NWC').replace('Capx', 'CapEx')
    
    # Formatting numbers to avoid massive scientific notation strings in LaTeX if not needed
    # Using general format or scientific if very large/small
    
    stats.append({
        'Feature': clean_name,
        'N': f"{n:,}",
        'Mean': f"{mean:.3f}" if abs(mean) < 1000 else f"{mean:.2e}",
        'Std': f"{std:.3f}" if abs(std) < 1000 else f"{std:.2e}",
        'Miss': f"{missing}",
        'P99': f"{p99:.3f}" if abs(p99) < 1000 else f"{p99:.2e}"
    })

stats_df = pd.DataFrame(stats)

# Generate LaTeX
print("\\begin{table}[h]")
print("\\centering")
print("\\scriptsize")
print("\\caption{Descriptive Statistics of Engineered Ratios (N=62,754)}")
print("\\label{tab:ratio_stats}")
print("\\begin{tabular}{lrrrrr}")
print("\\toprule")
print("\\textbf{Feature} & \\textbf{N} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Miss} & \\textbf{P99} \\")
print("\\midrule")

for _, row in stats_df.iterrows():
    # Escape underscores just in case, though we replaced them
    feat = row['Feature'].replace('&', '\\&')
    print(f"{feat} & {row['N']} & {row['Mean']} & {row['Std']} & {row['Miss']} & {row['P99']} \\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
