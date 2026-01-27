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

# Exclude known absolute values
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
    
    # Use raw name, escaped for LaTeX
    raw_name = col.replace('_', '\\_')
    
    stats.append({
        'Feature': raw_name,
        'N': f"{n:,}",
        'Miss': f"{missing}",
        'Mean': f"{mean:.3f}" if abs(mean) < 1000 else f"{mean:.2e}",
        'Std': f"{std:.3f}" if abs(std) < 1000 else f"{std:.2e}",
        'P99': f"{p99:.3f}" if abs(p99) < 1000 else f"{p99:.2e}"
    })

stats_df = pd.DataFrame(stats)

# Generate LaTeX
print("\begin{table}[h]")
print("\centering")
print("\scriptsize")
print("\caption{Descriptive Statistics of Engineered Ratios (N=62,754)}")
print("\label{tab:ratio_stats}")
print("\begin{tabular}{lrrrrr}")
print("\\toprule")
# Updated column order
print("\\textbf{Feature} & \\textbf{N} & \\textbf{Miss} & \\textbf{Mean} & \\textbf{Std} & \\textbf{P99} \\")
print("\\midrule")

for _, row in stats_df.iterrows():
    print(f"{row['Feature']} & {row['N']} & {row['Miss']} & {row['Mean']} & {row['Std']} & {row['P99']} \\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
