import pandas as pd
import numpy as np

# Load the features data
df = pd.read_parquet('task_data/features.parquet')

# Columns to exclude (Identifiers and metadata)
exclude_cols = {'gvkey', 'fyear', 'datadate', 'conm', 'indfmt', 'datafmt', 'popsrc', 'consol', 'ismod', 'costat', 'dvpsp_f', 'mkvalt', 'prcc_f', 'csho'}
# Keep 'target' and all other numeric feature columns
feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

# Sort columns alphabetically
feature_cols.sort()

# Calculate statistics
stats = []
for col in feature_cols:
    series = df[col]
    
    # Calculate stats
    n = series.count()
    missing = series.isna().sum()
    mean = series.mean()
    std = series.std()
    p99 = series.quantile(0.99) # 99th percentile (High extreme)
    
    # Clean name for LaTeX
    clean_name = col.replace('_', ' ').title()
    if col == 'roa': clean_name = 'ROA'
    if col == 'target': clean_name = 'Target (Binary)'
    
    # Handle potentially infinite or massive values for display
    mean_str = f"{mean:.3f}" if abs(mean) < 10000 else f"{mean:.2e}"
    std_str = f"{std:.3f}" if abs(std) < 10000 else f"{std:.2e}"
    p99_str = f"{p99:.3f}" if abs(p99) < 10000 else f"{p99:.2e}"

    stats.append({
        'Feature': clean_name.replace('&', '\\&'), # Escape special LaTeX chars
        'N': f"{n:,}",
        'Mean': mean_str,
        'Std': std_str,
        'Miss': f"{missing}",
        'P99': p99_str
    })

# Convert to DataFrame
stats_df = pd.DataFrame(stats)

# Generate LaTeX
print("\\begin{table}[h]")
print("\\centering")
print("\\scriptsize")
print("\\caption{Descriptive Statistics of Engineered Features (N=62,754)}")
print("\\label{tab:feature_stats}")
print("\\begin{tabular}{lrrrrr}")
print("\\toprule")
print("\\textbf{Feature} & \\textbf{N} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Miss} & \\textbf{P99} \\")
print("\\midrule")

for _, row in stats_df.iterrows():
    # Construct the row string carefully
    line = f"{row['Feature']} & {row['N']} & {row['Mean']} & {row['Std']} & {row['Miss']} & {row['P99']} \\"
    print(line)

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")