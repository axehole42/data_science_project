import pandas as pd

df = pd.read_parquet('task_data/features.parquet')
cols = ['equity_to_assets', 'roe']

for col in cols:
    missing_count = df[col].isna().sum()
    total = len(df)
    pct = (missing_count / total) * 100
    tgt_miss_mean = df.loc[df[col].isna(), 'target'].mean()
    
    print(f"Feature: {col}")
    print(f"  Missing Count: {missing_count}")
    print(f"  Total Rows:    {total}")
    print(f"  Percent Miss:  {pct:.4f}%")
    print(f"  Target(Miss):  {tgt_miss_mean}")
    print("-" * 20)
