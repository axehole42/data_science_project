import pandas as pd
import tkinter as tk
from tkinter import ttk
import numpy as np

FILE_PATH = 'task_data/cleaned_data.parquet'

def show_stats():
    print("Loading data for Summary Statistics...")
    try:
        df = pd.read_parquet(FILE_PATH)
    except FileNotFoundError:
        print(f"File not found: {FILE_PATH}")
        return

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate Statistics (similar to STATA 'summarize, detail')
    stats = numeric_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats['skew'] = numeric_df.skew()
    stats['kurt'] = numeric_df.kurt()
    
    # Reset index to make 'variable' a column
    stats.reset_index(inplace=True)
    stats.rename(columns={'index': 'Variable'}, inplace=True)
    
    # Rounding
    stats = stats.round(3)

    # --- GUI Construction ---
    root = tk.Tk()
    root.title("Summary Statistics (STATA-like)")
    root.geometry("1200x600")

    # Treeview (Table)
    cols = list(stats.columns)
    tree = ttk.Treeview(root, columns=cols, show='headings')
    
    # Scrollbars
    vsb = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(root, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    
    # Layout
    tree.grid(column=0, row=0, sticky='nsew')
    vsb.grid(column=1, row=0, sticky='ns')
    hsb.grid(column=0, row=1, sticky='ew')
    
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # Headings
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, minwidth=50, width=100)

    # Data
    for index, row in stats.iterrows():
        tree.insert("", "end", values=list(row))

    print("Opening GUI...")
    root.mainloop()

if __name__ == "__main__":
    show_stats()
