from pathlib import Path
import pandas as pd
import dtale
import time
import os
import webbrowser

# your script is one folder ABOVE task_data
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "task_data"

INFILE = DATA_DIR / "features.parquet"
OUTFILE = DATA_DIR / "col_stats.parquet"


def get_file_mtime(path: Path) -> float:
    return os.stat(path).st_mtime


def compute_col_stats(df: pd.DataFrame) -> pd.DataFrame:
    # numeric columns only for median/mean/std
    num = df.select_dtypes(include="number")

    col_stats = pd.DataFrame(
        {
            "median": num.median(axis=0, skipna=True),
            "mean": num.mean(axis=0, skipna=True),
            "std": num.std(axis=0, skipna=True),  # ddof=1
        }
    )

    # missing across ALL columns (then align to numeric columns we report stats for)
    n_missing = df.isna().sum(axis=0).reindex(col_stats.index)
    col_stats["n_missing"] = n_missing

    # percentage missing (relative to number of rows)
    col_stats["pct_missing"] = (n_missing / len(df)) * 100.0

    col_stats.index.name = "column"
    return col_stats.reset_index()


def write_and_serve(current_instance=None, open_browser=False):
    print(f"Reading {INFILE}...")
    df = pd.read_parquet(INFILE)
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    col_stats = compute_col_stats(df)

    col_stats.to_parquet(OUTFILE, index=False)
    print(f"Wrote: {OUTFILE}")

    if current_instance:
        try:
            current_instance.kill()
        except Exception:
            pass

    d = dtale.show(
        col_stats,
        host="127.0.0.1",
        subprocess=True,
    )

    url = d._main_url
    print(f"Server active at: {url}")

    if open_browser:
        print(f"Opening browser at {url}...")
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            print(f"Could not open browser automatically. Visit: {url}")

    return d


if __name__ == "__main__":
    print("========================================================")
    print(" COLUMN-STATS BROWSER (AUTO-RELOAD)")
    print("========================================================")
    print(f"Watching input:  {INFILE}")
    print(f"Writing output:  {OUTFILE}")
    print("1. Keep this terminal open.")
    print("2. If you update the input file, the browser reloads automatically.")
    print("========================================================")

    if not INFILE.exists():
        print(f"Waiting for {INFILE} to be created...")
        while not INFILE.exists():
            time.sleep(1)

    dtale_instance = write_and_serve(open_browser=True)
    last_mtime = get_file_mtime(INFILE)

    try:
        while True:
            time.sleep(2)
            try:
                current_mtime = get_file_mtime(INFILE)
                if current_mtime > last_mtime:
                    print("\n[!] Data update detected. Recomputing + reloading...")
                    time.sleep(1)
                    dtale_instance = write_and_serve(current_instance=dtale_instance, open_browser=False)
                    last_mtime = current_mtime
                    print(f"[✓] Reloaded. Active URL: {dtale_instance._main_url}")
            except OSError:
                pass

    except KeyboardInterrupt:
        print("\nStopping server...")
        if dtale_instance:
            dtale_instance.kill()
        raise SystemExit(0)

