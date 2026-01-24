from pathlib import Path
import pandas as pd
import dtale
import time
import os
import webbrowser

# base directory of this script
BASE_DIR = Path(__file__).resolve().parent

# directory where input and output data are stored
DATA_DIR = BASE_DIR / "task_data"

# input file containing the features
INFILE = DATA_DIR / "features.parquet"

# output file with computed column statistics
OUTFILE = DATA_DIR / "col_stats.parquet"


def get_file_mtime(path: Path) -> float:
    # return last modification time of a file
    return os.stat(path).st_mtime


def compute_col_stats(df: pd.DataFrame) -> pd.DataFrame:
    # select only numeric columns for statistical calculations
    num = df.select_dtypes(include="number")

    # compute basic statistics per numeric column
    col_stats = pd.DataFrame(
        {
            "median": num.median(axis=0, skipna=True),
            "mean": num.mean(axis=0, skipna=True),
            "std": num.std(axis=0, skipna=True),  # sample standard deviation (ddof=1)
        }
    )

    # count missing values for all columns,
    # then align them with the numeric columns used above
    n_missing = df.isna().sum(axis=0).reindex(col_stats.index)
    col_stats["n_missing"] = n_missing

    # percentage of missing values relative to total number of rows
    col_stats["pct_missing"] = (n_missing / len(df)) * 100.0

    # name the index for clarity and return a clean DataFrame
    col_stats.index.name = "column"
    return col_stats.reset_index()


def write_and_serve(current_instance=None, open_browser=False):
    # load input data
    print(f"Reading {INFILE}...")
    df = pd.read_parquet(INFILE)
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # compute column-level statistics
    col_stats = compute_col_stats(df)

    # write results to disk
    col_stats.to_parquet(OUTFILE, index=False)
    print(f"Wrote: {OUTFILE}")

    # stop an existing dtale instance if one is running
    if current_instance:
        try:
            current_instance.kill()
        except Exception:
            pass

    # start a new dtale server to explore the results interactively
    d = dtale.show(
        col_stats,
        host="127.0.0.1",
        subprocess=True,
    )

    # retrieve the local URL of the dtale server
    url = d._main_url
    print(f"Server active at: {url}")

    # optionally open the browser automatically
    if open_browser:
        print(f"Opening browser at {url}...")
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            print(f"Could not open browser automatically. Visit: {url}")

    return d


if __name__ == "__main__":
    # basic instructions printed to the terminal
    print("========================================================")
    print(" COLUMN-STATS BROWSER (AUTO-RELOAD)")
    print("========================================================")
    print(f"Watching input:  {INFILE}")
    print(f"Writing output:  {OUTFILE}")
    print("1. Keep this terminal open.")
    print("2. If the input file changes, the statistics are recomputed automatically.")
    print("========================================================")

    # wait until the input file exists
    if not INFILE.exists():
        print(f"Waiting for {INFILE} to be created...")
        while not INFILE.exists():
            time.sleep(1)

    # initial computation and server start
    dtale_instance = write_and_serve(open_browser=True)
    last_mtime = get_file_mtime(INFILE)

    try:
        # continuously monitor the input file for updates
        while True:
            time.sleep(2)
            try:
                current_mtime = get_file_mtime(INFILE)
                if current_mtime > last_mtime:
                    print("\n[!] Data update detected. Recomputing + reloading...")
                    time.sleep(1)
                    dtale_instance = write_and_serve(
                        current_instance=dtale_instance,
                        open_browser=False,
                    )
                    last_mtime = current_mtime
                    print(f"[✓] Reloaded. Active URL: {dtale_instance._main_url}")
            except OSError:
                # ignore temporary file access issues
                pass

    except KeyboardInterrupt:
        # clean shutdown on manual interruption
        print("\nStopping server...")
        if dtale_instance:
            dtale_instance.kill()
        raise SystemExit(0)

