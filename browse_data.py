import pandas as pd
import dtale
import sys
import os
import time
import webbrowser

# Import Labels
try:
    from task_data.var_labels import VAR_LABELS
except ImportError:
    VAR_LABELS = {}

# Default file path
DEFAULT_FILE = 'task_data/cleaned_data.parquet'

def get_file_mtime(path):
    """Returns the last modification time of the file."""
    return os.stat(path).st_mtime

def load_and_serve(path, current_instance=None, open_browser=False):
    """
    Loads the Parquet file and starts D-Tale with clean headers + descriptions.
    """
    print(f"Reading {path}...")
    try:
        # Use read_parquet for better performance and type preservation
        df = pd.read_parquet(path)
        print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        
        if current_instance:
            try:
                current_instance.kill()
            except:
                pass

        # Pass VAR_LABELS directly to D-Tale
        # This keeps headers clean ('at') but adds metadata descriptions
        d = dtale.show(
            df, 
            host='127.0.0.1', 
            subprocess=True, 
            column_descriptions=VAR_LABELS
        )
        
        url = d._main_url
        print(f"Server active at: {url}")
        
        if open_browser:
            print(f"Opening browser at {url}...")
            try:
                webbrowser.open_new_tab(url)
            except:
                print(f"Could not open browser automatically. Visit: {url}")
        
        return d

    except Exception as e:
        print(f"Error loading data: {e}")
        return current_instance

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    
    # Wait for the file to be created if it doesn't exist yet
    if not os.path.exists(target_file):
        print(f"Waiting for {target_file} to be created by data_cleanup.py...")
        while not os.path.exists(target_file):
            time.sleep(1)

    print("========================================================")
    print(" AUTO-RELOAD DATA BROWSER (CLEAN HEADERS)")
    print("========================================================")
    print(f"Watching: {target_file}")
    print("1. Keep this terminal open.")
    print("2. Run data_cleanup.py to update the Parquet file.")
    print("3. The data will reload automatically.")
    print("========================================================")

    dtale_instance = load_and_serve(target_file, open_browser=True)
    last_mtime = get_file_mtime(target_file)

    try:
        while True:
            time.sleep(2)
            try:
                current_mtime = get_file_mtime(target_file)
                if current_mtime > last_mtime:
                    print("\n[!] Data update detected. Reloading...")
                    time.sleep(1)
                    dtale_instance = load_and_serve(target_file, current_instance=dtale_instance, open_browser=False)
                    last_mtime = current_mtime
                    print(f"[✓] Data reloaded. Active URL: {dtale_instance._main_url}")
            except OSError:
                pass
                
    except KeyboardInterrupt:
        print("\nStopping server...")
        if dtale_instance:
            dtale_instance.kill()
        sys.exit(0)
