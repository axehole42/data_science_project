import pandas as pd
import dtale
import sys
import webbrowser
import threading
import time

# Default file path
DEFAULT_FILE = 'task_data/itaiif_compustat_data_24112025.csv'

def browse_data(file_path=DEFAULT_FILE):
    """
    Loads data and opens it in a D-Tale browser window for interactive analysis.
    """
    try:
        print(f"Reading {file_path}...")
        df = pd.read_csv(file_path)
        
        print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print("Starting D-Tale server...")
        
        # Start D-Tale instance
        d = dtale.show(df, subprocess=False)
        
        print(f"\n========================================================")
        print(f" D-Tale is running at: {d._main_url}")
        print(f"========================================================")
        print("Press Ctrl+C to stop the server.")
        
        # Open in default web browser
        try:
            webbrowser.open_new_tab(d._main_url)
        except:
            print(f"Could not open browser automatically. Please visit: {d._main_url}")

        # Keep the script running so the server stays up
        while True:
            time.sleep(1)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except KeyboardInterrupt:
        print("\nStopping D-Tale server...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Allow user to pass a specific file path as an argument
    target_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    browse_data(target_file)