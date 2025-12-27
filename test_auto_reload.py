import pandas as pd
import time

FILE_PATH = 'task_data/itaiif_compustat_data_24112025.csv'

def test_update():
    print(f"Reading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    
    # Store original value to revert later if needed
    original_name = df.at[0, 'conm']
    print(f"Original Company Name at row 0: {original_name}")
    
    # Modify
    new_name = "!!! AUTO-RELOAD WORKED !!!"
    df.at[0, 'conm'] = new_name
    
    print(f"Changing to: {new_name}")
    print("Saving file...")
    df.to_csv(FILE_PATH, index=False)
    
    print("Done! check your D-Tale browser window.")
    print("(Don't forget to refresh the page if D-Tale asks you to!)")

if __name__ == "__main__":
    test_update()
