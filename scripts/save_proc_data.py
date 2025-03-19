import os 
import pandas as pd

PROCESSED_DATA_DIR = "data/processed/"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def save_datframe(df: pd.DataFrame, filename: str):
    if not filename.endswith(".parquet"):
        raise ValueError("File name must be of .parquet format.")
    
    file_path = os.path.join(PROCESSED_DATA_DIR, filename)
    
    df.to_parquet(file_path, index=False)