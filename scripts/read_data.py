import os
import kaggle
import zipfile
import pandas as pd

RAW_DATA_DIR = "data/raw/"

def load_dataset(filename: str):
    file_path = os.path.join(RAW_DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {filename} was not found in {RAW_DARA_DIR}.")
    
    if filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif filename.endswith(".parquet"):
        df=pd.read_parquet
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")
    
    return df 