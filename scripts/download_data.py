import os
import kaggle
import zipfile

#Dataset downlaod path
RAW_DATA_DIR = "data/raw/"
ZIP_FILE_DIR = "data/raw/dataset.zip"

os.makedirs(RAW_DATA_DIR, exist_ok=True)

def download_data(kaggle_dataset: str):
    kaggle_api_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_api_path):
        print("Kaggle API key not found.")
        return
    
    if os.path.exists(ZIP_FILE_DIR):
        print("Dataset already downloaded.")
    else:
        kaggle.api.dataset_download_files(kaggle_dataset, path=RAW_DATA_DIR, unzip=False)
        
    #Extract the zip file
    with zipfile.ZipFile(ZIP_FILE_DIR, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
        
    #Remove the .zip file after extraxtion
    os.remove(ZIP_FILE_DIR)
    

if __name__ == "__main__":
    dataset_name = "gauravtopre/bank-customer-churn-dataset"
    download_data(dataset_name)