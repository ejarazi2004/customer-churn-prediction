import os
import kaggle
import zipfile

#Dataset downlaod path
RAW_DATA_DIR = "data/raw/"

os.makedirs(RAW_DATA_DIR, exist_ok=True)

def download_data(kaggle_dataset: str):
    dataset_name = kaggle_dataset.split("/")[-1]
    zip_file_dir = os.path.join(RAW_DATA_DIR, f"{dataset_name}.zip")
    
    kaggle_api_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_api_path):
        print("Kaggle API key not found.")
        return
    
    if os.path.exists(zip_file_dir):
        print("Dataset already downloaded.")
    else:
        kaggle.api.dataset_download_files(kaggle_dataset, path=RAW_DATA_DIR, unzip=False)
        
    #Extract the zip file
    with zipfile.ZipFile(zip_file_dir, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
        
    #Remove the .zip file after extraxtion
    os.remove(zip_file_dir)
    

if __name__ == "__main__":
    dataset_name = "gauravtopre/bank-customer-churn-dataset"
    download_data(dataset_name)