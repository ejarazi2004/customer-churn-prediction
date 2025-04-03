import pandas as pd
from split_data import load_split_data
from preprocess_data import build_preproc_pipeline

def prepare_data(data_path):
    #Loading the split datasets
    X_train, X_test, y_train, y_test = load_split_data(data_path)
    
    #Pipline building
    preprocessor = build_preproc_pipeline()
    
    #Fitting the pipeline only in the train data
    preprocessor.fit(X_train)
    
    #Transforming both train and test
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor

#If script is run once through the terminal
if __name__ == "__main__":
    DATA_PATH = "../data/raw/Bank Customer Churn Prediction.csv"
    
    X_train_p, X_test_p, y_train, y_test, pipeline = prepare_data(DATA_PATH)
    
    print("Train transformed shape:", X_train_p.shape)
    print("Test transformed shape: ", X_test_p.shape)