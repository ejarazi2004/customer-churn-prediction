import pandas as pd
from sklearn.model_selection import train_test_split

def load_split_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split_data("../data/raw/Bank Customer Churn Prediction.csv")
    

