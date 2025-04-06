import os
import sys
import joblib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

#To avoid import errors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.split_data import load_split_data
from scripts.preprocess_data import build_preproc_pipeline

def train_random_forest(data_path):
    
    #Load the split train/test data
    X_train, X_test, y_train, y_test = load_split_data(data_path)
    
    #Building the preprocessing pipeline
    preprocessor = build_preproc_pipeline()
    
    #Defining the model pipeline
    rf_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        ))
    ])
    
    #Defining evaluation metrics
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score)
    }
    
    #Performing 5-fold cross-validation
    cv_results = cross_validate(
        rf_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )
    
    #Displaying results
    print("Random Forest Cross-Validation Results:")
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"{metric.upper()} scores: {scores}")
        print(f"Mean {metric.upper()}: {scores.mean():.4f}")
        print("-" * 40)
        
    #Fiting the pipeline on the full training data
    rf_pipeline.fit(X_train, y_train)
    
    #Saving pipeline and CV results
    timestamp = datetime.now().strftime("%Y%m%d_%H&M%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)
    
    model_path = os.path.join("models", f"rf_pipeline_{timestamp}.joblib")
    joblib.dump(rf_pipeline, model_path)
    print(f"Model pipeline saved to: {model_path}")
    
    #Saving CV scores to JSON
    cv_summary = {
        metric: {
            "scores": cv_results[f"test_{metric}"].tolist(),
            "mean": float(cv_results[f"test_{metric}"].mean())
        }for metric in scoring
    }
    
    cv_path = os.path.join("results/evaluation", f"rf_cv_results_{timestamp}.json")
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"Cv results saved to {cv_path}")
    
    return model_path, cv_path


if __name__ == "__main__":
    DATA_PATH = "data/raw/Bank Customer Churn Prediction.csv"
    train_random_forest(DATA_PATH)