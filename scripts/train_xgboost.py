import os
import json
import joblib
import sys
from datetime import datetime

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from scripts.split_data import load_split_data
from scripts.preprocess_data import build_preproc_pipeline

def train_xgboost(data_path):
    #Loading and splitting data
    X_train, X_test, y_train, y_test = load_split_data(data_path)
    
    #Building preprocessing pipeline
    preprocessor = build_preproc_pipeline()
    
    #Building full pipeline
    xgb_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ))
    ])
    
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score)
    }

    #Cross-validation 5 folds
    cv_results = cross_validate(
        xgb_pipeline,
        X_train,
        y_train,
        scoring = scoring,
        cv=5,
        return_train_score=False
    )
    
    print("XGBoost Cross-Validation Results:")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        print(f"{metric.upper()} scores: {scores}")
        print(f"Mean {metric.upper()}: {scores.mean():.4f}")
        print("-" * 40)
        
    #Fiting the pipeline on the full training data
    xgb_pipeline.fit(X_train, y_train)
    
    #Saving pipeline and CV results
    timestamp = datetime.now().strftime("%Y%m%d_%H&M%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)
    
    model_path = os.path.join("models", f"xgb_pipeline_{timestamp}.joblib")
    joblib.dump(xgb_pipeline, model_path)
    print(f"Model pipeline saved to: {model_path}")
    
    #Saving CV scores to JSON
    cv_summary = {
        metric: {
            "scores": cv_results[f"test_{metric}"].tolist(),
            "mean": float(cv_results[f"test_{metric}"].mean())
        }for metric in scoring
    }
    
    cv_path = os.path.join("results/evaluation", f"xgb_cv_results_{timestamp}.json")
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"Cv results saved to {cv_path}")
    
    return model_path, cv_path


if __name__ == "__main__":
    DATA_PATH = "data/raw/Bank Customer Churn Prediction.csv"
    train_xgboost(DATA_PATH)
