import os
import sys
import json
import joblib
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from scripts.split_data import load_split_data
from scripts.preprocess_data import build_preproc_pipeline

def tune_lightgbm(data_path, n_iter: int = 50, cv: int = 5):
    #Load split data
    X_train, _, y_train, _ = load_split_data(data_path)
    
    #Building preprocessing pipeline
    preprocessor = build_preproc_pipeline()
    
    lgbm = LGBMClassifier(random_state=42)
    
    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", lgbm)
    ])
    
    #Param grid to search
    param_grid = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [3, 5, 7, 10],
        "classifier__num_leaves": [20, 31, 50],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__reg_alpha": [0.0, 0.1, 1.0],
        "classifier__reg_lambda": [0.0, 0.1, 1.0]
    }
    
    search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_grid,
        scoring=make_scorer(f1_score),
        n_iter=n_iter,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    print("Best Score: ", search.best_score_)
    print("Best Params:", search.best_params_)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models/tuned", exist_ok=True)

    model_path = f"models/tuned/lightgbm_tuned_{timestamp}.joblib"
    joblib.dump(search.best_estimator_, model_path)
    print(f" Best model saved to: {model_path}")

    # Save results
    os.makedirs("results/tuning", exist_ok=True)
    results_path = f"results/tuning/lightgbm_search_results_{timestamp}.json"

    search_results = {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_
    }
    with open(results_path, "w") as f:
        json.dump(search_results, f, indent=2)

    print(f" Tuning results saved to: {results_path}")
    
    return model_path, results_path


if __name__ == "__main__":
    DATA_PATH = "data/raw/Bank Customer Churn Prediction.csv"
    tune_lightgbm(DATA_PATH)

