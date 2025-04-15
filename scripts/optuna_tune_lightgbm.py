import os 
import json
import sys
import optuna
import joblib
from datetime import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from scripts.split_data import load_split_data
from scripts.preprocess_data import build_preproc_pipeline

def objective(trial):
    X_train, _, y_train, _ = load_split_data("data/raw/Bank Customer Churn Prediction.csv")
    
    preprocessor = build_preproc_pipeline()
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 15, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": 42,
    }
    
    lgbm_model = LGBMClassifier(**params)
    
    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", lgbm_model)
    ])
    
    scores = cross_val_score(
        full_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=make_scorer(f1_score),
        n_jobs=-1
    )
    
    return scores.mean()

def run_study(n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    X_train, _, y_train, _ = load_split_data("data/raw/Bank Customer Churn Prediction.csv")
    
    preprocessor = build_preproc_pipeline()
    
    best_lgbm = LGBMClassifier(**study.best_params)
    
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", best_lgbm)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models/tuned", exist_ok=True)
    model_path = f"models/tuned/lightgbm_optuna_{timestamp}.joblib"
    joblib.dump(pipeline, model_path)
    print(f" Best model saved to: {model_path}")

    # Save results
    os.makedirs("results/tuning", exist_ok=True)
    results_path = f"results/tuning/lightgbm_optuna_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_score": study.best_value,
            "best_params": study.best_params
        }, f, indent=2)
    print(f" Results saved to: {results_path}")

    return model_path, results_path

if __name__ == "__main__":
    run_study(n_trials=50)