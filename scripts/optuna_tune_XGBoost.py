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

from xgboost import XGBClassifier

from scripts.split_data import load_split_data
from scripts.preprocess_data import build_preproc_pipeline

def objective(trial):
    X_train, _, y_train, _ = load_split_data("data/raw/Bank Customer Churn Prediction.csv")
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
    }
    
    model = XGBClassifier(**params)
    
    preprocessor = build_preproc_pipeline()
    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])
    
    score = cross_val_score(
        full_pipeline, X_train, y_train,
        scoring=make_scorer(f1_score),
        cv=5
    ).mean()
    
    return score

def tune_xgboost():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000)

    best_score = study.best_value
    best_params = study.best_params
    print(f"Best score: {best_score:.4f}")
    print(f"Best params: {best_params}")
    
    X_train, _, y_train, _ = load_split_data("data/raw/Bank Customer Churn Prediction.csv")
    
    model = XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )

    pipeline = Pipeline([
        ("preprocessing", build_preproc_pipeline()),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    # Step 3: Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models/tuned", exist_ok=True)
    model_path = f"models/tuned/xgboost_optuna_{timestamp}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Best model saved to: {model_path}")
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/tuning", exist_ok=True)
    result_path = f"results/tuning/xgboost_optuna_results_{timestamp}.json"

    with open(result_path, "w") as f:
        json.dump({
            "best_score": best_score,
            "best_params": best_params
        }, f, indent=2)

    print(f"Best score: {best_score:.4f}")
    print(f"Best params saved to: {result_path}")

    return best_params, best_score


if __name__ == "__main__":
    DATA_PATH = "data/raw/Bank Customer Churn Prediction.csv"
    tune_xgboost()