import os
import json
import joblib
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

from scripts.split_data import load_split_data
from scripts.preprocess_data import build_preproc_pipeline

def train_lightgbm(data_path):
    #Loading and splitng the data
    X_train, _, y_train, _ = load_split_data(data_path)
    
    #Building the preprocessing
    preprocessor = build_preproc_pipeline()
    
    #Full pipeline
    lightgbm_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ))
    ])
    
    #Scoring metrics
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score)
    }
    
    #Cross-validation 5 folds
    cv_results = cross_validate(
        lightgbm_pipeline,
        X_train,
        y_train,
        scoring=scoring,
        cv=5,
        return_train_score=False
    )
    
    print("Lightgbm Cross-Validation Results:")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        print(f"{metric.upper()} scores: {scores}")
        print(f"Mean {metric.upper()}: {scores.mean():.4f}")
        print("-" * 40)
        
    lightgbm_pipeline.fit(X_train, y_train)
    
    #Saving pipeline and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)
    
    model_path = os.path.join("models", f"lightgbm_pipeline_{timestamp}.joblib")
    joblib.dump(lightgbm_pipeline, model_path)
    print(f"Model pipeline saved to: {model_path}")
    
    #Saving CV results
    cv_summary = {
        metric: {
            "scores": cv_results[f"test_{metric}"].tolist(),
            "mean": float(cv_results[f"test_{metric}"].mean())
        } for metric in scoring
    }
    
    cv_path = os.path.join("results/evaluation", f"lightgbm__cv_results_{timestamp}.json")
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"CV results saved to {cv_path}")
    
    return model_path, cv_path

if __name__ == "__main__":
    DATA_PATH = "data/raw/Bank Customer Churn Prediction.csv"
    train_lightgbm(DATA_PATH)