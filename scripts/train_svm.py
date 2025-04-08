import os
import json
import joblib
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

from scripts.split_data import load_split_data
from scripts.preprocess_data import build_preproc_pipeline

#To avoid import errors

def train_svm(data_path):
    #Loading the split data
    X_train, _, y_train, _ = load_split_data(data_path)
    
    #Building the preprocessing pipeline
    preprocessor = build_preproc_pipeline()
    
    #SVM classifier with RBF kernel and probability=True for ROC AUC
    #Full pipeline
    svm_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    )
    )
    ])
    
    #Scoring metrics
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score)
    }
    
    #Cross-Validation 5 folds
    cv_results = cross_validate(
        svm_pipeline,
        X_train,
        y_train,
        scoring=scoring,
        cv=5,
        return_train_score=False
    )
    
    print("SVM Cross-Validation Results:")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        print(f"{metric.upper()} scores: {scores}")
        print(f"Mean {metric.upper()}: {scores.mean():.4f}")
        print("-" * 40)
        
    svm_pipeline.fit(X_train, y_train)
    
    #Saving pipeline and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)
    
    model_path = os.path.join("models", f"svm_pipeline_{timestamp}.joblib")
    joblib.dump(svm_pipeline, model_path)
    print(f"Model pipeline saved to: {model_path}")
    
    #Saving CV results
    cv_summary = {
        metric: {
            "scores": cv_results[f"test_{metric}"].tolist(),
            "mean": float(cv_results[f"test_{metric}"].mean())
        } for metric in scoring
    }
    
    cv_path = os.path.join("results/evaluation", f"svm_cv_results_{timestamp}.json")
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"CV results saved to {cv_path}")
    
    return model_path, cv_path

if __name__ == "__main__":
    DATA_PATH = "data/raw/Bank Customer Churn Prediction.csv"
    train_svm(DATA_PATH)