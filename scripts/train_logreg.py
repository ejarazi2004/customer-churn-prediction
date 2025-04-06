import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.split_data import load_split_data
from scripts.preprocess_data import build_preproc_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
from datetime import datetime

def train_logistic_regression(data_path):
    #Loading data and preprocessing pipeline
    X_train, _, y_train, _ = load_split_data(data_path)
    
    preproc_pipeline = build_preproc_pipeline()
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    #Defining full model pipeline
    logreg_pipeline = Pipeline(steps=[
        ('preprocessor', preproc_pipeline),
        ('classifier', clf)
    ])
    
    #Defining evaluation metrics for cross-validate
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    #Cross-validation (only on training set)
    cv_results = cross_validate(
        logreg_pipeline,
        X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    #Printing CV results to console
    print("Logistic Regression Cross-Validation Results:")
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"{metric.upper()} scores: {scores}")
        print(f"Mean {metric.upper()}: {scores.mean():.4f}")
        print("-" * 40)
        
    #Saving full model pipeline and CV results
    timestamp = datetime.now().strftime("%Y%m%d_%H&M%S")
    os.makedirs("models", exist_ok=True)
    
    #Saving trained model
    model_path = os.path.join("models", f"logreg_pipeline_{timestamp}.joblib")
    logreg_pipeline.fit(X_train, y_train)
    joblib.dump(logreg_pipeline, model_path)
    print(f"Model pipeline saved to: {model_path}")
    
    #Saving CV results
    cv_summary = {
        metric: {
            "scores": cv_results[f"test_{metric}"].tolist(),
            "mean": float(cv_results[f"test_{metric}"].mean())
        }for metric in scoring
    }
    
    cv_path = os.path.join("results/evaluation", f"logreg_cv_results_{timestamp}.json")
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"CV results saved to {cv_path}")
    
    return model_path, cv_path


if __name__ == "__main__":
    DATA_PATH = "data/raw/Bank Customer Churn Prediction.csv"
    train_logistic_regression(DATA_PATH)