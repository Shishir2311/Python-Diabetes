# src/train_xgboost.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb

INPUT = "../data/diabetes_with_genetic_pca.csv"
MODEL_OUT = "../models/xgb_model.pkl"

def train_xgb():
    print("Loading dataset...")
    df = pd.read_csv(INPUT)

    # target assumed to be 'Outcome'
    if "Outcome" not in df.columns:
        raise ValueError("No 'Outcome' column found in dataset.")

    # Prepare features - drop sample_id, Outcome
    drop_cols = ["sample_id", "Outcome"] if "sample_id" in df.columns else ["Outcome"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Outcome"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Training XGBoost...")
    model = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_OUT)
    print("Saved XGBoost model to", MODEL_OUT)

if __name__ == "__main__":
    train_xgb()
