import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from preprocess import load_data, preprocess_data

def evaluate_model():
    # Load raw data
    df = load_data()

    # Preprocess with same scaler logic
    X_scaled, y, scaler = preprocess_data(df)

    # Load trained model
    model = joblib.load("../models/diabetes_model.pkl")

    # Predict
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    print("\n=== MODEL EVALUATION ===")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred))
    print("ROC AUC:", roc_auc_score(y, y_proba))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
