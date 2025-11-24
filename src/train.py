import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

INPUT = "../data/diabetes_with_genetic_pca.csv"
MODEL_OUT = "../models/rf_pca_model.pkl"
SCALER_OUT = "../models/rf_pca_scaler.pkl"

def train_rf():
    print("Loading dataset...")
    df = pd.read_csv(INPUT)

    # Outcome as target
    y = df["Outcome"]

    # Drop outcome + sample_id if exists
    drop_cols = ["Outcome", "sample_id"]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Random Forest on combined clinical + PCA genetic features...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Random Forest (with PCA Genetic Features) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model & scaler
    joblib.dump(model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)

    print("\nSaved model to:", MODEL_OUT)
    print("Saved scaler to:", SCALER_OUT)

if __name__ == "__main__":
    train_rf()
