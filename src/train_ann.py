# src/train_ann.py
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

INPUT = "../data/diabetes_with_genetic_pca.csv"
MODEL_OUT = "../models/ann_model.h5"
SCALER_OUT = "../models/ann_scaler.pkl"

def train_ann():
    print("Loading dataset...")
    df = pd.read_csv(INPUT)
    if "Outcome" not in df.columns:
        raise ValueError("No 'Outcome' column found in dataset.")

    drop_cols = ["sample_id", "Outcome"] if "sample_id" in df.columns else ["Outcome"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Outcome"].astype(int)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_OUT)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    # Build ANN
    n_in = X_train.shape[1]
    model = Sequential([
        Dense(128, activation="relu", input_shape=(n_in,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_loss")

    history = model.fit(X_train, y_train, validation_split=0.15, epochs=200, batch_size=32, callbacks=[es, mc], verbose=2)

    # Evaluate
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("ANN training complete. Model saved to", MODEL_OUT)

if __name__ == "__main__":
    # Limit TensorFlow logging if verbose
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train_ann()
