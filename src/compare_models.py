# src/compare_models.py  (UPDATED: saves normalized confusion matrices)
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from tensorflow.keras.models import load_model

# Paths (update if needed)
DATA_IN = "../data/diabetes_with_genetic_pca.csv"
RF_MODEL = "../models/rf_pca_model.pkl"
RF_SCALER = "../models/rf_pca_scaler.pkl"
XGB_MODEL = "../models/xgb_model.pkl"
ANN_MODEL = "../models/ann_model.h5"
ANN_SCALER = "../models/ann_scaler.pkl"
OUTPUT_DIR = "../results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_IN)
    drop_cols = ["sample_id", "Outcome"] if "sample_id" in df.columns else ["Outcome"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Outcome"]
    return X, y

def plot_confusion_matrix(y_true, y_pred, out_path, title="Confusion Matrix", normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        # normalize by true labels (rows)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        data = cm_norm
        fmt = ".2f"
    else:
        data = cm
        fmt = "d"

    plt.figure(figsize=(5,4))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {out_path}")

def eval_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Load models & scalers
    rf = joblib.load(RF_MODEL)
    xgb = joblib.load(XGB_MODEL)
    ann = load_model(ANN_MODEL)
    rf_scaler = joblib.load(RF_SCALER)
    ann_scaler = joblib.load(ANN_SCALER)

    # Scale for RF and ANN
    X_test_rf = rf_scaler.transform(X_test)
    X_test_ann = ann_scaler.transform(X_test)

    # Predict probabilities
    y_rf_proba = rf.predict_proba(X_test_rf)[:,1]
    y_xgb_proba = xgb.predict_proba(X_test)[:,1]
    y_ann_proba = ann.predict(X_test_ann).ravel()

    models = {
        "RandomForest": {"proba": y_rf_proba, "preds": (y_rf_proba >= 0.5).astype(int)},
        "XGBoost": {"proba": y_xgb_proba, "preds": (y_xgb_proba >= 0.5).astype(int)},
        "ANN": {"proba": y_ann_proba, "preds": (y_ann_proba >= 0.5).astype(int)}
    }

    stats = []
    plt.figure(figsize=(8,6))
    for name, info in models.items():
        probs = info["proba"]
        preds = info["preds"]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        stats.append([name, acc, f1, roc, precision, recall])

        # ROC
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc:.3f})")

        # confusion matrices - raw and normalized
        out_raw = os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.lower()}.png")
        out_norm = os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.lower()}_norm.png")
        plot_confusion_matrix(y_test, preds, out_raw, title=f"{name} Confusion Matrix", normalize=False)
        plot_confusion_matrix(y_test, preds, out_norm, title=f"{name} Confusion Matrix", normalize=True)

    # ROC figure
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    roc_out = os.path.join(OUTPUT_DIR, "roc_comparison.png")
    plt.tight_layout()
    plt.savefig(roc_out, dpi=300)
    plt.close()
    print(f"Saved ROC comparison: {roc_out}")

    df_stats = pd.DataFrame(stats, columns=["Model","Accuracy","F1","ROC_AUC","Precision","Recall"])
    stats_out = os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv")
    df_stats.to_csv(stats_out, index=False)
    print(f"Saved model metrics: {stats_out}")
    print("Done.")

if __name__ == "__main__":
    eval_models()
