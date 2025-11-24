# src/genetic_shap.py
import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

DATA_CSV = "../data/diabetes_with_genetic_pca.csv"
RF_MODEL = "../models/rf_pca_model.pkl"
XGB_MODEL = "../models/xgb_model.pkl"
OUT_SUM = "../models/shap_pca_summary.png"
OUT_BAR = "../models/shap_pca_bar.png"

os.makedirs("../models", exist_ok=True)

def run_shap_for_pca(model_path):
    # Load model
    model = joblib.load(model_path)

    # Load dataset
    df = pd.read_csv(DATA_CSV)

    # Identify all features used in training
    feature_cols = [c for c in df.columns if c not in ("Outcome", "sample_id")]
    pc_cols = [c for c in feature_cols if c.startswith("PC_gen_")]

    if len(pc_cols) == 0:
        raise ValueError("PC_gen_* columns not found")

    # SHAP must run on full dataset (all features)
    X = df[feature_cols].fillna(0)

    # Use a smaller background for SHAP speed
    background = X.sample(min(300, len(X)), random_state=42)

    print("Running SHAP on ALL features, then extracting PCA PCs only...")
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")

    # Compute SHAP values on background
    shap_values = explainer.shap_values(background)

    # For binary classifiers, shap_values is a list → use index 1 (positive class)
    if isinstance(shap_values, list):
        shap_values_use = shap_values[1]
    else:
        shap_values_use = shap_values

    # Extract only PCA components from shap output
    # Get index positions of PC_gen_* columns
    pc_idx = [feature_cols.index(col) for col in pc_cols]
    shap_pca = shap_values_use[:, pc_idx]

    # Create DataFrame for PCA part
    background_pca = background[pc_cols]

    # ========== SAVE SHAP SUMMARY ==========
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_pca, background_pca, show=False)
    plt.tight_layout()
    plt.savefig(OUT_SUM, dpi=300)
    plt.close()

    # ========== SAVE SHAP BAR CHART ==========
    plt.figure(figsize=(6, 4))
    shap.summary_plot(shap_pca, background_pca, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT_BAR, dpi=300)
    plt.close()

    print("Saved PCA-only SHAP plots:")
    print(" →", OUT_SUM)
    print(" →", OUT_BAR)


def main():
    # Prefer RF model; fallback to XGB
    if os.path.exists(RF_MODEL):
        run_shap_for_pca(RF_MODEL)
    elif os.path.exists(XGB_MODEL):
        run_shap_for_pca(XGB_MODEL)
    else:
        raise FileNotFoundError("No tree models found for SHAP PCA analysis.")


if __name__ == "__main__":
    main()
