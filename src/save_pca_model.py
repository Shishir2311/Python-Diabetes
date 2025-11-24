# src/save_pca_model.py
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

INPUT = "../data/diabetes_with_genetic_pca.csv"   # contains PC_gen_* columns if previous run; we want raw SNPs
MERGED_RAW = "../data/diabetes_with_genetics.csv"  # original merged file with rs1..rs50
PCA_OUT = "../models/genetic_pca.pkl"
SCALER_OUT = "../models/snp_scaler.pkl"
N_COMPONENTS = 5

os.makedirs("../models", exist_ok=True)

def run_and_save():
    # Prefer raw merged file with SNP columns
    if not os.path.exists(MERGED_RAW):
        raise FileNotFoundError(f"Expected file: {MERGED_RAW} (merged raw SNPs).")

    df = pd.read_csv(MERGED_RAW)
    snp_cols = [c for c in df.columns if c.startswith("rs")]
    if len(snp_cols) == 0:
        raise ValueError("No SNP columns found in merged file; expected rs1..rsN")

    X = df[snp_cols].fillna(0).astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pcs = pca.fit(Xs)

    # Save scaler and PCA
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(pca, PCA_OUT)

    print("Saved SNP scaler to:", SCALER_OUT)
    print("Saved PCA object to:", PCA_OUT)
    print("PCA explained variance ratio (first components):", pca.explained_variance_ratio_)

if __name__ == "__main__":
    run_and_save()
