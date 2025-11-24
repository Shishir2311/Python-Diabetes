# src/genetic_pca.py
import pandas as pd
from sklearn.decomposition import PCA
import os

INPUT = "../data/diabetes_with_genetics.csv"
OUT_PCA = "../data/genetic_pca_components.csv"
OUT_FULL = "../data/diabetes_with_genetic_pca.csv"
N_COMPONENTS = 5

def run_pca():
    print("Loading merged dataset...")
    df = pd.read_csv(INPUT)

    # SNP columns: rs1..rs50 (assumes this naming)
    snp_cols = [c for c in df.columns if c.startswith("rs")]
    if len(snp_cols) == 0:
        raise ValueError("No SNP columns found. Expected columns named rs1..rs50")

    print(f"Found {len(snp_cols)} SNP columns. Running PCA -> {N_COMPONENTS} components...")
    X = df[snp_cols].fillna(0).astype(float)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pcs = pca.fit_transform(X)

    pc_cols = [f"PC_gen_{i+1}" for i in range(N_COMPONENTS)]
    df_pcs = pd.DataFrame(pcs, columns=pc_cols)
    df_pcs.insert(0, "sample_id", df["sample_id"].values)

    print("Saving PCA components to:", OUT_PCA)
    df_pcs.to_csv(OUT_PCA, index=False)

    # Merge PCs back to original dataset (drop raw SNPs if you want)
    df_out = pd.concat([df.drop(columns=snp_cols), df_pcs.drop(columns=["sample_id"])], axis=1)
    print("Saving final dataset to:", OUT_FULL)
    df_out.to_csv(OUT_FULL, index=False)
    print("PCA done. Explained variance ratio (first 5):", pca.explained_variance_ratio_)

if __name__ == "__main__":
    run_pca()
