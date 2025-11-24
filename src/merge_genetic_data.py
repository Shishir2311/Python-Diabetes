import pandas as pd

# Paths
DIABETES_PATH = "../data/diabetes.csv"
SNP_PATH = "../data/synthetic_snps_50_768.csv"
OUTPUT_PATH = "../data/diabetes_with_genetics.csv"

def merge_datasets():
    print("Loading diabetes data...")
    df = pd.read_csv(DIABETES_PATH)

    print("Loading synthetic SNP data...")
    df_snps = pd.read_csv(SNP_PATH)

    # Ensure diabetes dataset has 768 rows
    if len(df) != 768:
        raise ValueError(f"Diabetes dataset has {len(df)} rows, expected 768.")

    # Add sample_id column to diabetes dataset
    print("Adding sample IDs...")
    df["sample_id"] = [f"sample_{i+1}" for i in range(len(df))]

    print("Merging datasets...")
    merged_df = df.merge(df_snps, on="sample_id", how="inner")

    # Save merged dataset
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== MERGE COMPLETE ===")
    print(f"Merged dataset saved at: {OUTPUT_PATH}")
    print(f"Final shape: {merged_df.shape}")

if __name__ == "__main__":
    merge_datasets()
