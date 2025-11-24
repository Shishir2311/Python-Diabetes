import shap
import joblib
import pandas as pd
import numpy as np
from preprocess import load_data, preprocess_data
import matplotlib.pyplot as plt

def flatten_shap_vector(v):
    v = np.array(v)
    if v.ndim == 2:  # (n_features, something)
        return v[:, 0]
    return v

def explain_model():
    print("Loading data and model...")
    
    df = load_data()
    X_scaled, y, scaler = preprocess_data(df)

    model = joblib.load("../models/diabetes_model.pkl")

    feature_names = df.drop("Outcome", axis=1).columns
    X_df = pd.DataFrame(X_scaled, columns=feature_names)

    print("Creating SHAP explainer (classic API)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # positive class
    else:
        shap_vals = shap_values

    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_vals, X_df, show=False)
    plt.savefig("../models/shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generating SHAP bar plot...")
    shap.summary_plot(shap_vals, X_df, plot_type="bar", show=False)
    plt.savefig("../models/shap_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generating SHAP waterfall plot...")
    v = flatten_shap_vector(shap_vals[0])

    exp = shap.Explanation(
        values=v,
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=X_df.iloc[0],
        feature_names=X_df.columns
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False)
    plt.savefig("../models/shap_waterfall_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== SHAP FILES CREATED SUCCESSFULLY ===")
    print("✔ shap_summary_plot.png")
    print("✔ shap_feature_importance.png")
    print("✔ shap_waterfall_plot.png")

if __name__ == "__main__":
    explain_model()
