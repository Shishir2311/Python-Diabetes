import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
from preprocess import load_data, preprocess_data

def explain_lime():
    print("Loading data and model...")
    
    df = load_data()
    X_scaled, y, scaler = preprocess_data(df)

    model = joblib.load("../models/diabetes_model.pkl")

    feature_names = df.drop("Outcome", axis=1).columns
    class_names = ["No Diabetes", "Diabetes"]

    explainer = LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    # Choose a sample to explain
    sample_index = 0
    sample = X_scaled[sample_index]

    print(f"Explaining sample #{sample_index} with LIME...")
    explanation = explainer.explain_instance(
        data_row=sample,
        predict_fn=model.predict_proba,
        num_features=10
    )

    explanation.save_to_file("../models/lime_explanation.html")

    print("\n=== LIME EXPLANATION CREATED ===")
    print("✔ lime_explanation.html generated")

if __name__ == "__main__":
    explain_lime()
