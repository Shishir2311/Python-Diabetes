import sys, os
sys.path.append(os.path.abspath("../src"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile
import base64

from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model
import shap


# =============================
# PATHS
# =============================

RF_MODEL = "../models/rf_pca_model.pkl"
RF_SCALER = "../models/rf_pca_scaler.pkl"

XGB_MODEL = "../models/xgb_model.pkl"

ANN_MODEL = "../models/ann_model.h5"
ANN_SCALER = "../models/ann_scaler.pkl"

PCA_MODEL = "../models/genetic_pca.pkl"
SNP_SCALER = "../models/snp_scaler.pkl"

TRAIN_DATA = "../data/diabetes_with_genetic_pca.csv"
PCA_SUMMARY_IMG = "../models/shap_pca_summary.png"
PCA_BAR_IMG = "../models/shap_pca_bar.png"

RESULTS_DIR = "../results"
ROC_IMG = "../results/roc_comparison.png"

# Normalized confusion matrices
CM_RF = "../results/confusion_matrix_randomforest_norm.png"
CM_XGB = "../results/confusion_matrix_xgboost_norm.png"
CM_ANN = "../results/confusion_matrix_ann_norm.png"


# =============================
# STREAMLIT CONFIG
# =============================

st.set_page_config(
    page_title="MedAssist360 — Diabetes Prediction with Genetics",
    layout="wide"
)


# =============================
# LOAD MODELS (cached)
# =============================

@st.cache_resource
def load_all_models():

    models = {}

    # RF
    models["rf"] = joblib.load(RF_MODEL) if os.path.exists(RF_MODEL) else None
    models["rf_scaler"] = joblib.load(RF_SCALER) if os.path.exists(RF_SCALER) else None

    # XGB
    models["xgb"] = joblib.load(XGB_MODEL) if os.path.exists(XGB_MODEL) else None

    # ANN
    models["ann"] = load_model(ANN_MODEL) if os.path.exists(ANN_MODEL) else None
    models["ann_scaler"] = joblib.load(ANN_SCALER) if os.path.exists(ANN_SCALER) else None

    # PCA model
    models["pca"] = joblib.load(PCA_MODEL) if os.path.exists(PCA_MODEL) else None
    models["snp_scaler"] = joblib.load(SNP_SCALER) if os.path.exists(SNP_SCALER) else None

    # Load training dataset for feature order + LIME background
    if os.path.exists(TRAIN_DATA):
        df = pd.read_csv(TRAIN_DATA)
        models["train_df"] = df
        models["feature_cols"] = [c for c in df.columns if c not in ("Outcome","sample_id")]
    else:
        models["train_df"] = None
        models["feature_cols"] = []

    return models


models = load_all_models()


# =============================
# HEADER
# =============================

st.markdown("""
<h1 style='text-align:center; margin-top:-30px;'>
🩺 MedAssist360 – Diabetes Risk Prediction (Clinical + Genetic PCA)
</h1>
""", unsafe_allow_html=True)

st.markdown("---")


# =============================
# SIDEBAR
# =============================

st.sidebar.header("⚙ Configuration")

model_choice = st.sidebar.selectbox(
    "Choose Prediction Model:",
    ["RandomForest", "XGBoost", "ANN"]
)

use_snp = st.sidebar.checkbox("Upload SNP CSV for genetic PCA?", value=False)

uploaded_file = None
if use_snp:
    uploaded_file = st.sidebar.file_uploader("Upload SNP CSV (columns: rs1..rs50)", type=["csv"])

st.sidebar.markdown("---")

if os.path.exists(ROC_IMG):
    st.sidebar.image(ROC_IMG, caption="Model ROC Comparison", use_column_width=True)


# =============================
# LAYOUT
# =============================

left, right = st.columns(2)


# ===========================================================
# LEFT PANEL — USER INPUTS
# ===========================================================

with left:
    st.subheader("🧍 Patient Clinical & Lifestyle Information")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin Level", 0, 1000, 80)
    bmi = st.number_input("BMI", 0.0, 80.0, 28.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
    age = st.number_input("Age", 0, 120, 35)
    family_history = st.selectbox("Family History (binary)", [0,1], index=0)

    # Manual PCs (fallback)
    st.markdown("### 🧬 Genetic PCA Inputs (if no SNP uploaded)")
    pc1 = st.number_input("PC_gen_1", value=0.0)
    pc2 = st.number_input("PC_gen_2", value=0.0)
    pc3 = st.number_input("PC_gen_3", value=0.0)
    pc4 = st.number_input("PC_gen_4", value=0.0)
    pc5 = st.number_input("PC_gen_5", value=0.0)


# ===========================================================
# RIGHT PANEL — PREDICTION & EXPLAINABILITY
# ===========================================================

with right:
    st.subheader("📈 Prediction & Explainability")

    # Build clinical feature vector
    clinical_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "family_history": family_history
    }])

    # ========== Genetic PCA from uploaded SNP CSV ==========
    pcs = None

    if use_snp and uploaded_file:
        try:
            snp_df = pd.read_csv(uploaded_file)
            snp_cols = [c for c in snp_df.columns if c.startswith("rs")]

            if len(snp_cols) < 50:
                st.error("Uploaded SNP file must include rs1..rs50 columns.")
            else:
                scaler = models["snp_scaler"]
                pca = models["pca"]

                Xs = snp_df[snp_cols].fillna(0).astype(float).values
                Xs_scaled = scaler.transform(Xs)
                pcs_all = pca.transform(Xs_scaled)

                pcs = pcs_all[0, :5]   # first sample
                st.success("Genetic PCA applied successfully.")
        except Exception as e:
            st.error("Error processing SNP CSV: " + str(e))

    # fallback to manual PCs
    if pcs is None:
        pcs = np.array([pc1, pc2, pc3, pc4, pc5])


    # ========== Combine full feature vector ==========
    feature_cols = models["feature_cols"]

    input_row = pd.DataFrame(columns=feature_cols)
    for col, val in clinical_df.iloc[0].items():
        if col in input_row.columns:
            input_row.loc[0, col] = val

    # Add PCs
    for i in range(5):
        col = f"PC_gen_{i+1}"
        if col in input_row.columns:
            input_row.loc[0, col] = pcs[i]

    input_row = input_row.fillna(0)

    st.markdown("### 🔍 Processed Input Used for Prediction:")
    st.dataframe(input_row.T, height=300)


    # ===========================================================
    # PREDICT BUTTON
    # ===========================================================

    if st.button("🔮 Predict Diabetes Risk"):
        try:
            if model_choice == "RandomForest":
                model = models["rf"]
                scaler = models["rf_scaler"]
                X_scaled = scaler.transform(input_row.values)
                proba = model.predict_proba(X_scaled)[0,1]

            elif model_choice == "XGBoost":
                model = models["xgb"]
                proba = model.predict_proba(input_row.values)[0,1]

            else:
                model = models["ann"]
                scaler = models["ann_scaler"]
                X_scaled = scaler.transform(input_row.values)
                proba = float(model.predict(X_scaled).ravel()[0])

            # Risk classification
            if proba >= 0.75:
                risk = "HIGH"
                color = "red"
            elif proba >= 0.45:
                risk = "MODERATE"
                color = "orange"
            else:
                risk = "LOW"
                color = "green"

            st.markdown(f"""
            ### 🎯 Predicted Probability: **{proba:.4f}**  
            ### ⚠ Risk Level: <span style='color:{color}; font-size:24px;'><b>{risk}</b></span>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # ===========================================================
            # EXPLAINABILITY — SHAP or LIME
            # ===========================================================

            st.subheader("🧠 Explainability (Local)")
            tmpfile = None

            if model_choice in ("RandomForest", "XGBoost"):
                try:
                    full_df = models["train_df"]
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(pd.DataFrame(input_row.values, columns=feature_cols))

                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]

                    exp = shap.Explanation(
                        values=shap_vals[0],
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                        else explainer.expected_value[1],
                        data=input_row.iloc[0].values,
                        feature_names=feature_cols
                    )

                    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    plt.figure(figsize=(8,4))
                    shap.plots.waterfall(exp, show=False)
                    plt.tight_layout()
                    plt.savefig(tmpfile.name, dpi=300)
                    plt.close()
                    st.image(tmpfile.name, caption="Local SHAP Waterfall", use_column_width=True)

                except Exception as e:
                    st.warning("SHAP local explanation failed: " + str(e))

            else:
                # LIME
                try:
                    train_df = models["train_df"]
                    lime_explainer = LimeTabularExplainer(
                        training_data=train_df[feature_cols].fillna(0).values,
                        feature_names=feature_cols,
                        class_names=["No Diabetes","Diabetes"],
                        mode="classification"
                    )

                    x_scaled = models["ann_scaler"].transform(input_row.values)
                    exp = lime_explainer.explain_instance(
                        x_scaled[0],
                        model.predict,
                        num_features=10
                    )
                    st.components.v1.html(exp.as_html(), height=450)
                except Exception as e:
                    st.warning("LIME explanation failed: " + str(e))


            # ===========================================================
            # GLOBAL PCA SHAP PLOTS
            # ===========================================================

            st.subheader("🌐 Global Explainability — Genetic PCA Importance")

            if os.path.exists(PCA_SUMMARY_IMG) and os.path.exists(PCA_BAR_IMG):
                colA, colB = st.columns(2)
                colA.image(PCA_SUMMARY_IMG, caption="SHAP Summary (PCA components)", use_column_width=True)
                colB.image(PCA_BAR_IMG, caption="SHAP Bar (PCA importance)", use_column_width=True)
            else:
                st.info("Global PCA SHAP plots not found. Run `python src/genetic_shap.py`.")

            # ===========================================================
            # PDF REPORT EXPORT
            # ===========================================================

            if st.button("📄 Download PDF Report"):
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", size=16)
                pdf.cell(0, 10, "MedAssist360 — Diabetes Prediction Report", ln=True, align="C")
                
                pdf.ln(5)
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 8, f"Model Used: {model_choice}", ln=True)
                pdf.cell(0, 8, f"Predicted Probability: {proba:.4f}", ln=True)
                pdf.cell(0, 8, f"Risk Level: {risk}", ln=True)

                pdf.ln(5)
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 8, "Patient Inputs:", ln=True)
                for k, v in input_row.iloc[0].items():
                    pdf.cell(0, 6, f"{k}: {v}", ln=True)

                # Add SHAP image
                if tmpfile and os.path.exists(tmpfile.name):
                    pdf.add_page()
                    pdf.image(tmpfile.name, x=10, y=20, w=180)

                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="medassist360_report.pdf">Click here to download PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error("Prediction failed: " + str(e))


# ===========================================================
# MODEL PERFORMANCE SECTION
# ===========================================================

st.markdown("---")
st.subheader("📊 Model Comparison & Performance Metrics")

if os.path.exists(ROC_IMG):
    st.image(ROC_IMG, caption="ROC Comparison", use_column_width=True)

col_rf, col_xgb, col_ann = st.columns(3)
if os.path.exists(CM_RF):
    col_rf.image(CM_RF, caption="RandomForest — Normalized Confusion Matrix")
if os.path.exists(CM_XGB):
    col_xgb.image(CM_XGB, caption="XGBoost — Normalized Confusion Matrix")
if os.path.exists(CM_ANN):
    col_ann.image(CM_ANN, caption="ANN — Normalized Confusion Matrix")
