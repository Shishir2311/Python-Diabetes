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

RF_MODEL = "models/rf_pca_model.pkl"
RF_SCALER = "models/rf_pca_scaler.pkl"

XGB_MODEL = "models/xgb_model.pkl"

ANN_MODEL = "models/ann_model.h5"
ANN_SCALER = "models/ann_scaler.pkl"

PCA_MODEL = "models/genetic_pca.pkl"
SNP_SCALER = "models/snp_scaler.pkl"

TRAIN_DATA = "data/diabetes_with_genetic_pca.csv"
PCA_SUMMARY_IMG = "models/shap_pca_summary.png"
PCA_BAR_IMG = "models/shap_pca_bar.png"

RESULTS_DIR = "results"
ROC_IMG = "results/roc_comparison.png"

# Normalized confusion matrices
CM_RF = "results/confusion_matrix_randomforest_norm.png"
CM_XGB = "results/confusion_matrix_xgboost_norm.png"
CM_ANN = "results/confusion_matrix_ann_norm.png"


# =============================
# STREAMLIT CONFIG
# =============================

st.set_page_config(
    page_title="DiabetesAI Pro — Advanced Diabetes Prediction",
    layout="wide",
    page_icon="🩺"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #ffa726, #ff9800);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #66bb6a, #4caf50);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)


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

# Display model loading status
with st.sidebar:
    st.markdown("### 🔧 Model Status")
    st.write(f"RandomForest: {'✅' if models['rf'] is not None else '❌'}")
    st.write(f"XGBoost: {'✅' if models['xgb'] is not None else '❌'}")
    st.write(f"ANN: {'✅' if models['ann'] is not None else '❌'}")
    st.write(f"Training Data: {'✅' if models['train_df'] is not None else '❌'}")
    st.markdown("---")


# =============================
# HEADER
# =============================

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.5rem;">
        🩺 DiabetesAI Pro
    </h1>
    <p style="margin:0.5rem 0 0 0; font-size:1.2rem; opacity:0.9;">
        Advanced Diabetes Risk Prediction with Clinical & Genetic Analysis
    </p>
</div>
""", unsafe_allow_html=True)


# =============================
# SIDEBAR
# =============================

st.sidebar.markdown("""
<div style="text-align:center; padding:1rem; background:linear-gradient(135deg, #667eea, #764ba2); border-radius:10px; margin-bottom:1rem;">
    <h2 style="color:white; margin:0;">⚙️ Configuration</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🤖 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose AI Model:",
    ["RandomForest", "XGBoost", "ANN"],
    help="Select the machine learning model for prediction"
)

st.sidebar.markdown("### 🧬 Genetic Analysis")
use_snp = st.sidebar.checkbox("🧠 Enable Genetic PCA Analysis", value=False)

uploaded_file = None
if use_snp:
    uploaded_file = st.sidebar.file_uploader(
        "📄 Upload SNP Data (CSV)", 
        type=["csv"],
        help="Upload CSV file with rs1 to rs50 columns"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Performance")

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
    st.markdown("""
    <div style="background:linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding:1rem; border-radius:10px; margin-bottom:1rem;">
        <h3 style="color:white; margin:0; text-align:center;">🧍 Patient Information</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🩺 Clinical Measurements")
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("🤰 Pregnancies", 0, 20, 1)
        glucose = st.number_input("🍭 Glucose Level (mg/dL)", 0, 300, 120)
        bp = st.number_input("❤️ Blood Pressure (mmHg)", 0, 200, 70)
        skin = st.number_input("🔍 Skin Thickness (mm)", 0, 100, 20)
        insulin = st.number_input("💉 Insulin Level (μU/mL)", 0, 1000, 80)
    
    with col2:
        bmi = st.number_input("⚖️ BMI (kg/m²)", 0.0, 80.0, 28.0)
        dpf = st.number_input("🧬 Diabetes Pedigree Function", 0.0, 5.0, 0.5)
        age = st.number_input("🎂 Age (years)", 0, 120, 35)
        family_history = st.selectbox("👨‍👩‍👧‍👦 Family History", [0,1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")

    st.markdown("#### 🧬 Manual Genetic Components")
    st.info("📝 Use these fields only if not uploading SNP data")
    
    pc_col1, pc_col2 = st.columns(2)
    with pc_col1:
        pc1 = st.number_input("🧬 PC_gen_1", value=0.0)
        pc2 = st.number_input("🧬 PC_gen_2", value=0.0)
        pc3 = st.number_input("🧬 PC_gen_3", value=0.0)
    with pc_col2:
        pc4 = st.number_input("🧬 PC_gen_4", value=0.0)
        pc5 = st.number_input("🧬 PC_gen_5", value=0.0)


# ===========================================================
# RIGHT PANEL — PREDICTION & EXPLAINABILITY
# ===========================================================

with right:
    st.markdown("""
    <div style="background:linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding:1rem; border-radius:10px; margin-bottom:1rem;">
        <h3 style="color:white; margin:0; text-align:center;">📈 AI Prediction & Analysis</h3>
    </div>
    """, unsafe_allow_html=True)

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
    
    if not feature_cols:
        st.error("Feature columns not found. Please check training data file.")
        st.stop()

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

    with st.expander("🔍 View Processed Input Data", expanded=False):
        st.dataframe(input_row.T, height=300)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button("🔮 Predict Diabetes Risk", use_container_width=True)

    if predict_clicked:
        try:
            # Check if models are loaded
            if model_choice == "RandomForest":
                model = models["rf"]
                scaler = models["rf_scaler"]
                if model is None or scaler is None:
                    st.error("RandomForest model or scaler not found. Please check model files.")
                    st.stop()
                X_scaled = scaler.transform(input_row.values)
                proba = model.predict_proba(X_scaled)[0,1]

            elif model_choice == "XGBoost":
                model = models["xgb"]
                if model is None:
                    st.error("XGBoost model not found. Please check model files.")
                    st.stop()
                proba = model.predict_proba(input_row.values)[0,1]

            else:
                model = models["ann"]
                scaler = models["ann_scaler"]
                if model is None or scaler is None:
                    st.error("ANN model or scaler not found. Please check model files.")
                    st.stop()
                X_scaled = scaler.transform(input_row.values)
                proba = float(model.predict(X_scaled).ravel()[0])

            # Risk classification with enhanced styling
            if proba >= 0.75:
                risk = "HIGH RISK"
                risk_class = "risk-high"
                risk_icon = "🔴"
            elif proba >= 0.45:
                risk = "MODERATE RISK"
                risk_class = "risk-moderate"
                risk_icon = "🟠"
            else:
                risk = "LOW RISK"
                risk_class = "risk-low"
                risk_icon = "🟢"

            # Display results in beautiful cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#667eea;">🎯 Probability Score</h3>
                    <h1 style="margin:0.5rem 0; color:#333;">{proba:.1%}</h1>
                    <p style="margin:0; color:#666;">Confidence: {proba:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="{risk_class}">
                    <h3 style="margin:0;">{risk_icon} Risk Assessment</h3>
                    <h1 style="margin:0.5rem 0;">{risk}</h1>
                    <p style="margin:0; opacity:0.9;">Based on AI Analysis</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ===========================================================
            # EXPLAINABILITY — SHAP or LIME
            # ===========================================================

            st.markdown("""
            <div style="background:linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding:1rem; border-radius:10px; margin:1rem 0;">
                <h3 style="margin:0; text-align:center; color:#333;">🧠 AI Model Explanation</h3>
                <p style="margin:0.5rem 0 0 0; text-align:center; color:#666;">Understanding how the AI made this prediction</p>
            </div>
            """, unsafe_allow_html=True)
            tmpfile = None

            if model_choice in ("RandomForest", "XGBoost"):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(input_row.values)
                    
                    # Get values for binary classification
                    if isinstance(shap_vals, list):
                        values = shap_vals[1][0]  # positive class, first sample
                    else:
                        values = shap_vals[0]
                    
                    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    plt.figure(figsize=(8,4))
                    shap.plots.bar(shap.Explanation(
                        values=values,
                        feature_names=feature_cols
                    ), show=False)
                    plt.tight_layout()
                    plt.savefig(tmpfile.name, dpi=300)
                    plt.close()
                    st.image(tmpfile.name, caption="Local SHAP Feature Importance", use_column_width=True)

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

            st.markdown("""
            <div style="background:linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding:1rem; border-radius:10px; margin:1rem 0;">
                <h3 style="margin:0; text-align:center; color:#333;">🌐 Global Model Insights</h3>
                <p style="margin:0.5rem 0 0 0; text-align:center; color:#666;">Genetic PCA component importance across all predictions</p>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists(PCA_SUMMARY_IMG) and os.path.exists(PCA_BAR_IMG):
                colA, colB = st.columns(2)
                colA.image(PCA_SUMMARY_IMG, caption="SHAP Summary (PCA components)", use_column_width=True)
                colB.image(PCA_BAR_IMG, caption="SHAP Bar (PCA importance)", use_column_width=True)
            else:
                st.info("Global PCA SHAP plots not found. Run `python src/genetic_shap.py`.")

            # ===========================================================
            # PDF REPORT EXPORT
            # ===========================================================

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Generate PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=16)
            pdf.cell(0, 10, "DiabetesAI Pro - Prediction Report", ln=True, align="C")
            
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

            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name="diabetesai_pro_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        except Exception as e:
            st.error("Prediction failed: " + str(e))


# ===========================================================
# MODEL PERFORMANCE SECTION
# ===========================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:2rem; border-radius:10px; margin:2rem 0;">
    <h2 style="color:white; margin:0; text-align:center;">📊 Model Performance Dashboard</h2>
    <p style="color:white; margin:0.5rem 0 0 0; text-align:center; opacity:0.9;">Comprehensive comparison of AI model accuracy and performance</p>
</div>
""", unsafe_allow_html=True)

# ROC Curve Comparison
if os.path.exists(ROC_IMG):
    st.markdown("### 📈 ROC Curve Analysis")
    st.image(ROC_IMG, caption="Model Performance Comparison - ROC Curves", use_column_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Confusion Matrices
st.markdown("### 🗓️ Confusion Matrix Analysis")
st.markdown("<p style='text-align:center; color:#666; margin-bottom:1rem;'>Detailed breakdown of model predictions vs actual outcomes</p>", unsafe_allow_html=True)

col_rf, col_xgb, col_ann = st.columns(3)

with col_rf:
    if os.path.exists(CM_RF):
        st.markdown("""
        <div style="text-align:center; padding:0.5rem; background:#f8f9fa; border-radius:8px; margin-bottom:0.5rem;">
            <h4 style="margin:0; color:#667eea;">🌲 Random Forest</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(CM_RF, use_column_width=True)

with col_xgb:
    if os.path.exists(CM_XGB):
        st.markdown("""
        <div style="text-align:center; padding:0.5rem; background:#f8f9fa; border-radius:8px; margin-bottom:0.5rem;">
            <h4 style="margin:0; color:#667eea;">⚡ XGBoost</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(CM_XGB, use_column_width=True)

with col_ann:
    if os.path.exists(CM_ANN):
        st.markdown("""
        <div style="text-align:center; padding:0.5rem; background:#f8f9fa; border-radius:8px; margin-bottom:0.5rem;">
            <h4 style="margin:0; color:#667eea;">🧠 Neural Network</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(CM_ANN, use_column_width=True)
