"""
Streamlit Web Application for AI-Powered Credit Card Fraud Detection.

Provides an interactive dashboard featuring executive KPIs, interactive model risk scoring,
multi-model performance comparison, and detailed evaluation curves (ROC & PR).
"""

import os
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Add project root to sys.path to resolve src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import (
    load_model_and_scaler,
    predict_transaction,
    FEATURE_COLUMNS,
)
from src.preprocessing import load_dataset, validate_dataset


# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection ML",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Cybersecurity & Modern Financial Visual Theme)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .alert-fraud {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.5rem;
        color: #991B1B;
    }
    .alert-legit {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 1.5rem;
        color: #166534;
    }
    .disclaimer-box {
        font-size: 0.85rem;
        color: #64748B;
        background-color: #F1F5F9;
        padding: 0.75rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_and_scaler():
    """Load model artifacts cached."""
    try:
        model, scaler = load_model_and_scaler(os.path.join(PROJECT_ROOT, "models"))
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data
def get_metrics_data():
    """Load evaluation metrics JSON cached."""
    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def get_dataset_sample():
    """Load dataset sample cached."""
    data_path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
    try:
        df, is_synthetic, msg = load_dataset(data_path)
        stats = validate_dataset(df)
        return df, is_synthetic, stats, msg
    except Exception:
        return None, True, None, "Dataset not found locally."


# Header Banner
st.markdown('<div class="main-title">💳 AI-Powered Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning based Credit Card Transaction Fraud Detection</div>', unsafe_allow_html=True)
st.markdown("---")

# Load Resources
model, scaler, model_err = get_model_and_scaler()
metrics_json = get_metrics_data()
df_sample, is_synthetic_data, dataset_stats, data_msg = get_dataset_sample()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/security-checked.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select System Module:",
    [
        "📊 Executive Dashboard",
        "🔍 Fraud Prediction",
        "⚖️ Model Comparison",
        "📈 Model Performance",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 System Status")
if model is not None:
    st.sidebar.success("✅ Model Artifacts Loaded")
    if metrics_json and "selected_model" in metrics_json:
        st.sidebar.info(f"Active Model: **{metrics_json['selected_model']}**")
else:
    st.sidebar.error("⚠️ Model Artifacts Missing")

if is_synthetic_data:
    st.sidebar.warning("⚡ Data Mode: Synthetic Test")
else:
    st.sidebar.success("🟢 Data Mode: Kaggle Real Dataset")


# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if page == "📊 Executive Dashboard":
    st.header("📊 Executive Fraud Detection Dashboard")

    if model_err:
        st.warning(
            "⚠️ **Model Artifacts Not Trained Yet**\n\n"
            "To train models and generate saved artifacts, please run the following command in terminal:\n"
            "```powershell\npython src/train.py\n```"
        )

    if is_synthetic_data:
        st.info(
            "ℹ️ **Notice:** System currently displaying statistical stats from **Synthetic Sample Dataset** "
            "for demonstration/validation. Place `creditcard.csv` in `data/` to process the complete 284k Kaggle dataset."
        )

    if dataset_stats:
        # KPI Metric Row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Transactions", f"{dataset_stats['total_rows']:,}")
        with c2:
            st.metric("Legitimate (Class 0)", f"{dataset_stats['legitimate_count']:,}")
        with c3:
            st.metric("Fraudulent (Class 1)", f"{dataset_stats['fraud_count']:,}")
        with c4:
            st.metric("Fraud Ratio", f"{dataset_stats['fraud_percentage']:.3f}%")
        with c5:
            avg_amt = df_sample["Amount"].mean() if df_sample is not None else 0.0
            st.metric("Avg Amount", f"${avg_amt:.2f}")

        st.markdown("### 📈 Dataset Distribution & Characteristics")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Class Imbalance Ratio")
            fig, ax = plt.subplots(figsize=(6, 4))
            counts = [dataset_stats['legitimate_count'], dataset_stats['fraud_count']]
            labels = ["Legitimate (0)", "Fraud (1)"]
            colors = ["#22C55E", "#EF4444"]
            bars = ax.bar(labels, counts, color=colors, width=0.5)
            ax.set_yscale("log")
            ax.set_ylabel("Count (Log Scale)", fontsize=10)
            ax.set_title("Legitimate vs Fraudulent Transactions (Log Scale)", fontsize=11, fontweight="bold")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
            st.pyplot(fig)

        with col_right:
            st.subheader("Transaction Amount Comparison")
            if df_sample is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x="Class", y="Amount", data=df_sample, palette=["#22C55E", "#EF4444"], ax=ax, showfliers=False)
                ax.set_xticklabels(["Legitimate", "Fraud"])
                ax.set_title("Transaction Amount Distribution (Excluding Extreme Outliers)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Amount ($)", fontsize=10)
                st.pyplot(fig)

    if metrics_json and "models_performance" in metrics_json:
        st.markdown("### 🏆 Active Selected Model Highlights")
        selected_name = metrics_json.get("selected_model", "Classifier")
        perf = metrics_json["models_performance"].get(selected_name, {})
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PR-AUC (Key Metric)", f"{perf.get('pr_auc', 0):.4f}")
        c2.metric("F1-Score", f"{perf.get('f1_score', 0):.4f}")
        c3.metric("Precision", f"{perf.get('precision', 0):.4f}")
        c4.metric("Recall", f"{perf.get('recall', 0):.4f}")
        c5.metric("ROC-AUC", f"{perf.get('roc_auc', 0):.4f}")


# ==========================================
# PAGE 2: FRAUD PREDICTION
# ==========================================
elif page == "🔍 Fraud Prediction":
    st.header("🔍 Interactive Transaction Fraud Risk Assessment")
    st.write("Enter financial transaction features below to calculate ML fraud risk probability.")

    if model is None:
        st.error("🚨 Model artifacts not found! Please run `python src/train.py` to train the model first.")
    else:
        # Quick Preset Buttons
        st.subheader("💡 Quick Input Presets")
        p_col1, p_col2, p_col3 = st.columns(3)
        
        preset_type = st.radio(
            "Select Test Scenario:",
            ["Legitimate Sample Preset", "Suspicious Fraud Sample Preset", "Custom Input"],
            horizontal=True
        )

        # Default feature dictionary
        input_data = {}

        if preset_type == "Legitimate Sample Preset":
            input_data["Time"] = 45000.0
            input_data["Amount"] = 45.50
            for i in range(1, 29):
                input_data[f"V{i}"] = round(float(np.random.normal(0, 0.5)), 4)
        elif preset_type == "Suspicious Fraud Sample Preset":
            input_data["Time"] = 120000.0
            input_data["Amount"] = 350.00
            # Set key fraud-indicative features to known fraud shifts
            for i in range(1, 29):
                if i in [1, 3, 4, 10, 12, 14, 17]:
                    input_data[f"V{i}"] = -4.50
                else:
                    input_data[f"V{i}"] = round(float(np.random.normal(0.5, 1.0)), 4)
        else: # Custom
            input_data["Time"] = 80000.0
            input_data["Amount"] = 100.00
            for i in range(1, 29):
                input_data[f"V{i}"] = 0.0

        st.subheader("📝 Transaction Feature Inputs")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                input_data["Time"] = st.number_input("Time (Seconds elapsed)", value=float(input_data["Time"]))
            with c2:
                input_data["Amount"] = st.number_input("Amount ($)", value=float(input_data["Amount"]), min_value=0.0)

            st.markdown("#### PCA Features (V1 to V28)")
            v_cols = st.columns(4)
            for idx in range(1, 29):
                col_target = v_cols[(idx - 1) % 4]
                feat_key = f"V{idx}"
                input_data[feat_key] = col_target.number_input(
                    feat_key, value=float(input_data[feat_key]), step=0.1, key=f"in_{feat_key}"
                )

            submit_btn = st.form_submit_button("🔍 Predict Transaction Risk")

        if submit_btn:
            try:
                res = predict_transaction(input_data, model, scaler)

                st.markdown("---")
                st.subheader("🎯 Risk Assessment Prediction Output")

                if res["prediction_class"] == 1:
                    st.markdown(
                        f"""
                        <div class="alert-fraud">
                            <h2>⚠️ Potential Fraud Detected</h2>
                            <p style="font-size: 1.2rem; font-weight: bold;">Fraud Probability: {res['fraud_percentage']:.2f}%</p>
                            <p>Legitimate Probability: {res['legit_percentage']:.2f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="alert-legit">
                            <h2>✅ Transaction Classified as Legitimate</h2>
                            <p style="font-size: 1.2rem; font-weight: bold;">Legitimate Probability: {res['legit_percentage']:.2f}%</p>
                            <p>Fraud Probability: {res['fraud_percentage']:.2f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Probability Progress Meter
                st.write("#### Risk Level Gauge")
                st.progress(res["fraud_probability"])

                # Safety Disclaimer
                st.markdown(
                    f'<div class="disclaimer-box">🛡️ <strong>Safety Note:</strong> {res["disclaimer"]}</div>',
                    unsafe_allow_html=True,
                )

            except Exception as ex:
                st.error(f"Error executing prediction: {str(ex)}")


# ==========================================
# PAGE 3: MODEL COMPARISON
# ==========================================
elif page == "⚖️ Model Comparison":
    st.header("⚖️ Classifier Performance Comparison")
    st.write("Compare Logistic Regression, Random Forest, and XGBoost across fraud evaluation metrics.")

    if not metrics_json or "comparison_table" not in metrics_json:
        st.warning("⚠️ Model comparison metrics not available. Please execute `python src/train.py`.")
    else:
        df_comp = pd.DataFrame(metrics_json["comparison_table"])

        st.subheader("📋 Evaluation Summary Table")
        st.dataframe(df_comp.style.highlight_max(subset=["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], color="#DCFCE7"))

        st.markdown("### 📊 Metric Comparison Chart")
        metrics_to_plot = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        
        df_plot = df_comp.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Metric", y="Score", hue="Model", data=df_plot, palette="Set2", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
        ax.set_title("Classifier Comparison across Imbalance-Aware Metrics", fontsize=12, fontweight="bold")
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
        st.pyplot(fig)


# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================
elif page == "📈 Model Performance":
    st.header("📈 Detailed Selected Model Diagnostics")

    if not metrics_json or "models_performance" not in metrics_json:
        st.warning("⚠️ Detailed diagnostic curves unavailable. Run `python src/train.py`.")
    else:
        selected_model_name = metrics_json.get("selected_model", "Classifier")
        perf = metrics_json["models_performance"].get(selected_model_name, {})

        st.subheader(f"Diagnostic Metrics for Selected Model: '{selected_model_name}'")

        col1, col2 = st.columns(2)

        # Confusion Matrix Heatmap
        with col1:
            st.subheader("Confusion Matrix")
            cm = np.array(perf.get("confusion_matrix", [[0, 0], [0, 0]]))
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Legit (0)", "Fraud (1)"], yticklabels=["Legit (0)", "Fraud (1)"])
            ax.set_xlabel("Predicted Label", fontsize=10)
            ax.set_ylabel("True Label", fontsize=10)
            ax.set_title(f"Confusion Matrix ({selected_model_name})", fontsize=11, fontweight="bold")
            st.pyplot(fig)

        # Classification Report Table
        with col2:
            st.subheader("Classification Report")
            clf_rep = perf.get("classification_report", {})
            if clf_rep:
                rep_df = pd.DataFrame(clf_rep).transpose()
                st.dataframe(rep_df.style.format("{:.4f}"))

        st.markdown("---")
        st.subheader("📉 Evaluation Curves")
        c_curve1, c_curve2 = st.columns(2)

        # ROC Curve Plot
        with c_curve1:
            st.subheader("ROC Curve")
            roc_data = perf.get("roc_curve", {})
            if roc_data:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(roc_data["fpr"], roc_data["tpr"], color="#3B82F6", lw=2, label=f"ROC (AUC = {perf.get('roc_auc', 0):.4f})")
                ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
                ax.set_xlabel("False Positive Rate", fontsize=10)
                ax.set_ylabel("True Positive Rate (Recall)", fontsize=10)
                ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
                ax.legend(loc="lower right")
                st.pyplot(fig)

        # PR Curve Plot
        with c_curve2:
            st.subheader("Precision-Recall Curve")
            pr_data = perf.get("pr_curve", {})
            if pr_data:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(pr_data["recall"], pr_data["precision"], color="#10B981", lw=2, label=f"PR (AUC = {perf.get('pr_auc', 0):.4f})")
                ax.set_xlabel("Recall", fontsize=10)
                ax.set_ylabel("Precision", fontsize=10)
                ax.set_title("Precision-Recall Curve (Primary Imbalance Metric)", fontsize=11, fontweight="bold")
                ax.legend(loc="lower left")
                st.pyplot(fig)
