from __future__ import annotations

from datetime import date, time
from pathlib import Path
import shutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    generate_transactions,
    normalize_transactions,
)
from src.modeling import train_models


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "credit_card_fraud_full_training_dataset.csv"
DOWNLOADS_PATH = Path(r"C:\Users\Vishakha\Downloads\credit_card_fraud_full_training_dataset.csv")
MODEL_PATH = ROOT / "models" / "model_bundle.joblib"
PAGES = ["Dashboard", "Predict Transaction", "Model Comparison", "Analytics", "Data & Models", "About"]

st.set_page_config(page_title="Credit Card Fraud Detection | Operations", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#e8f1f2; --muted:#91a8b0; --paper:#0a141b; --line:#263b45; --teal:#43d6c4; --panel:#12232c; --panel-raised:#172a34; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(145deg,rgba(33,75,78,.18),transparent 34%),linear-gradient(0deg,#0a141b,#0c1921); color:var(--ink); }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:0 !important; }
    .block-container { padding-top:1.45rem; padding-bottom:2rem; max-width:1500px; }
    [data-testid="stSidebar"] { background:#081219; border-right:1px solid #20343d; }
    [data-testid="stSidebar"] * { color:#d9e7e9; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:5px; padding:.35rem .45rem; }
    [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; }
    [data-testid="stMetricLabel"] p { color:var(--muted); }
    [data-testid="stMetricValue"] { color:#f2f8f8; }
    div[data-testid="stPlotlyChart"] { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:5px; }
    [data-testid="stWidgetLabel"], [data-testid="stForm"] label, .stSelectbox label, .stNumberInput label, .stDateInput label, .stTimeInput label, .stTextInput label, .stCheckbox label { color:#F1F5F9 !important; font-weight:600 !important; }
    div[data-testid="stForm"] input, div[data-testid="stForm"] select, div[data-testid="stForm"] .stNumberInput input, div[data-testid="stForm"] .stDateInput input, div[data-testid="stForm"] .stTimeInput input, div[data-testid="stForm"] .stSelectbox div[data-baseweb="select"] > div, div[data-testid="stForm"] .stSelectbox [data-testid="stBaseButton-secondary"], div[data-testid="stForm"] .stDateInput [data-testid="stBaseButton-secondary"] { background:#F8FAFC !important; color:#0F172A !important; border:1px solid #CBD5E1 !important; }
    div[data-testid="stForm"] .stNumberInput input:focus, div[data-testid="stForm"] .stSelectbox div[role="combobox"]:focus-within, div[data-testid="stForm"] .stDateInput input:focus, div[data-testid="stForm"] .stTimeInput input:focus { border-color:#43d6c4 !important; box-shadow:0 0 0 1px rgba(67,214,196,0.35) !important; }
    div[data-testid="stForm"] .stSelectbox [data-baseweb="select"] > div[role="button"], div[data-testid="stForm"] .stDateInput .react-datepicker-wrapper input, div[data-testid="stForm"] .stTimeInput input { background:#F8FAFC !important; color:#0F172A !important; }
    div[data-testid="stForm"] .stCheckbox label { color:#F1F5F9 !important; }
    div[data-testid="stDownloadButton"] > button,
    div.stDownloadButton > button {
        background:linear-gradient(135deg,#67e8f9,#7dd3fc) !important;
        color:#0f172a !important;
        border:1px solid #67e8f9 !important;
        border-radius:0.65rem !important;
        font-weight:700 !important;
        box-shadow:none !important;
        padding:0.7rem 1.1rem !important;
        width:auto !important;
        min-width:220px !important;
        display:inline-flex !important;
        align-items:center !important;
        justify-content:center !important;
        gap:0.5rem !important;
    }
    div[data-testid="stDownloadButton"] > button:hover,
    div.stDownloadButton > button:hover { background:linear-gradient(135deg,#a5f3fc,#bae6fd) !important; }
    div[data-testid="stDownloadButton"] > button:disabled,
    div.stDownloadButton > button:disabled { opacity:1 !important; cursor:not-allowed !important; }
    .eyebrow { color:var(--teal); text-transform:uppercase; font-weight:700; font-size:.76rem; letter-spacing:1.1px; }
    .page-title { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; margin:.1rem 0 .3rem; }
    .subtle { color:var(--muted); }
    .notice { border-left:4px solid #f0bc61; background:#2b261b; padding:12px 16px; border-radius:3px; color:#f3dfb5; }
    .result { padding:22px; border-radius:6px; text-align:center; }
    .result.fraud { background:#351e25; border:1px solid #773c46; color:#ffd6d2; }
    .result.legit { background:#14342f; border:1px solid #28695c; color:#c8f1e4; }
    .result strong { display:block; font-family:'Space Grotesk',sans-serif; font-size:1.4rem; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_active_dataset_path() -> Path:
    """Ensure dataset is located at data/credit_card_fraud_full_training_dataset.csv."""
    if DATA_PATH.exists():
        return DATA_PATH
    if DOWNLOADS_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DOWNLOADS_PATH, DATA_PATH)
        return DATA_PATH
    return DATA_PATH


def page_heading(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">Transaction Risk Operations</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


@st.cache_data(show_spinner=False)
def read_dataset(path_text: str, modified: float) -> pd.DataFrame:
    del modified
    return normalize_transactions(pd.read_csv(path_text, low_memory=False))


@st.cache_resource(show_spinner=False)
def read_bundle(path_text: str, modified: float) -> dict:
    import joblib

    del modified
    return joblib.load(path_text)


def ensure_project_data(rows: int = 4000) -> None:
    path = get_active_dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raw_data = pd.read_csv(path, low_memory=False)
        data = normalize_transactions(raw_data)
        data_note = f"Loaded dataset: {path.name} ({len(data):,} rows)."
    else:
        data = generate_transactions(rows=rows)
        data.to_csv(path, index=False)
        data_note = "Generated demonstration dataset."

    train_models(data, MODEL_PATH, data_note=data_note)
    read_dataset.clear()
    read_bundle.clear()


def setup_panel() -> None:
    st.markdown(
        '<div class="notice"><b>Setup needed.</b> Click below to load your dataset <code>credit_card_fraud_full_training_dataset.csv</code> and fit leak-free ML models.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("Load dataset and train models", type="primary", icon=":material/settings:"):
        with st.spinner("Training leak-free models on your dataset…"):
            ensure_project_data()
        st.success("Dataset loaded and models trained successfully!")
        st.rerun()


def data_models_page() -> None:
    page_heading("Data & Models", "Inspect active dataset and train/retrain machine learning models.")
    path = get_active_dataset_path()
    if path.exists():
        current_data = read_dataset(str(path), path.stat().st_mtime)
        st.metric("Loaded transactions", f"{len(current_data):,}")
        if MODEL_PATH.exists():
            bundle = read_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            st.caption(f"Active model note: {bundle.get('data_note', 'Source not recorded.')}")
    else:
        st.caption("No dataset loaded yet. Upload your CSV or click below to train.")

    if st.button("Retrain models on current dataset", type="primary", icon=":material/refresh:"):
        with st.spinner("Retraining leak-free models…"):
            ensure_project_data()
        st.success("Models retrained successfully.")
        st.rerun()

    st.markdown("---")
    st.markdown("### Upload Custom Dataset CSV")
    st.markdown("Upload a labeled transaction CSV containing `is_fraud` (0 = legitimate, 1 = fraud).")
    uploaded_file = st.file_uploader("Transaction dataset CSV", type=["csv"])
    if st.button("Validate CSV and train all models", type="primary", icon=":material/upload_file:", disabled=uploaded_file is None):
        try:
            raw_transactions = pd.read_csv(uploaded_file, low_memory=False)
            transactions = normalize_transactions(raw_transactions)
            if transactions[TARGET_COLUMN].value_counts().min() < 2:
                raise ValueError("The dataset needs at least two examples of each class for stratified evaluation.")
            temporary_model = MODEL_PATH.with_name("model_bundle.upload.tmp.joblib")
            with st.spinner(f"Training on {len(transactions):,} uploaded transactions…"):
                bundle = train_models(
                    transactions,
                    temporary_model,
                    data_note=f"Uploaded dataset: {uploaded_file.name} ({len(transactions):,} rows).",
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(uploaded_file.getvalue())
                temporary_model.replace(MODEL_PATH)
            read_dataset.clear()
            read_bundle.clear()
            st.success(f"Successfully loaded {len(transactions):,} transactions and trained models!")
            st.rerun()
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
            st.error(f"Could not load this dataset: {error}")


def chart_layout(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=12, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12232c",
        font=dict(family="DM Sans, sans-serif", color="#d5e4e7"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#b9cbd0")),
        hoverlabel=dict(bgcolor="#20343e", font_color="#eff7f7"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#354b55", tickfont=dict(color="#a9bdc2"), title_font=dict(color="#c4d4d7"))
    fig.update_yaxes(gridcolor="#263a43", zeroline=False, tickfont=dict(color="#a9bdc2"), title_font=dict(color="#c4d4d7"))
    return fig


def sample_transactions_by_class(data: pd.DataFrame, examples_per_class: int = 5) -> pd.DataFrame:
    fraud_examples = data.loc[data[TARGET_COLUMN] == 1].head(examples_per_class)
    legitimate_examples = data.loc[data[TARGET_COLUMN] == 0].head(examples_per_class)

    sample = pd.concat([fraud_examples, legitimate_examples])
    display_cols = [col for col in ["year", "month", "day", "card_network", "payment_method", "category", "amount", "is_foreign", TARGET_COLUMN] if col in sample.columns]
    return sample[display_cols].sample(frac=1, random_state=7).reset_index(drop=True)


def get_best_model_by_metric(bundle: dict, metric: str) -> tuple[str, float]:
    model_name, metrics = max(bundle["metrics"].items(), key=lambda item: item[1][metric])
    return model_name, float(metrics[metric])


def dashboard(data: pd.DataFrame, bundle: dict) -> None:
    page_heading("Fraud Detection Overview", "Monitor transaction volume, risk signals, and class balance.")
    total = len(data)
    fraud_count = int(data[TARGET_COLUMN].sum())
    legit_count = total - fraud_count
    fraud_rate = 100 * fraud_count / max(total, 1)

    best_accuracy_model, best_accuracy = get_best_model_by_metric(bundle, "accuracy")
    best_pr_auc_model, best_pr_auc = get_best_model_by_metric(bundle, "pr_auc")

    columns = st.columns(5)
    columns[0].metric("Transactions", f"{total:,}", "Loaded dataset")
    columns[1].metric("Flagged fraud (is_fraud=1)", f"{fraud_count:,}", f"{fraud_rate:.2f}% of sample")
    columns[2].metric("Legitimate (is_fraud=0)", f"{legit_count:,}", f"{100 - fraud_rate:.2f}% of sample")
    columns[3].metric("Best Model · Accuracy", f"{best_accuracy * 100:.1f}%", best_accuracy_model.replace("_", " "))
    columns[4].metric("Best Model · PR-AUC", f"{best_pr_auc:.3f}", f"{best_pr_auc_model.replace('_', ' ')} · highest PR-AUC")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Transaction Amount Distribution")
        amount_view = data.assign(Label=data[TARGET_COLUMN].map({0: "Legitimate (0)", 1: "Fraud (1)"}))
        amount_cap = float(data["amount"].quantile(0.99))
        amount_view = amount_view[amount_view["amount"] <= amount_cap]
        fig = px.histogram(
            amount_view,
            x="amount",
            color="Label",
            color_discrete_map={"Legitimate (0)": "#43d6c4", "Fraud (1)": "#ff796f"},
            nbins=40,
            barmode="overlay",
            opacity=0.78,
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
    with right:
        st.subheader("Class Balance")
        counts = pd.DataFrame({"Class": ["Legitimate (0)", "Fraud (1)"], "Transactions": [legit_count, fraud_count]})
        fig = px.bar(counts, x="Class", y="Transactions", color="Class", text_auto=".2s",
                     color_discrete_map={"Legitimate (0)": "#43d6c4", "Fraud (1)": "#ff796f"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig), width="stretch")

    st.subheader("Sample Transactions · Both Classes")
    sample = sample_transactions_by_class(data)
    st.dataframe(sample.rename(columns={TARGET_COLUMN: "is_fraud (0=legit, 1=fraud)"}),
                 width="stretch", hide_index=True)
    st.caption(bundle.get("data_note", "Dataset source not recorded."))


def prediction_page(bundle: dict) -> None:
    page_heading("Predict a Transaction", "Enter transaction details to evaluate fraud probability using the trained model.")
    model_name = bundle["best_model"]
    st.caption(f"Active model: **{model_name}** · prediction threshold: 0.50")

    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)
        transaction_date = col1.date_input("Transaction Date", value=date(2022, 5, 15))
        transaction_time = col2.time_input("Transaction Time", value=time(14, 30))
        amount = col3.number_input("Transaction Amount ($)", min_value=0.0, value=125.0, step=5.0)

        col4, col5, col6 = st.columns(3)
        card_network = col4.selectbox("Card Network", ["Visa", "Mastercard", "American Express", "JCB", "Discover"])
        payment_method = col5.selectbox("Payment Method", ["Chip Transaction", "Online Transaction", "Contactless Payment", "Swipe Transaction"])
        category = col6.selectbox("Category", ["gas_transport", "online_shopping", "restaurant", "health", "retail", "services", "grocery_pos", "travel"])

        col7, col8, col9, col10 = st.columns(4)
        mcc = col7.number_input("MCC Code", min_value=1000, max_value=9999, value=5964, step=1)
        is_foreign = col8.selectbox("Is Foreign Transaction", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        days_since_last = col9.number_input("Days Since Last Tx", min_value=0.0, value=15.2, step=1.0)
        amount_delta = col10.number_input("Tx Amount Delta ($)", value=25.0, step=5.0)

        submitted = st.form_submit_button("Score Transaction", type="primary", icon=":material/search:")

    if submitted:
        time_str = f"{transaction_time.hour:02d}:{transaction_time.minute:02d}"
        input_dict = {
            "year": transaction_date.year,
            "month": transaction_date.month,
            "day": transaction_date.day,
            "time": time_str,
            "card_network": card_network,
            "payment_method": payment_method,
            "category": category,
            "mcc": mcc,
            "amount": amount,
            "is_foreign": is_foreign,
            "days_since_last_transaction": days_since_last,
            "transaction_amount_delta": amount_delta,
            "is_fraud": 0,  # Placeholder target for schema normalization
        }

        raw_df = pd.DataFrame([input_dict])
        processed_df = normalize_transactions(raw_df)[FEATURE_COLUMNS]

        pipeline = bundle["models"][model_name]
        probability = float(pipeline.predict_proba(processed_df)[0, 1])
        is_fraud = probability >= 0.5
        label = "Potential Fraud" if is_fraud else "Likely Legitimate"
        style = "fraud" if is_fraud else "legit"

        result, detail = st.columns([1, 2])
        with result:
            st.markdown(
                f'<div class="result {style}"><strong>{label}</strong><br>Fraud probability<br><b>{probability:.1%}</b></div>',
                unsafe_allow_html=True,
            )
        with detail:
            st.progress(probability, text=f"Fraud probability: {probability:.1%}")
            st.warning("Model prediction signal for operational review; not a guaranteed financial decision.")


def comparison_page(bundle: dict) -> None:
    page_heading("Model Comparison", "Evaluated on stratified held-out test split (25%). Preprocessing fitted strictly on training data to prevent leakage.")
    metrics_df = pd.DataFrame(bundle["metrics"]).T.reset_index(names="Model")
    metrics_df["Model"] = metrics_df["Model"].str.replace("_", " ")

    best_acc_model_key, best_acc = get_best_model_by_metric(bundle, "accuracy")
    best_acc_model_name = best_acc_model_key.replace("_", " ")
    acc_pct = best_acc * 100.0

    st.markdown(
        f"""
        <div style="
            background: #12232c;
            border: 1px solid #263b45;
            border-left: 4px solid #43d6c4;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 20px;
        ">
            <div style="color: #91a8b0; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">
                Highest Accuracy Model
            </div>
            <div style="display: flex; align-items: baseline; gap: 14px; margin-top: 4px;">
                <span style="font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; color: #43d6c4;">
                    {acc_pct:.1f}%
                </span>
                <span style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 600; color: #e8f1f2;">
                    {best_acc_model_name}
                </span>
            </div>
            <div style="color: #91a8b0; font-size: 0.88rem; margin-top: 2px;">
                Highest accuracy on the held-out test dataset
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Model Evaluation Metrics")
    st.dataframe(
        metrics_df.style.format({col: "{:.3f}" for col in metrics_df.columns if col != "Model"}),
        width="stretch", hide_index=True,
    )

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.subheader("Metrics Comparison")
        plot_data = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(
            plot_data, x="Metric", y="Score", color="Model", barmode="group",
            color_discrete_sequence=["#43d6c4", "#ff796f", "#f0bc61", "#91b8ff"]
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
    with col_b:
        best_model = bundle.get("best_model", "Random_Forest")
        st.subheader(f"{best_model.replace('_', ' ')} Confusion Matrix")
        matrix = bundle["confusion_matrices"][best_model]
        fig = px.imshow(
            matrix, text_auto=True, x=["Legit (0)", "Fraud (1)"], y=["Legit (0)", "Fraud (1)"],
            color_continuous_scale="Teal", labels=dict(x="Predicted", y="Actual")
        )
        st.plotly_chart(chart_layout(fig), width="stretch")

    importance = bundle.get("feature_importance", [])
    if importance:
        st.subheader("Feature Importance · Selected Model")
        important_df = pd.DataFrame(importance, columns=["Feature", "Importance"]).head(12).sort_values("Importance")
        fig = px.bar(important_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#43d6c4"])
        st.plotly_chart(chart_layout(fig, 370), width="stretch")


def analytics_page(data: pd.DataFrame) -> None:
    page_heading("Transaction Analytics", "Explore transaction amounts, foreign activity, and days since last transaction.")
    maximum = float(data["amount"].quantile(0.99))
    selected = st.slider("Maximum amount shown (99th percentile cap)", min_value=0.0,
                         max_value=max(maximum, 1.0), value=max(maximum, 1.0))
    view = data[data["amount"] <= selected].copy()
    view["Label"] = view[TARGET_COLUMN].map({0: "Legitimate (0)", 1: "Fraud (1)"})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Amount Distribution by Class")
        fig = px.box(view, x="Label", y="amount", color="Label", points=False,
                     color_discrete_map={"Legitimate (0)": "#43d6c4", "Fraud (1)": "#ff796f"})
        st.plotly_chart(chart_layout(fig), width="stretch")
    with c2:
        st.subheader("Amount vs. Days Since Last Tx")
        sample_size = min(2500, len(view))
        fig = px.scatter(
            view.sample(sample_size, random_state=7), x="days_since_last_transaction", y="amount",
            color="Label", opacity=0.55,
            color_discrete_map={"Legitimate (0)": "#43d6c4", "Fraud (1)": "#ff796f"}
        )
        st.plotly_chart(chart_layout(fig), width="stretch")

    st.download_button(
        "Download Filtered Transactions CSV",
        view.drop(columns="Label").to_csv(index=False),
        file_name="filtered_transactions.csv", mime="text/csv", icon=":material/download:"
    )


def about_page(bundle: dict | None = None) -> None:
    page_heading("Project Notes & Data Leakage Analysis", "Detailed breakdown of data leakage prevention, modeling pipeline, and metrics.")
    st.markdown("""
    ### Data Leakage Analysis & 99% Accuracy Explanation

    **Why did the previous model achieve ~99% accuracy?**
    1. **Preprocessing / Scaling Leakage:** In previous implementations, scalers (`StandardScaler`) or categorical encoders were fit on the entire dataset prior to train-test splitting. This leaked global mean, standard deviation, and categorical distribution information from the test set into training algorithms.
    2. **Global Aggregation Leakage:** Features like `transaction_amount_delta` or historical averages computed across all records leak future transaction patterns into past samples when calculated across full datasets without strict temporal boundaries.
    3. **Synthetic Separability:** Synthetic fraud benchmarks add synthetic offsets to feature subsets, enabling tree ensembles to find artificial decision boundaries with near 100% precision and recall.

    ---

    ### Leak-Free Machine Learning Pipeline

    Our updated codebase implements strict controls to prevent data leakage:
    - **Stratified Train-Test Split:** Dataset is split 75% train / 25% test stratified on `is_fraud`.
    - **Scikit-Learn Pipelines:** All scaling (`StandardScaler`) and encoding (`OneHotEncoder`) are encapsulated inside a Scikit-Learn `Pipeline`. Preprocessors are fitted **STRICTLY on training data (`x_train`)**.
    - **Evaluation Metrics:** Evaluated exclusively on unseen test data (`x_test`), reporting Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and 2x2 Confusion Matrices alongside Accuracy.
    """)


def main() -> None:
    path = get_active_dataset_path()

    with st.sidebar:
        st.markdown("## ◈ Risk Desk")
        st.caption("CREDIT CARD FRAUD MONITOR")
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("DATASET & MODEL STATUS")
        if path.exists():
            st.caption(f"CSV: {path.name}")
            st.caption(f"Size: {path.stat().st_size / 1024:.1f} KB")

    if page == "Data & Models":
        data_models_page()
        return

    if not path.exists() or not MODEL_PATH.exists():
        page_heading("Credit Card Fraud Detection", "Machine learning · transaction monitoring · model evaluation")
        setup_panel()
        if page == "About":
            about_page()
        return

    data = read_dataset(str(path), path.stat().st_mtime)
    bundle = read_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)

    if page == "Dashboard":
        dashboard(data, bundle)
    elif page == "Predict Transaction":
        prediction_page(bundle)
    elif page == "Model Comparison":
        comparison_page(bundle)
    elif page == "Analytics":
        analytics_page(data)
    else:
        about_page(bundle)


if __name__ == "__main__":
    main()
