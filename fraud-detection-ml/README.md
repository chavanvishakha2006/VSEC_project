# 💳 AI-Powered Credit Card & ATM Fraud Detection System

> **An End-to-End Machine Learning System for Financial Transaction Fraud Detection**  
> *Final-Year College Project & Viva-Ready Production Codebase*

---

## 📌 Project Overview

Financial credit card and ATM fraud causes billions of dollars in losses annually. Detecting fraudulent transactions in real time is a critical challenge due to extreme **class imbalance**: out of 284,807 transactions, only **492 (0.172%)** are fraudulent.

This project delivers a complete, production-ready machine learning system that ingests financial transaction data, performs leakage-free preprocessing, trains multiple imbalance-aware classifiers (Logistic Regression, Random Forest, XGBoost), evaluates them using fraud-specific metrics (PR-AUC, F1-Score, Precision, Recall, ROC-AUC), and serves predictions via an interactive **Streamlit Web Application** backed by **Docker** containerization and **GitHub Actions** CI/CD.

---

## 🎯 Objectives

1. **Leakage-Free Preprocessing**: Implement stratified dataset splitting and scale numerical features (`Time` and `Amount`) using transformers fitted strictly on training data.
2. **Imbalance-Aware Model Training**: Address extreme class imbalance using cost-sensitive learning (`class_weight='balanced'` and `scale_pos_weight`) without fabricating data.
3. **Rigorous Metric Evaluation**: Evaluate models using PR-AUC, F1-Score, Recall, Precision, and ROC-AUC rather than misleading accuracy.
4. **Interactive Dashboard**: Build a Streamlit application featuring live transaction risk scoring, preset scenario test inputs, safety disclaimers, and metric comparison charts.
5. **Production Engineering**: Package the project with Pytest unit testing, Docker multi-stage builds, and GitHub Actions CI.

---

## 📊 Dataset Description

The system uses the **Credit Card Fraud Detection Dataset** (available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).

### Dataset Schema:
- **`Time`**: Seconds elapsed between the transaction and the first transaction in the dataset.
- **`V1` through `V28`**: Principal components derived via Principal Component Analysis (PCA) to protect user anonymity.
- **`Amount`**: Transaction monetary value.
- **`Class`**: Target label (`0` = Legitimate, `1` = Fraudulent).

### Class Distribution:
| Class | Count | Percentage |
| :--- | :--- | :--- |
| **0 (Legitimate)** | 284,315 | 99.827% |
| **1 (Fraudulent)** | 492 | 0.173% |
| **Total** | 284,807 | 100.000% |

> [!IMPORTANT]
> **Dataset Setup**: Download `creditcard.csv` from Kaggle and place it at `data/creditcard.csv`.  
> If the file is absent, the system provides a clear setup message and offers a synthetic data generator script (`src/generate_sample_data.py`) **strictly for local code validation and testing**.

---

## 🏗️ Project Architecture & Structure

```
fraud-detection-ml/
│
├── data/
│   ├── creditcard.csv            # Kaggle dataset (User-provided)
│   └── sample_creditcard.csv     # Synthetic sample for testing only
│
├── notebooks/
│   └── fraud_detection.ipynb     # Exploratory Data Analysis & Walkthrough
│
├── src/
│   ├── __init__.py               # Package marker
│   ├── generate_sample_data.py   # Synthetic test dataset generator
│   ├── preprocessing.py          # Validation, stratified split & RobustScaling
│   ├── train.py                  # Training pipeline & artifact serializer
│   ├── evaluate.py               # PR-AUC, ROC-AUC, F1 metric computations
│   └── predict.py                # Reusable prediction & scaling interface
│
├── models/
│   ├── fraud_model.pkl           # Trained model binary
│   ├── scaler.pkl                # Fitted RobustScaler binary
│   └── metrics.json              # Evaluation summary payload
│
├── app/
│   └── app.py                    # Streamlit web application
│
├── tests/
│   └── test_model.py             # Pytest unit testing suite
│
├── requirements.txt              # Dependency specifications
├── Dockerfile                    # Containerization script
├── .dockerignore                 # Docker ignore patterns
├── .gitignore                    # Git ignore rules
├── README.md                     # Documentation & Viva Q&A Guide
└── .github/
    └── workflows/
        └── ci.yml                # GitHub Actions CI workflow
```

---

## ⚙️ Local Setup & Execution Guide

### 1. Environment Setup

```powershell
# Navigate into project directory
cd fraud-detection-ml

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\activate

# Activate virtual environment (Linux / macOS)
# source .venv/bin/activate

# Upgrade pip and install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Download `creditcard.csv` from Kaggle and place it in the `data/` directory:
```
fraud-detection-ml/data/creditcard.csv
```
*If you want to run quick code validation before downloading the full dataset, execute:*
```powershell
python src/generate_sample_data.py
```

### 3. Model Training
Run the machine learning training pipeline:
```powershell
python src/train.py
```
This script will:
1. Load and validate the dataset.
2. Perform stratified train/test split.
3. Fit `RobustScaler` strictly on training data.
4. Train Logistic Regression, Random Forest, and XGBoost models.
5. Display a model comparison matrix.
6. Save `fraud_model.pkl`, `scaler.pkl`, and `metrics.json` into `models/`.

### 4. Streamlit Web Application
Launch the interactive dashboard:
```powershell
streamlit run app/app.py
```
Open your browser at `http://localhost:8501` to view:
- **📊 Executive Dashboard**: Dataset KPIs, class imbalance charts, amount distributions.
- **🔍 Fraud Prediction**: Preset scenario inputs, custom feature scoring, fraud probability meters, and safety notes.
- **⚖️ Model Comparison**: Multi-model evaluation tables and metric comparison bar charts.
- **📈 Model Performance**: Confusion matrix heatmaps, classification reports, ROC curves, and PR curves.

### 5. Unit Testing
Run the Pytest test suite:
```powershell
pytest tests/ -v
```

---

## 🐳 Docker Deployment

Build and run the application inside a container:

```powershell
# Build Docker image
docker build -t fraud-detection .

# Run Docker container
docker run -p 8501:8501 fraud-detection
```
Access the application at `http://localhost:8501`.

---

## 🔄 GitHub Actions CI/CD Pipeline

The `.github/workflows/ci.yml` pipeline triggers automatically on `push` and `pull_request` to `main`:
1. Sets up Python 3.11.
2. Installs required packages.
3. Generates a synthetic test dataset for code validation.
4. Runs `pytest` unit tests.
5. Validates `python src/train.py`.
6. Builds the Docker container image.

---

## 🧠 Model Evaluation & Imbalance Metrics

In fraud detection, **Accuracy is a misleading metric**. A dummy classifier that predicts every transaction is legitimate achieves **99.83% Accuracy**, but misses **100% of frauds**.

### Evaluated Metrics:
- **PR-AUC (Precision-Recall Area Under Curve)**: Primary metric for severe imbalance. Evaluates precision across all recall thresholds.
- **Recall (Sensitivity)**: Percentage of actual fraudulent transactions detected ($\frac{TP}{TP + FN}$). High recall ensures minimal missed frauds.
- **Precision**: Percentage of predicted frauds that are genuinely fraudulent ($\frac{TP}{TP + FP}$). High precision minimizes false alarms.
- **F1-Score**: Harmonic mean of Precision and Recall ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$).
- **ROC-AUC**: Trade-off between True Positive Rate and False Positive Rate.

---

## 🎓 College Viva Q&A Guide

### Q1: Why is Accuracy an inappropriate metric for Credit Card Fraud Detection?
**Answer**: Because of the 99.83% to 0.17% class imbalance. A trivial model that classifies all transactions as legitimate achieves 99.83% accuracy while failing to detect a single fraudulent transaction. Metrics such as Precision, Recall, F1-Score, and PR-AUC measure positive class performance directly.

### Q2: What is Data Leakage, and how does this project prevent it?
**Answer**: Data Leakage occurs when information from outside the training dataset (e.g. test set statistics) is used to fit transformers or models. In this project, `split_and_scale_data()` performs stratified train/test splitting **before** fitting `RobustScaler`. The scaler is fitted strictly on `X_train` and then used to transform `X_train` and `X_test`.

### Q3: Why use PR-AUC instead of ROC-AUC for imbalanced datasets?
**Answer**: ROC-AUC uses False Positive Rate ($\frac{FP}{FP + TN}$). When $TN$ (Legitimate transactions) is extremely large (~284,000), even a large number of false positives yields a small $FPR$, making ROC-AUC look deceptively high (~0.98). PR-AUC plots Precision ($\frac{TP}{TP + FP}$) against Recall ($\frac{TP}{TP + FN}$), isolating the positive (fraud) class without being inflated by true negatives.

### Q4: How do the models handle class imbalance?
**Answer**:
- **Logistic Regression**: Uses `class_weight='balanced'`, which scales class loss weights inversely proportional to class frequencies ($w_j = \frac{N}{2 \times N_j}$).
- **Random Forest**: Uses `class_weight='balanced_subsample'`, calculating class weights dynamically for each bootstrap sample tree.
- **XGBoost**: Uses `scale_pos_weight = \frac{\text{Count(Negative)}}{\text{Count(Positive)}}`, scaling gradient updates for positive instances.

### Q5: What are the V1 through V28 features?
**Answer**: They are numerical features resulting from a Principal Component Analysis (PCA) transformation applied by the dataset providers to anonymize sensitive customer card details and transaction metadata.

### Q6: Why scale `Time` and `Amount` with `RobustScaler`?
**Answer**: Unlike V1–V28, `Time` and `Amount` are in raw units. `Amount` contains extreme monetary outliers. `RobustScaler` uses median and Interquartile Range ($IQR = Q_3 - Q_1$), making scaling resilient to extreme values.

### Q7: Why class-weighting over blind SMOTE synthetic oversampling?
**Answer**: SMOTE creates synthetic samples by interpolating between k-nearest fraud neighbors, which can introduce artificial noise in high-dimensional PCA spaces and increase false positive rates. Cost-sensitive class weighting adjusts loss gradients directly during optimization without altering dataset boundary integrity.

### Q8: How is code modularity enforced between training and inference?
**Answer**: `src/predict.py` imports `RobustScaler` logic and calls `preprocess_transaction()`. Streamlit's `app.py` passes user inputs directly to `predict_transaction()`, ensuring inference preprocessing is identical to training.

### Q9: How does Docker containerization support deployment?
**Answer**: Docker packages the application code, dependencies (`requirements.txt`), Python runtime, and Streamlit server into an isolated container image, ensuring identical behavior across local machines, servers, and cloud environments.

### Q10: How does the CI workflow run without the Kaggle dataset?
**Answer**: GitHub Actions executes `src/generate_sample_data.py` to create a lightweight synthetic dataset strictly for code structure validation and unit test execution. The actual Kaggle dataset remains uncommitted as per `.gitignore`.

---

## 🔮 Future Improvements & Limitations

- **Concept Drift Monitoring**: Implement drift detection (e.g., Evidently AI) to monitor model decay over time.
- **Real-Time Kafka Streaming**: Ingest transaction streams via Apache Kafka for sub-second inference.
- **SHAP Explainability**: Integrate SHAP (SHapley Additive exPlanations) values in Streamlit to explain feature contributions for flagged transactions.
