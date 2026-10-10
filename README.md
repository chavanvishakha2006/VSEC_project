# VSEC Fraud Detection Project

This project is a Streamlit-based fraud detection system for analyzing transaction risk and flagging suspicious activity. It combines a clean ML workflow, interactive dashboarding, and model comparison tools for practical monitoring and decision support.

## Overview

The application helps a team review transaction behavior, identify likely fraud, and compare multiple machine-learning models using a single dataset. It supports:

- Dashboard summaries of fraud trends and operational KPIs
- Manual transaction risk prediction from user-entered data
- Model performance comparison across multiple algorithms
- Analytics for investigation and review
- Dataset/model management via the app UI

## Included functionality

- Synthetic and real transaction dataset support
- Transaction feature normalization and validation
- Model training with class balancing and leakage-safe setup
- Evaluation metrics including accuracy, precision, recall, F1, ROC-AUC, and PR-AUC
- Confusion matrices and feature importance review
- Streamlit pages for dashboard, prediction, analytics, and model comparison

## Tech stack

- Python 3.10+
- Streamlit
- pandas
- scikit-learn
- XGBoost
- Plotly
- joblib

## Project structure

```text
app.py                          Streamlit application
src/
  data.py                       Transaction validation and synthetic data generation
  modeling.py                   ML training, evaluation, and model bundle creation
scripts/
  prepare_demo.py               Script to prepare dataset and train models
models/
  model_bundle.joblib           Trained model bundle (generated locally)
data/
  credit_card_fraud_full_training_dataset.csv
requirements.txt                Python dependencies
tests/
  test_project.py               Validation tests
Dockerfile                      Containerized app setup
render.yaml                     Render deployment configuration
.github/workflows/             CI and deployment automation
README.md                      Project documentation
```

## Dataset and schema

The app expects transaction records that include fields such as:

- `year`
- `month`
- `day`
- `time`
- `card_network`
- `payment_method`
- `category`
- `mcc`
- `amount`
- `is_foreign`
- `days_since_last_transaction`
- `transaction_amount_delta`
- `is_fraud`

The project can generate a synthetic sample automatically if no dataset is found, and it also accepts a CSV that matches the supported fraud-detection schema.

## Local setup

Create a virtual environment and install dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Prepare the dataset and train the ML models:

```powershell
python -m scripts.prepare_demo
```

Start the app:

```powershell
streamlit run app.py
```

## Running tests

```powershell
pytest
```

## Model information

The training pipeline evaluates several models:

- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost

The best model is selected based on PR-AUC, and the trained bundle is saved to `models/model_bundle.joblib`.

## Deployment

The repository includes Docker and Render configuration files for deployment. A Docker image can be built with:

```powershell
docker build -t vsec-fraud-detection .
docker run --rm -p 8501:8501 vsec-fraud-detection
```

Then open:

```text
http://localhost:8501
```

## Notes

- Generated model files and datasets are local artifacts and can be reproduced by running the training script.
- This project is intended for educational and operational demonstration use, not for regulatory or production-only fraud decisioning without further validation.

