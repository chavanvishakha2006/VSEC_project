"""
Prediction Interface Module for Credit Card Fraud Detection.

Provides reusable functions to load serialized models, preprocess single or batch
transaction features identically to training, and infer fraud risk probabilities.
"""

import os
from typing import Dict, Any, Union, List, Tuple
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import FEATURE_COLUMNS


def load_model_and_scaler(model_dir: str = "models") -> Tuple[Any, Any]:
    """
    Load saved model and scaler from directory.
    """
    model_path = os.path.join(model_dir, "fraud_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Trained model artifacts not found in '{model_dir}'.\n"
            "Please run 'python src/train.py' first to train and save the model."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def preprocess_transaction(
    raw_input: Union[Dict[str, float], pd.DataFrame, np.ndarray, List[float]],
    scaler: Any
) -> pd.DataFrame:
    """
    Transforms raw transaction inputs to match exact scaling used during training.
    
    Inputs can be:
        - Dict with keys 'Time', 'V1'..'V28', 'Amount'
        - List/Array of length 30
        - Pandas DataFrame with feature columns
    """
    if isinstance(raw_input, dict):
        # Verify feature keys
        missing_keys = [col for col in FEATURE_COLUMNS if col not in raw_input]
        if missing_keys:
            raise ValueError(f"Input dictionary missing required features: {missing_keys}")
        df = pd.DataFrame([raw_input])[FEATURE_COLUMNS]
    elif isinstance(raw_input, (list, np.ndarray)):
        arr = np.array(raw_input)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 30:
            raise ValueError(f"Expected 30 features (Time, V1..V28, Amount), got {arr.shape[1]}")
        df = pd.DataFrame(arr, columns=FEATURE_COLUMNS)
    elif isinstance(raw_input, pd.DataFrame):
        missing_cols = [col for col in FEATURE_COLUMNS if col not in raw_input.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required feature columns: {missing_cols}")
        df = raw_input[FEATURE_COLUMNS].copy()
    else:
        raise TypeError("Input must be a dict, list, numpy array, or pandas DataFrame.")

    # Apply exact RobustScaler transformation to Time and Amount
    scaled_df = df.copy()
    cols_to_scale = ["Time", "Amount"]
    scaled_df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return scaled_df


def predict_transaction(
    raw_input: Union[Dict[str, float], pd.DataFrame, np.ndarray, List[float]],
    model: Any,
    scaler: Any,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Infers fraud probability and risk classification for a financial transaction.
    """
    scaled_df = preprocess_transaction(raw_input, scaler)

    # Calculate probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_df)[0]
        legit_prob = float(probs[0])
        fraud_prob = float(probs[1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(scaled_df)[0])
        fraud_prob = float(1 / (1 + np.exp(-score)))
        legit_prob = 1.0 - fraud_prob
    else:
        pred_val = int(model.predict(scaled_df)[0])
        fraud_prob = 1.0 if pred_val == 1 else 0.0
        legit_prob = 1.0 - fraud_prob

    is_fraud = fraud_prob >= threshold
    label = "⚠️ Potential Fraud" if is_fraud else "✅ Legitimate Transaction"
    status_code = "ALERT" if is_fraud else "SAFE"

    return {
        "prediction_class": 1 if is_fraud else 0,
        "prediction_label": label,
        "status_code": status_code,
        "fraud_probability": round(fraud_prob, 4),
        "legit_probability": round(legit_prob, 4),
        "fraud_percentage": round(fraud_prob * 100, 2),
        "legit_percentage": round(legit_prob * 100, 2),
        "risk_threshold": threshold,
        "disclaimer": "Disclaimer: This result is an ML risk prediction and not a financial guarantee.",
    }
