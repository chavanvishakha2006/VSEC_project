import pytest

import numpy as np
import pandas as pd

from src.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    generate_transactions,
    normalize_transactions,
    validate_transactions,
)
from src.modeling import train_models
from app import sample_transactions_by_class


def test_synthetic_data_is_reproducible_and_has_expected_schema():
    first = generate_transactions(rows=1000, seed=42)
    second = generate_transactions(rows=1000, seed=42)
    assert first.equals(second)
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(first.columns)
    assert first[TARGET_COLUMN].isin([0, 1]).all()


def test_data_validation_rejects_missing_features():
    frame = generate_transactions(rows=500)
    with pytest.raises(ValueError, match="Missing required"):
        validate_transactions(frame.drop(columns="amount"))


def test_normalize_transactions_parses_time_and_target():
    raw = pd.DataFrame({
        "year": [2022, 2023],
        "month": [5, 11],
        "day": [12, 28],
        "time": ["14:30", "09:15"],
        "card_network": ["Visa", "Mastercard"],
        "payment_method": ["Chip Transaction", "Online Transaction"],
        "category": ["grocery_pos", "online_shopping"],
        "mcc": [5411, 5964],
        "amount": [45.5, 120.0],
        "is_foreign": [0, 1],
        "days_since_last_transaction": [12.5, 3.2],
        "transaction_amount_delta": [-10.0, 50.0],
        "is_fraud": [0, 1]
    })
    normalized = normalize_transactions(raw)
    assert TARGET_COLUMN in normalized.columns
    assert "hour" in normalized.columns
    assert "minute" in normalized.columns
    assert "hour_sin" in normalized.columns
    assert normalized.loc[0, "hour"] == 14.0
    assert normalized.loc[0, "minute"] == 30.0


def test_training_saves_four_leak_free_pipelines_and_metrics(tmp_path):
    frame = generate_transactions(rows=1000, seed=29)
    output = tmp_path / "model_bundle.joblib"
    bundle = train_models(frame, output, seed=29)
    assert output.exists()
    assert set(bundle["models"]) == {"Logistic_Regression", "Linear_SVM", "Random_Forest", "XGBoost"}

    # Test single row prediction via pipeline
    single_row = frame[FEATURE_COLUMNS].iloc[:1]
    prob = bundle["models"]["Random_Forest"].predict_proba(single_row)
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0

    # Test evaluation metrics
    for model_name, metric_dict in bundle["metrics"].items():
        assert "accuracy" in metric_dict
        assert "precision" in metric_dict
        assert "recall" in metric_dict
        assert "f1" in metric_dict
        assert "roc_auc" in metric_dict
        assert "pr_auc" in metric_dict

    assert bundle["best_model"] in bundle["models"]


def test_dashboard_sample_includes_both_classes():
    frame = generate_transactions(rows=1000, seed=37)
    sample = sample_transactions_by_class(frame)
    assert set(sample[TARGET_COLUMN].unique()) == {0, 1}
