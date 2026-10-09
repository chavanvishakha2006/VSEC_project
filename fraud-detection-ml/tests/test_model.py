"""
Pytest Unit Test Suite for Credit Card Fraud Detection ML Pipeline.

Verifies data validation, leakage-free scaling, model serialization,
prediction range constraints, and feature count schemas.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.generate_sample_data import generate_synthetic_creditcard_data
from src.preprocessing import (
    validate_dataset,
    split_and_scale_data,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
)
from src.train import train_and_evaluate_models
from src.predict import (
    load_model_and_scaler,
    preprocess_transaction,
    predict_transaction,
)


@pytest.fixture
def sample_dataframe():
    """Fixture supplying synthetic credit card dataframe for fast unit tests."""
    return generate_synthetic_creditcard_data(n_samples=500, random_state=42)


def test_dataset_columns(sample_dataframe):
    """Test that generated/loaded dataset contains all 31 expected columns."""
    assert len(sample_dataframe.columns) == 31
    for col in REQUIRED_COLUMNS:
        assert col in sample_dataframe.columns


def test_dataset_validation(sample_dataframe):
    """Test dataset validation metrics dictionary."""
    val_stats = validate_dataset(sample_dataframe)
    assert val_stats["total_rows"] == 500
    assert val_stats["total_missing_values"] == 0
    assert "legitimate_count" in val_stats
    assert "fraud_count" in val_stats
    assert val_stats["legitimate_count"] + val_stats["fraud_count"] == 500


def test_split_and_scale_data(sample_dataframe):
    """Test stratified splitting and RobustScaler behavior without data leakage."""
    X_train, X_test, y_train, y_test, scaler, feat_names = split_and_scale_data(
        sample_dataframe, test_size=0.2, random_state=42
    )

    # Check split shapes
    assert len(X_train) == 400
    assert len(X_test) == 100
    assert len(y_train) == 400
    assert len(y_test) == 100
    assert feat_names == FEATURE_COLUMNS

    # Check that scaled Time and Amount have non-raw values
    assert not np.array_equal(X_train["Amount"].values, sample_dataframe.iloc[:400]["Amount"].values)


def test_training_pipeline(tmp_path):
    """Test complete training script execution and artifact generation."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_csv = data_dir / "sample_creditcard.csv"
    
    df = generate_synthetic_creditcard_data(n_samples=300, random_state=42)
    df.to_csv(sample_csv, index=False)

    models_dir = tmp_path / "models"
    
    results = train_and_evaluate_models(
        data_path=str(sample_csv),
        output_dir=str(models_dir),
        generate_sample_if_missing=False
    )

    assert results["is_synthetic"] is True
    assert os.path.exists(models_dir / "fraud_model.pkl")
    assert os.path.exists(models_dir / "scaler.pkl")
    assert os.path.exists(models_dir / "metrics.json")

    # Verify metrics content structure
    with open(models_dir / "metrics.json", "r") as f:
        metrics_data = json.load(f)
    assert "selected_model" in metrics_data
    assert "comparison_table" in metrics_data
    assert len(metrics_data["comparison_table"]) == 3


def test_prediction_output_and_probabilities(tmp_path):
    """Test prediction interface probability bounds and schema validation."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_csv = data_dir / "sample_creditcard.csv"
    
    df = generate_synthetic_creditcard_data(n_samples=500, random_state=42)
    df.to_csv(sample_csv, index=False)

    models_dir = tmp_path / "models"
    train_and_evaluate_models(
        data_path=str(sample_csv),
        output_dir=str(models_dir),
        generate_sample_if_missing=False
    )

    model, scaler = load_model_and_scaler(str(models_dir))

    # Test single transaction dict input
    test_dict = {"Time": 50000.0, "Amount": 150.0}
    for i in range(1, 29):
        test_dict[f"V{i}"] = 0.0

    res = predict_transaction(test_dict, model, scaler)

    assert "prediction_class" in res
    assert res["prediction_class"] in [0, 1]
    assert "fraud_probability" in res
    assert "legit_probability" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert 0.0 <= res["legit_probability"] <= 1.0
    assert abs((res["fraud_probability"] + res["legit_probability"]) - 1.0) < 1e-3
    assert "disclaimer" in res
    assert "ML risk prediction" in res["disclaimer"]


def test_invalid_feature_count_raises_error(tmp_path):
    """Test that submitting incorrect feature lengths raises ValueError."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_csv = data_dir / "sample_creditcard.csv"
    df = generate_synthetic_creditcard_data(n_samples=200, random_state=42)
    df.to_csv(sample_csv, index=False)

    models_dir = tmp_path / "models"
    train_and_evaluate_models(
        data_path=str(sample_csv),
        output_dir=str(models_dir),
        generate_sample_if_missing=False
    )

    model, scaler = load_model_and_scaler(str(models_dir))

    # Array with 25 elements instead of 30
    bad_arr = np.random.randn(25)
    with pytest.raises(ValueError, match="Expected 30 features"):
        predict_transaction(bad_arr, model, scaler)
