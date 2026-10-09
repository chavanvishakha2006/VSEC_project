"""
Data Preprocessing Module for Credit Card Fraud Detection.

Handles:
- Dataset loading
- Schema validation
- Missing-value checks
- Duplicate checks
- Stratified train/test splitting
- Leakage-free feature scaling
"""

import os
from typing import Tuple, Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


REQUIRED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

TARGET_COLUMN = "Class"


def load_dataset(
    data_path: str = "data/creditcard.csv",
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Load the credit card dataset.

    Returns:
        Tuple containing:
        - DataFrame
        - Boolean indicating whether the dataset is synthetic
        - Status message
    """

    # Check the requested dataset path
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)

            # Sample dataset is used only for testing/validation.
            is_synthetic = (
                os.path.basename(data_path) == "sample_creditcard.csv"
            )

            if is_synthetic:
                message = (
                    f"Loaded synthetic sample dataset from "
                    f"'{data_path}' ({len(df):,} rows) "
                    "FOR CODE TESTING ONLY."
                )
            else:
                message = (
                    f"Successfully loaded real dataset from "
                    f"'{data_path}' ({len(df):,} rows)."
                )

            return df, is_synthetic, message

        except Exception as e:
            raise ValueError(
                f"Error reading dataset at '{data_path}': {str(e)}"
            ) from e

    # If the real dataset does not exist, check the synthetic dataset.
    fallback_path = "data/sample_creditcard.csv"

    if os.path.exists(fallback_path):
        try:
            df = pd.read_csv(fallback_path)

            message = (
                f"[NOTICE] Primary dataset '{data_path}' not found. "
                f"Loaded synthetic sample dataset from "
                f"'{fallback_path}' ({len(df):,} rows) "
                "FOR CODE TESTING ONLY."
            )

            return df, True, message

        except Exception as e:
            raise ValueError(
                f"Error reading sample dataset at "
                f"'{fallback_path}': {str(e)}"
            ) from e

    # Neither dataset exists.
    error_message = (
        f"Dataset file not found at '{data_path}'.\n"
        "SETUP INSTRUCTIONS:\n"
        "1. Download 'creditcard.csv' from Kaggle.\n"
        "2. Place the file at: data/creditcard.csv\n"
        "3. For code testing only, run:\n"
        "   python src/generate_sample_data.py"
    )

    raise FileNotFoundError(error_message)


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform structural and data-quality validation.

    Returns:
        Dictionary containing dataset statistics.
    """

    # Check required columns.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Check missing values.
    missing_values_count = (
        df[REQUIRED_COLUMNS]
        .isnull()
        .sum()
        .to_dict()
    )

    total_missing_values = sum(missing_values_count.values())

    # Check duplicate rows.
    duplicate_rows_count = int(
        df.duplicated(subset=REQUIRED_COLUMNS).sum()
    )

    # Dataset statistics.
    total_rows = len(df)

    class_counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .to_dict()
    )

    legitimate_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))

    fraud_percentage = (
        fraud_count / total_rows * 100
        if total_rows > 0
        else 0.0
    )

    validation_summary = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "total_missing_values": int(total_missing_values),
        "duplicate_rows": duplicate_rows_count,
        "legitimate_count": legitimate_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_percentage, 4),
        "columns_present": list(df.columns),
    }

    return validation_summary


def split_and_scale_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    RobustScaler,
    List[str],
]:
    """
    Split the dataset using stratification and apply leakage-free scaling.

    Time and Amount are scaled using RobustScaler.

    IMPORTANT:
        The scaler is fitted ONLY on X_train.
        X_test is transformed using the already-fitted scaler.
    """

    # Remove rows containing missing required values.
    clean_df = df.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    # Separate features and target.
    X = clean_df[FEATURE_COLUMNS].copy()
    y = clean_df[TARGET_COLUMN].astype(int).copy()

    # Stratified train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Only Time and Amount require scaling.
    columns_to_scale = ["Time", "Amount"]

    scaler = RobustScaler()

    # IMPORTANT: Fit ONLY on training data.
    scaler.fit(X_train[columns_to_scale])

    # Create copies before transformation.
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Transform training data.
    X_train_scaled[columns_to_scale] = scaler.transform(
        X_train[columns_to_scale]
    )

    # Transform testing data using the same training-fitted scaler.
    X_test_scaled[columns_to_scale] = scaler.transform(
        X_test[columns_to_scale]
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        FEATURE_COLUMNS,
    )