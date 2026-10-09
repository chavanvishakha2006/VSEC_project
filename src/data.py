from __future__ import annotations

import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "year", "month", "day", "time", "card_network", "payment_method",
    "category", "mcc", "amount", "is_foreign", "days_since_last_transaction",
    "transaction_amount_delta"
]
TARGET_COLUMN = "is_fraud"

NUMERICAL_FEATURES = [
    "year", "month", "day", "hour", "minute", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "mcc", "amount", "is_foreign",
    "days_since_last_transaction", "transaction_amount_delta"
]
CATEGORICAL_FEATURES = ["card_network", "payment_method", "category"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def parse_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, minute, and cyclical time features from time string HH:MM or date columns."""
    df = frame.copy()
    if "time" in df.columns:
        # Format HH:MM
        time_series = df["time"].astype(str).str.strip()
        hours = time_series.str.split(":").str[0].apply(pd.to_numeric, errors="coerce").fillna(12)
        minutes = time_series.str.split(":").str[1].apply(pd.to_numeric, errors="coerce").fillna(0)
    elif "hour" in df.columns:
        hours = pd.to_numeric(df["hour"], errors="coerce").fillna(12)
        minutes = pd.to_numeric(df.get("minute", 0), errors="coerce").fillna(0)
    else:
        hours = pd.Series(12, index=df.index)
        minutes = pd.Series(0, index=df.index)

    df["hour"] = hours.astype(float)
    df["minute"] = minutes.astype(float)

    months = pd.to_numeric(df.get("month", 1), errors="coerce").fillna(1).astype(float)

    hour_angle = 2 * np.pi * df["hour"] / 24.0
    month_angle = 2 * np.pi * (months - 1) / 12.0

    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)

    return df


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, preprocess, and standardize transaction DataFrame for model input."""
    if frame.empty:
        raise ValueError("Transaction dataset is empty")

    df = frame.copy()

    # Standardize target column to 'is_fraud'
    if "is_fraud" in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df["is_fraud"], errors="coerce")
    elif "IsFraud" in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df["IsFraud"], errors="coerce")
    elif "Class" in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df["Class"], errors="coerce")
    else:
        raise ValueError("Dataset missing target column ('is_fraud', 'IsFraud', or 'Class')")

    if df[TARGET_COLUMN].isna().any() or not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target column 'is_fraud' must contain non-null binary labels (0 or 1)")

    # Extract time features
    df = parse_time_features(df)

    # Ensure numerical features are numeric
    num_cols = ["year", "month", "day", "mcc", "amount", "is_foreign", "days_since_last_transaction", "transaction_amount_delta"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Ensure categorical features are strings
    cat_cols = ["card_network", "payment_method", "category"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        else:
            df[col] = "Unknown"

    validate_transactions(df)
    return df


def validate_transactions(frame: pd.DataFrame) -> None:
    """Ensure required features and target are present and valid."""
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Transaction dataset is empty")

    features = frame[NUMERICAL_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Numerical features must contain finite numeric values")


def generate_transactions(rows: int = 4000, seed: int = 42, fraud_rate: float = 0.15) -> pd.DataFrame:
    """Generate reproducible sample transactions using credit_card_fraud_full_training_dataset schema."""
    rng = np.random.default_rng(seed)
    fraud_count = max(1, int(round(rows * fraud_rate)))

    years = rng.choice([2020, 2021, 2022, 2023], size=rows)
    months = rng.integers(1, 13, size=rows)
    days = rng.integers(1, 29, size=rows)
    hours = rng.integers(0, 24, size=rows)
    minutes = rng.integers(0, 60, size=rows)

    card_networks = rng.choice(["Visa", "Mastercard", "American Express", "JCB", "Discover"], size=rows)
    payment_methods = rng.choice(["Chip Transaction", "Online Transaction", "Contactless Payment", "Swipe Transaction"], size=rows)
    categories = rng.choice(["gas_transport", "online_shopping", "restaurant", "health", "retail", "services", "grocery_pos", "travel"], size=rows)
    mccs = rng.choice([5542, 4121, 5964, 5814, 5812, 5912, 5651, 5311, 7922, 8041], size=rows)

    amounts = rng.lognormal(mean=4.2, sigma=0.9, size=rows)
    is_foreign = rng.choice([0, 1], size=rows, p=[0.85, 0.15])
    days_since_last = rng.exponential(scale=35.0, size=rows)
    deltas = rng.normal(loc=10.0, scale=120.0, size=rows)

    labels = np.zeros(rows, dtype=int)
    fraud_idx = rng.choice(rows, size=fraud_count, replace=False)
    labels[fraud_idx] = 1

    amounts[fraud_idx] *= rng.uniform(1.8, 4.5, fraud_count)
    is_foreign[fraud_idx] = rng.choice([0, 1], size=fraud_count, p=[0.3, 0.7])
    deltas[fraud_idx] += rng.uniform(100.0, 600.0, fraud_count)

    time_strings = [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]

    frame = pd.DataFrame({
        "year": years,
        "month": months,
        "day": days,
        "time": time_strings,
        "card_network": card_networks,
        "payment_method": payment_methods,
        "category": categories,
        "mcc": mccs,
        "amount": amounts,
        "is_foreign": is_foreign,
        "days_since_last_transaction": days_since_last,
        "transaction_amount_delta": deltas,
        "is_fraud": labels
    })

    return normalize_transactions(frame)
