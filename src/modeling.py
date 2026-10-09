from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    normalize_transactions,
    validate_transactions,
)


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer fitting StandardScaler on numeric features and OneHotEncoder on categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def train_models(
    frame: pd.DataFrame,
    output_path: str | Path,
    seed: int = 42,
    data_note: str = "Training dataset: credit_card_fraud_full_training_dataset.csv",
) -> dict:
    df = normalize_transactions(frame)
    validate_transactions(df)

    y = df[TARGET_COLUMN].astype(int)
    X = df[FEATURE_COLUMNS]

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Training requires at least two examples in each class")

    # Stratified Train-Test Split (Zero Data Leakage)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    scale_pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    candidates = {
        "Logistic_Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=seed))
        ]),
        "Linear_SVM": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed),
                method="sigmoid", cv=3
            ))
        ]),
        "Random_Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=160, min_samples_leaf=2, class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed
            ))
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(
                n_estimators=160, max_depth=4, learning_rate=0.06, subsample=0.9,
                colsample_bytree=0.9, scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=seed
            ))
        ]),
    }

    fitted_pipelines = {}
    metrics = {}
    matrices = {}

    for name, pipeline in candidates.items():
        # Preprocessing and model training fitted STRICTLY on x_train
        pipeline.fit(x_train, y_train)

        # Predict probabilities and labels on unseen test set (x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        fitted_pipelines[name] = pipeline
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
        }
        matrices[name] = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()

    best_name = max(metrics, key=lambda name: metrics[name]["pr_auc"])
    best_pipeline = fitted_pipelines[best_name]

    # Calculate feature importances for best model
    preproc = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]

    # Get feature names after one-hot encoding
    cat_encoder = preproc.named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERICAL_FEATURES + encoded_cat_names

    raw_importance = getattr(classifier, "feature_importances_", None)
    if raw_importance is None:
        if hasattr(classifier, "coef_"):
            raw_importance = np.abs(classifier.coef_[0])
        elif hasattr(classifier, "calibrated_classifiers_"):
            raw_importance = np.abs(classifier.calibrated_classifiers_[0].estimator.coef_[0])

    if raw_importance is not None and len(raw_importance) == len(all_feature_names):
        feature_imp_map = dict(zip(all_feature_names, raw_importance))
        # Group one-hot categorical features back to original columns for UI display
        grouped_importance = {col: 0.0 for col in FEATURE_COLUMNS}
        for name, val in feature_imp_map.items():
            base_name = name.split("_")[0] if "_" in name and name.split("_")[0] in CATEGORICAL_FEATURES else name
            if base_name in grouped_importance:
                grouped_importance[base_name] += val
            else:
                grouped_importance[name] = val
        ranked = sorted(grouped_importance.items(), key=lambda item: item[1], reverse=True)
    else:
        ranked = [(col, 1.0 / len(FEATURE_COLUMNS)) for col in FEATURE_COLUMNS]

    bundle = {
        "models": fitted_pipelines,
        "metrics": metrics,
        "confusion_matrices": matrices,
        "best_model": best_name,
        "feature_importance": ranked,
        "features": FEATURE_COLUMNS,
        "threshold": 0.5,
        "input_schema": "credit_card_fraud_full_training_dataset",
        "data_note": data_note,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return bundle
