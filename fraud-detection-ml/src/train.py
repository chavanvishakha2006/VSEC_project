"""
Training Module for Credit Card Fraud Detection.

Loads financial dataset, validates structure, performs stratified train/test split,
applies leakage-free scaling, trains multiple imbalance-aware classifiers (Logistic Regression,
Random Forest, XGBoost), evaluates models on fraud metrics (PR-AUC, F1, Recall, Precision),
selects the top performing model, and serializes artifacts to the 'models/' directory.
"""

import os
import json
import argparse
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing import (
    load_dataset,
    validate_dataset,
    split_and_scale_data,
)
from src.evaluate import (
    evaluate_model_performance,
    generate_comparison_table,
)
from src.generate_sample_data import save_synthetic_dataset


def train_and_evaluate_models(
    data_path: str = "data/creditcard.csv",
    output_dir: str = "models",
    generate_sample_if_missing: bool = True
) -> Dict[str, Any]:
    """
    Main training workflow pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # 1. Load Dataset
    print("=" * 70)
    print(" 💳 CREDIT CARD FRAUD DETECTION - ML TRAINING PIPELINE ")
    print("=" * 70)

    try:
        df, is_synthetic, status_msg = load_dataset(data_path)
        print(f"\n[INFO] {status_msg}")
    except FileNotFoundError as e:
        if generate_sample_if_missing:
            print(f"\n[NOTICE] Real dataset not found at '{data_path}'.")
            print("[NOTICE] Generating synthetic sample dataset for code testing/validation ONLY...")
            fallback_file = save_synthetic_dataset("data/sample_creditcard.csv", n_samples=5000)
            df, is_synthetic, status_msg = load_dataset(fallback_file)
            print(f"[INFO] {status_msg}")
        else:
            raise e

    if is_synthetic:
        print("\n" + "!" * 70)
        print(" WARNING: TRAINING IS OPERATING ON SYNTHETIC SAMPLE DATA.")
        print(" Metrics produced here are for software validation ONLY and MUST NOT")
        print(" be cited as real-world model accuracy or performance.")
        print("!" * 70 + "\n")

    # 2. Validate Dataset & EDA Stats
    val_stats = validate_dataset(df)
    print(f"\n📊 DATASET STATISTICS:")
    print(f"  • Total Transactions   : {val_stats['total_rows']:,}")
    print(f"  • Legitimate Count (0) : {val_stats['legitimate_count']:,}")
    print(f"  • Fraudulent Count (1) : {val_stats['fraud_count']:,}")
    print(f"  • Fraud Percentage     : {val_stats['fraud_percentage']:.4f}%")
    print(f"  • Missing Values       : {val_stats['total_missing_values']}")
    print(f"  • Duplicate Rows       : {val_stats['duplicate_rows']:,}")

    # 3. Stratified Split and Leakage-Free Scaling
    print("\n⚙️ PREPROCESSING & SPLITTING DATA:")
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale_data(df)
    print(f"  • Train Set Shape : {X_train.shape[0]:,} samples")
    print(f"  • Test Set Shape  : {X_test.shape[0]:,} samples")
    print(f"  • Features Scaled : Time, Amount (RobustScaler fitted strictly on Train)")

    # Compute scale_pos_weight for XGBoost
    train_neg = (y_train == 0).sum()
    train_pos = (y_train == 1).sum()
    scale_pos_weight = (train_neg / train_pos) if train_pos > 0 else 1.0

    # 4. Instantiate Models with Imbalance Handling
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    # 5. Train & Evaluate Models
    print("\n🚀 TRAINING & EVALUATING MODELS:")
    all_results = {}

    for name, model in models.items():
        print(f"\n  • Training {name}...")
        model.fit(X_train, y_train)
        results = evaluate_model_performance(model, X_test, y_test, model_name=name)
        all_results[name] = results
        print(
            f"    -> Precision: {results['precision']:.4f} | "
            f"Recall: {results['recall']:.4f} | "
            f"F1: {results['f1_score']:.4f} | "
            f"ROC-AUC: {results['roc_auc']:.4f} | "
            f"PR-AUC: {results['pr_auc']:.4f}"
        )

    # 6. Model Comparison Table
    comparison_df = generate_comparison_table(all_results)
    print("\n" + "=" * 70)
    print(" 🏆 MODEL COMPARISON TABLE ")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # 7. Select Best Model (Highest PR-AUC primary, F1 secondary)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    best_metrics = all_results[best_model_name]

    print(f"\n🌟 Selected Best Model: '{best_model_name}' (PR-AUC = {best_metrics['pr_auc']:.4f})")

    # 8. Save Model, Scaler, and Metrics
    model_path = os.path.join(output_dir, "fraud_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    metrics_path = os.path.join(output_dir, "metrics.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save summary payload to metrics.json
    export_payload = {
        "is_synthetic": is_synthetic,
        "selected_model": best_model_name,
        "feature_names": feature_names,
        "dataset_stats": val_stats,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "models_performance": all_results,
    }

    with open(metrics_path, "w") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\n💾 ARTIFACTS SAVED SUCCESSFULLY:")
    print(f"  • Model  : '{model_path}'")
    print(f"  • Scaler : '{scaler_path}'")
    print(f"  • Metrics: '{metrics_path}'")
    print("=" * 70 + "\n")

    return export_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Credit Card Fraud Detection Models")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv", help="Path to creditcard.csv")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--no-sample-fallback", action="store_true", help="Disable synthetic fallback if real data is missing")

    args = parser.parse_args()
    train_and_evaluate_models(
        data_path=args.data_path,
        output_dir=args.output_dir,
        generate_sample_if_missing=not args.no_sample_fallback
    )
