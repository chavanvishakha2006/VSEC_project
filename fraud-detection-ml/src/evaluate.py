"""
Model Evaluation Module for Credit Card Fraud Detection.

Calculates comprehensive fraud detection metrics: Accuracy, Precision, Recall, F1-Score,
ROC-AUC, PR-AUC, Confusion Matrix, Classification Report, and ROC / PR Curve arrays.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def evaluate_model_performance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Classifier"
) -> Dict[str, Any]:
    """
    Evaluates a trained model on test data and returns all relevant metrics.
    
    Returns:
        Dict containing scalar metrics, confusion matrix, report, and curve data.
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities for positive class (Fraud = 1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        y_prob = y_pred.astype(float)

    # Core Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_test, y_prob)
    except Exception:
        pr_auc = 0.0

    # Detailed Matrices & Curves
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    prec_pts, rec_pts, pr_thresholds = precision_recall_curve(y_test, y_prob)

    metrics_result = {
        "model_name": model_name,
        "accuracy": float(round(accuracy, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1_score": float(round(f1, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "pr_auc": float(round(pr_auc, 4)),
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": prec_pts.tolist(),
            "recall": rec_pts.tolist(),
        },
    }

    return metrics_result


def generate_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Creates a model comparison pandas DataFrame from evaluation metrics dict.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Model": model_name,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "Accuracy": metrics["accuracy"],
        })
    df_comp = pd.DataFrame(rows)
    # Sort by PR-AUC descending, then F1 descending
    df_comp = df_comp.sort_values(by=["PR-AUC", "F1"], ascending=False).reset_index(drop=True)
    return df_comp
