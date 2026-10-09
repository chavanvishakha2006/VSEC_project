from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.data import generate_transactions, normalize_transactions
from src.modeling import train_models


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "credit_card_fraud_full_training_dataset.csv"
DOWNLOADS_PATH = Path(r"C:\Users\Vishakha\Downloads\credit_card_fraud_full_training_dataset.csv")
MODEL_PATH = ROOT / "models" / "model_bundle.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit card fraud detection models on training dataset.")
    parser.add_argument("--rows", type=int, default=4000, help="Number of synthetic transactions if dataset missing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input", type=Path, help="Optional dataset CSV path")
    args = parser.parse_args()

    DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = args.input if args.input.is_absolute() else ROOT / args.input
        raw_frame = pd.read_csv(input_path, low_memory=False)
        frame = normalize_transactions(raw_frame)
        data_note = f"Uploaded dataset: {input_path.name} ({len(frame):,} rows)."
    elif DEFAULT_DATA_PATH.exists():
        raw_frame = pd.read_csv(DEFAULT_DATA_PATH, low_memory=False)
        frame = normalize_transactions(raw_frame)
        data_note = f"Training dataset: {DEFAULT_DATA_PATH.name} ({len(frame):,} rows)."
    elif DOWNLOADS_PATH.exists():
        shutil.copy(DOWNLOADS_PATH, DEFAULT_DATA_PATH)
        raw_frame = pd.read_csv(DEFAULT_DATA_PATH, low_memory=False)
        frame = normalize_transactions(raw_frame)
        data_note = f"Training dataset: {DEFAULT_DATA_PATH.name} ({len(frame):,} rows)."
    else:
        frame = generate_transactions(rows=args.rows, seed=args.seed)
        frame.to_csv(DEFAULT_DATA_PATH, index=False)
        raw_frame = frame
        data_note = "Synthetic demonstration data."

    bundle = train_models(raw_frame, MODEL_PATH, seed=args.seed, data_note=data_note)
    best_model = bundle["best_model"]
    best_metrics = bundle["metrics"][best_model]

    print(f"Dataset: {data_note}")
    print(f"Trained models: {', '.join(bundle['models'].keys())}")
    print(f"Selected best model: {best_model}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  F1-Score:  {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {best_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
