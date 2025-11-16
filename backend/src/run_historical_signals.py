# backend/src/run_historical_signals.py

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd


# BASE_DIR is the /src folder
BASE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT is the backend/ folder (one level up)
PROJECT_ROOT = BASE_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "nvda_daily_dataset.csv"
OUT_PATH = PROJECT_ROOT / "data" / "nvda_daily_with_signals.csv"
META_PATH = PROJECT_ROOT / "models" / "nvda_rf_metadata.json"


def load_model_and_meta():
    """Load trained RF model + metadata (feature list, label mapping)."""
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {META_PATH}")

    with META_PATH.open("r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)

    # Resolve model path relative to /models
    model_name = meta.get("model_path", "nvda_rf_signal.pkl")
    model_path = (PROJECT_ROOT / "models" / model_name).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)

    feature_cols = meta["feature_cols"]
    # metadata label_mapping: {"-1": "SELL", "0": "HOLD", "1": "BUY"}
    raw_label_mapping: Dict[str, str] = meta["label_mapping"]

    # int → label, e.g. {-1: "SELL", 0: "HOLD", 1: "BUY"}
    int_to_label = {int(k): v for k, v in raw_label_mapping.items()}
    # label → int, e.g. {"SELL": -1, "HOLD": 0, "BUY": 1}
    label_to_int = {v: int(k) for k, v in int_to_label.items()}

    return model, feature_cols, int_to_label, label_to_int


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Ensure Date is parsed (not strictly required, but nice to have)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    model, feature_cols, int_to_label, label_to_int = load_model_and_meta()

    print("Running model over historical data...")
    # Select feature columns (fill NA for safety)
    X = df[feature_cols].fillna(0.0)

    # predict_proba returns shape (n_samples, n_classes)
    probas = model.predict_proba(X)
    classes = model.classes_  # e.g. array([-1, 0, 1])

    # Determine predicted class index for each row
    pred_idx = probas.argmax(axis=1)
    pred_classes = classes[pred_idx]

    # Map classes → action labels, e.g. -1 -> "SELL"
    actions = [int_to_label[int(c)] for c in pred_classes]

    # Add action + numeric signal
    df["model_action"] = actions

    signal_map = {"SELL": -1, "HOLD": 0, "BUY": 1}
    df["model_signal"] = df["model_action"].map(signal_map).astype("int64")

    # Find column indices for each class in the probas array
    def _class_index(target: int) -> int:
        idx_arr = np.where(classes == target)[0]
        if len(idx_arr) == 0:
            raise ValueError(f"Class {target} not found in model.classes_: {classes}")
        return int(idx_arr[0])

    idx_sell = _class_index(-1)
    idx_hold = _class_index(0)
    idx_buy = _class_index(1)

    # Add probability columns
    df["proba_sell"] = probas[:, idx_sell]
    df["proba_hold"] = probas[:, idx_hold]
    df["proba_buy"] = probas[:, idx_buy]

    print("Saving dataset with signals...")
    df.to_csv(OUT_PATH, index=False)
    print(f"Done. Saved with signals to: {OUT_PATH}")


if __name__ == "__main__":
    main()
