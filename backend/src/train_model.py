"""
Train RandomForest classifier to predict BUY / HOLD / SELL for NVDA.

Inputs:
- backend/data/nvda_daily_dataset.csv

Outputs:
- backend/models/nvda_rf_signal.pkl          (trained model)
- backend/models/nvda_rf_metadata.json       (feature list & label mapping)

The model predicts:
- signal = -1 → SELL
- signal =  0 → HOLD
- signal =  1 → BUY

Later we'll use predict_proba() to get confidence percentages and
combine with budget to suggest how many shares to buy.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump


ROOT_DIR = Path(__file__).resolve().parents[1]   # backend/
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATASET_CSV = DATA_DIR / "nvda_daily_dataset.csv"


def train_model():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_CSV}. Run build_dataset.py first."
        )

    # Load dataset
    df = pd.read_csv(DATASET_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Features = everything except Date, future_return_1d, signal
    feature_cols = [
        c for c in df.columns
        if c not in ("Date", "future_return_1d", "signal")
    ]
    X = df[feature_cols].values
    y = df["signal"].values

    # --- Time-based train/test split ---
    # Use most recent ~252 trading days (~1 year) as test,
    # rest as training. If dataset shorter, fallback to 20% split.
    if len(df) > 600:
        test_size = 252
    else:
        test_size = max(int(0.2 * len(df)), 50)

    split_idx = len(df) - test_size

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")

    # --- Train RandomForest model ---
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    # --- Evaluate on test period (most recent history) ---
    y_pred = clf.predict(X_test)

    print("\nClassification report (on last part of history):")
    print(classification_report(y_test, y_pred, digits=3))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    # --- Save model and metadata ---
    model_path = MODEL_DIR / "nvda_rf_signal.pkl"
    dump(clf, model_path)

    # label_mapping: numeric class → human action name
    label_mapping = {
        "-1": "SELL",
        "0": "HOLD",
        "1": "BUY",
    }

    # class_order: order of classes as used by predict_proba
    class_order = [int(c) for c in clf.classes_]

    meta = {
        "feature_cols": feature_cols,
        "label_mapping": label_mapping,
        "class_order": class_order,
        # relative model path so other scripts can resolve it
        "model_path": model_path.name,
    }

    meta_path = MODEL_DIR / "nvda_rf_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to   : {model_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    train_model()
