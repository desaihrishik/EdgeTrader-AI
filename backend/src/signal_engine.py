"""
Signal engine for NVDA.

- Loads latest processed row from nvda_daily_dataset.csv
- Loads trained RandomForest model
- Computes BUY / HOLD / SELL with confidence
- Detects active candlestick patterns for the latest day
- Suggests number of shares to trade for a given budget + risk profile

This is a pure Python helper. Later, we'll wrap this in a REST API
so the frontend can call it.
"""

from pathlib import Path
import json
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd
from joblib import load


ROOT_DIR = Path(__file__).resolve().parents[1]   # backend/
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

DATASET_CSV = DATA_DIR / "nvda_daily_dataset.csv"
MODEL_PATH = MODEL_DIR / "nvda_rf_signal.pkl"
META_PATH = MODEL_DIR / "nvda_rf_metadata.json"

RiskProfile = Literal["low", "medium", "high"]


def load_model_and_data():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_CSV}. Run build_dataset.py first."
        )
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            f"Model or metadata not found. Run train_model.py first."
        )

    df = pd.read_csv(DATASET_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    raw_label_mapping: Dict[str, str] = meta["label_mapping"]
    # convert key strings back to ints: {"-1": "SELL"} → {-1: "SELL"}
    label_mapping = {int(k): v for k, v in raw_label_mapping.items()}

    clf = load(MODEL_PATH)

    return df, clf, feature_cols, label_mapping


def compute_position_size(
    budget: float,
    risk_profile: RiskProfile,
    confidence: float,
    price: float,
) -> int:
    """
    Very simple position sizing rule:
    - Allocate a fraction of budget based on risk_profile
    - Scale by model confidence
    - Convert to integer number of shares (floor)
    - Ensure we can still take a tiny position if there is a BUY signal
    """

    # Basic sanity check
    if budget <= 0 or price <= 0:
        return 0

    # Base risk fraction by profile
    if risk_profile == "low":
        base_frac = 0.25
    elif risk_profile == "medium":
        base_frac = 0.5
    else:  # "high"
        base_frac = 0.9

    # Clamp confidence between 0 and 1
    confidence = max(0.0, min(1.0, confidence))

    # Scale by confidence (0–1)
    capital_to_use = budget * base_frac * confidence

    # Raw integer share count
    shares = int(capital_to_use // price)

    # --- MINIMUM POSITION RULE (this is the new bit) ---
    # If this rounds to 0 shares but we still have a meaningful fraction
    # of a share's price, allow 1 share as a tiny test position.
    # Example: capital_to_use is at least 20% of a share.
    if shares <= 0:
        min_cap_threshold = 0.20  # 20% of price
        if capital_to_use / price >= min_cap_threshold:
            shares = 1
        else:
            shares = 0
    # ---------------------------------------------------

    if shares < 0:
        shares = 0
    return shares


    


def get_latest_recommendation(
    budget: float,
    risk_profile: RiskProfile = "medium",
) -> Dict[str, Any]:
    """
    Core function: return recommendation for the latest NVDA candle.

    Returns a dict with:
    - date
    - latest_close
    - action ("BUY"/"SELL"/"HOLD")
    - signal_value (-1/0/1)
    - confidence (0-1)
    - probas per class
    - suggested_shares
    - capital_used
    - patterns (list of detected patterns for that day)
    - explanation (short text)
    """

    df, clf, feature_cols, label_mapping = load_model_and_data()

    # Latest row (most recent trading day)
    latest = df.iloc[-1]
    date = latest["Date"]
    price = float(latest["Close"])

    # Prepare feature vector in the same column order used for training
    X_latest = latest[feature_cols].values.reshape(1, -1)

    # Predict probabilities and class
    proba = clf.predict_proba(X_latest)[0]  # shape: (n_classes,)
    classes = clf.classes_                  # e.g. [-1, 0, 1]
    best_idx = int(np.argmax(proba))
    best_class = int(classes[best_idx])
    best_label = label_mapping.get(best_class, str(best_class))
    best_conf = float(proba[best_idx])

    # Map probabilities to labels for more detailed display
    probas_by_label = {
        label_mapping.get(int(cls), str(int(cls))): float(p)
        for cls, p in zip(classes, proba)
    }

    # Detect which candlestick patterns are active
    pattern_cols = [c for c in df.columns if c.startswith("pattern_")]
    active_patterns = [
        c.replace("pattern_", "")
        for c in pattern_cols
        if int(latest[c]) == 1
    ]

    # Compute position size
    suggested_shares = 0
    if best_label == "BUY":
        suggested_shares = compute_position_size(
            budget=budget,
            risk_profile=risk_profile,
            confidence=best_conf,
            price=price,
        )
    # For SELL, this engine does not know user's holdings.
    # Frontend can decide to sell up to current position.

    capital_used = suggested_shares * price

    # Short explanation for UI
    if best_label == "BUY":
        decision_text = "Model suggests BUY based on current trend and indicators."
    elif best_label == "SELL":
        decision_text = "Model suggests SELL; downside risk is elevated."
    else:
        decision_text = "Model suggests HOLD; no strong edge detected."

    if active_patterns:
        decision_text += f" Detected patterns: {', '.join(active_patterns)}."

    result = {
        "date": date.strftime("%Y-%m-%d"),
        "latest_close": price,
        "action": best_label,
        "signal_value": best_class,
        "confidence": best_conf,
        "probas": probas_by_label,
        "suggested_shares": int(suggested_shares),
        "capital_used": capital_used,
        "patterns": active_patterns,
        "risk_profile": risk_profile,
        "budget": budget,
        # For debugging / future frontend: we can expose features too
        "features": {col: float(latest[col]) for col in feature_cols},
        "explanation": decision_text,
    }

    return result


if __name__ == "__main__":
    # Quick manual test
    example_budget = 1000.0
    example_risk: RiskProfile = "medium"

    rec = get_latest_recommendation(
        budget=example_budget,
        risk_profile=example_risk,
    )

    print("=== NVDA Recommendation ===")
    for k, v in rec.items():
        print(f"{k}: {v}")
