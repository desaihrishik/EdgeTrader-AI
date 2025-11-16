# backend/src/api/nvda_routes.py

from fastapi import APIRouter
from pathlib import Path
import pandas as pd

from src.signal_engine import get_latest_recommendation
from src.agent_engine import get_agentic_recommendation

router = APIRouter()

ROOT_DIR = Path(__file__).resolve().parents[2]   # backend/
DATA_PATH = ROOT_DIR / "data" / "nvda_daily_dataset.csv"


@router.get("/api/ping")
def ping():
    return {"status": "ok"}


@router.get("/api/nvda/latest_signal")
def latest_signal(budget: float = 1000.0, risk: str = "medium"):
    rec = get_latest_recommendation(
        budget=budget,
        risk_profile=risk,  # type: ignore
    )
    return rec


@router.get("/api/nvda/candles")
def get_candles(limit: int = 120):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.tail(limit)

    pattern_cols = [c for c in df.columns if c.startswith("pattern_")]

    candles = []
    for _, row in df.iterrows():
        patterns = [
            c.replace("pattern_", "") for c in pattern_cols if int(row[c]) == 1
        ]

        candles.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "patterns": patterns,
            "signal": int(row["signal"]),
        })

    return {"candles": candles}


@router.get("/api/nvda/agentic_signal")
def agentic_signal(
    budget: float = 1000.0,
    risk: str = "medium",
    entry_price: float | None = None,
):
    rec = get_agentic_recommendation(
        budget=budget,
        risk_profile=risk,  # type: ignore
        entry_price=entry_price,
    )
    return rec
