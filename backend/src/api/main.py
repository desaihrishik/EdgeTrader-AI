# backend/src/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd

from src.signal_engine import get_latest_recommendation
from src.agent_engine import get_agentic_recommendation

# ⭐ ADD THIS ⭐
from src.api.llm_routes import router as llm_router


app = FastAPI(
    title="Edge AI Trading Backend",
    description="NVDA prediction engine with patterns and ML signals",
    version="1.0",
)

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "nvda_daily_dataset.csv"


@app.get("/")
def home():
    return {"message": "NVDA Edge AI Server Running"}


@app.get("/api/ping")
def ping():
    return {"status": "ok"}


@app.get("/api/nvda/latest_signal")
def latest_signal(budget: float = 1000.0, risk: str = "medium"):
    rec = get_latest_recommendation(budget=budget, risk_profile=risk)
    return rec


@app.get("/api/nvda/candles")
def get_candles(limit: int = 120):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    if limit:
        df = df.tail(int(limit))

    candle_data = []
    pattern_cols = [c for c in df.columns if c.startswith("pattern_")]

    for _, row in df.iterrows():
        patterns = [
            c.replace("pattern_", "")
            for c in pattern_cols
            if int(row[c]) == 1
        ]

        candle_data.append(
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
                "patterns": patterns,
                "signal": int(row["signal"]),
            }
        )

    return {"candles": candle_data}


@app.get("/api/nvda/agentic_signal")
def agentic_signal(
    budget: float = 1000.0,
    risk: str = "medium",
    entry_price: float | None = None,
):
    rec = get_agentic_recommendation(
        budget=budget,
        risk_profile=risk,
        entry_price=entry_price,
    )
    return rec


# ⭐ REGISTER THE LLM ENDPOINTS ⭐
app.include_router(llm_router)
