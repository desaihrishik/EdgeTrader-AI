from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
import os

# --- Load .env ---
BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

# --- Import Routers ---
from .nvda_routes import router as nvda_router

# --- Create App ---
app = FastAPI(title="EdgeTrader Backend")

# --- CORS for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Attach all routes ---
app.include_router(nvda_router)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd

from src.signal_engine import get_latest_recommendation
from src.agent_engine import get_agentic_recommendation

# ⬇️ NEW: import the LLM router
from src.api.llm_routes import router as llm_router

app = FastAPI(
    title="Edge AI Trading Backend",
    description="NVDA prediction engine with patterns and ML signals",
    version="1.0",
)

# CORS etc ...
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "nvda_daily_dataset.csv"

# ... your existing endpoints here ...


# ⬇️ REGISTER LLM ROUTES (at the bottom)
app.include_router(llm_router)
