"""
Clean NVDA OHLCV downloader using yfinance.
Produces perfect CSVs with standard headers:
Date,Open,High,Low,Close,Adj Close,Volume
"""

from pathlib import Path
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parents[1]  # backend/
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_nvda_data():
    ticker = yf.Ticker("NVDA")

    print("Downloading NVDA daily (5 years)...")
    daily = ticker.history(period="5y", interval="1d")
    
    print("Downloading NVDA intraday (60 days, 5m)...")
    intraday = ticker.history(period="60d", interval="5m")

    # Ensure Date index becomes a column
    daily.reset_index(inplace=True)
    intraday.reset_index(inplace=True)

    # Save clean CSVs
    daily_path = DATA_DIR / "nvda_daily_5y.csv"
    intraday_path = DATA_DIR / "nvda_5min_60d.csv"

    daily.to_csv(daily_path, index=False)
    intraday.to_csv(intraday_path, index=False)

    print(f"\nSaved: {daily_path}")
    print(f"Saved: {intraday_path}")

    print("\nPreview:")
    print(daily.head())
    print(intraday.head())


if __name__ == "__main__":
    download_nvda_data()
