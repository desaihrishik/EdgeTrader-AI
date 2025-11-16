from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "nvda_daily_dataset.csv"

df = pd.read_csv(DATASET, parse_dates=["Date"])
pattern_cols = [c for c in df.columns if c.startswith("pattern_")]

# Last 60 days where any pattern is present
mask = df[pattern_cols].sum(axis=1) > 0
hits = df.loc[mask, ["Date"] + pattern_cols].tail(20)

print("Last pattern occurrences:")
print(hits if not hits.empty else "No pattern flags found (may be rare / thresholds strict).")
