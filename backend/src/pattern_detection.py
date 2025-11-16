"""
Pattern detection module for NVDA daily data.

Includes:
- Candlestick patterns (single, double, triple candle)
- Structural patterns (swing trading patterns) over a lookback window

NOTE:
These are heuristic detectors, good for hackathon / prototype use.
For production / research-grade use, refine thresholds and logic.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# ============ BASIC HELPERS ============

def _body(o: pd.Series, c: pd.Series) -> pd.Series:
    return (c - o).abs()


def _range(h: pd.Series, l: pd.Series) -> pd.Series:
    return h - l


def _upper_wick(o: pd.Series, c: pd.Series, h: pd.Series) -> pd.Series:
    return h - np.maximum(o, c)


def _lower_wick(o: pd.Series, c: pd.Series, l: pd.Series) -> pd.Series:
    return np.minimum(o, c) - l


# ============ CANDLESTICK PATTERNS (ROW-WISE / WITH SHIFTS) ============

def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a set of candlestick pattern columns to df.
    Patterns are computed using current + previous 1-2 candles.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    body = _body(o, c)
    rng = _range(h, l)
    upper = _upper_wick(o, c, h)
    lower = _lower_wick(o, c, l)

    df["c_body"] = body
    df["c_range"] = rng
    df["c_upper_wick"] = upper
    df["c_lower_wick"] = lower
    df["c_body_to_range"] = np.where(rng > 0, body / rng, 0.0)
    df["c_direction"] = np.sign(c - o).replace(0, 0)  # 1 up, -1 down, 0 doji

    # Single-candle patterns

    # Doji: very small body relative to range
    df["pattern_doji"] = (df["c_body_to_range"] < 0.1).astype(int)

    # Spinning top: small body, relatively long wicks
    df["pattern_spinning_top"] = (
        (df["c_body_to_range"] < 0.3) &
        (upper / (rng + 1e-9) > 0.2) &
        (lower / (rng + 1e-9) > 0.2)
    ).astype(int)

    # Hammer / Hanging Man: small body at top, long lower wick
    long_lower = lower / (rng + 1e-9) > 0.6
    small_body = df["c_body_to_range"] < 0.3

    df["pattern_hammer"] = (small_body & long_lower & (df["c_direction"] >= 0)).astype(int)
    df["pattern_hanging_man"] = (small_body & long_lower & (df["c_direction"] <= 0)).astype(int)

    # Inverted hammer / Shooting star: small body at bottom, long upper wick
    long_upper = upper / (rng + 1e-9) > 0.6
    df["pattern_inverted_hammer"] = (small_body & long_upper & (df["c_direction"] >= 0)).astype(int)
    df["pattern_shooting_star"] = (small_body & long_upper & (df["c_direction"] <= 0)).astype(int)

    # Marubozu (full body, near no wicks)
    df["pattern_bullish_marubozu"] = (
        (df["c_direction"] > 0) &
        (upper / (rng + 1e-9) < 0.05) &
        (lower / (rng + 1e-9) < 0.05)
    ).astype(int)

    df["pattern_bearish_marubozu"] = (
        (df["c_direction"] < 0) &
        (upper / (rng + 1e-9) < 0.05) &
        (lower / (rng + 1e-9) < 0.05)
    ).astype(int)

    # Two-candle patterns: use shifted series
    o_prev = o.shift(1)
    c_prev = c.shift(1)
    body_prev = (c_prev - o_prev).abs()

    # Bullish engulfing
    df["pattern_bullish_engulfing"] = (
        (c_prev < o_prev) &           # previous red
        (c > o) &                     # current green
        (body > body_prev) &
        (c >= o_prev) & (o <= c_prev) # body engulfs
    ).astype(int)

    # Bearish engulfing
    df["pattern_bearish_engulfing"] = (
        (c_prev > o_prev) &           # previous green
        (c < o) &                     # current red
        (body > body_prev) &
        (o >= c_prev) & (c <= o_prev)
    ).astype(int)

    # Piercing (bullish)
    mid_prev = (o_prev + c_prev) / 2.0
    df["pattern_piercing"] = (
        (c_prev < o_prev) &
        (o < c_prev) &
        (c > mid_prev) &
        (c < o_prev)
    ).astype(int)

    # Dark cloud cover (bearish)
    mid_prev = (o_prev + c_prev) / 2.0
    df["pattern_dark_cloud_cover"] = (
        (c_prev > o_prev) &
        (o > c_prev) &
        (c < mid_prev) &
        (c > o_prev)
    ).astype(int)

    # Harami patterns
    df["pattern_bullish_harami"] = (
        (c_prev < o_prev) &
        (c > o) &
        (c < o_prev) & (o > c_prev)
    ).astype(int)

    df["pattern_bearish_harami"] = (
        (c_prev > o_prev) &
        (c < o) &
        (c > o_prev) & (o < c_prev)
    ).astype(int)

    # Three-candle patterns

    # Morning star: down candle, small candle, strong up candle
    o_2 = o.shift(2)
    c_2 = c.shift(2)

    df["pattern_morning_star"] = (
        (c_2 < o_2) &  # first red
        (df["c_body"].shift(1) / (df["c_range"].shift(1) + 1e-9) < 0.3) &  # small middle
        (c > o) &      # last green
        (c > ((o_2 + c_2) / 2.0))   # close into body of first
    ).astype(int)

    df["pattern_evening_star"] = (
        (c_2 > o_2) &  # first green
        (df["c_body"].shift(1) / (df["c_range"].shift(1) + 1e-9) < 0.3) &
        (c < o) &      # last red
        (c < ((o_2 + c_2) / 2.0))
    ).astype(int)

    # Three white soldiers / three black crows
    df["pattern_three_white_soldiers"] = (
        (c.shift(2) > o.shift(2)) &
        (c.shift(1) > o.shift(1)) &
        (c > o) &
        (c > c.shift(1)) &
        (c.shift(1) > c.shift(2))
    ).astype(int)

    df["pattern_three_black_crows"] = (
        (c.shift(2) < o.shift(2)) &
        (c.shift(1) < o.shift(1)) &
        (c < o) &
        (c < c.shift(1)) &
        (c.shift(1) < c.shift(2))
    ).astype(int)

    # Inside bar / Outside bar
    df["pattern_inside_bar"] = (
        (h < h.shift(1)) &
        (l > l.shift(1))
    ).astype(int)

    df["pattern_outside_bar"] = (
        (h > h.shift(1)) &
        (l < l.shift(1))
    ).astype(int)

    return df


# ============ STRUCTURAL PATTERNS (SWING STRUCTURES) ============

def _find_local_extrema(prices: pd.Series, prominence: float = 0.01) -> Tuple[List[int], List[int]]:
    """
    Improved local peaks/trough detection.

    - prominence = minimum percentage difference vs neighbors
      (default 1% moves)
    - returns: (peaks, troughs)
    """

    prices = prices.values
    n = len(prices)
    peaks = []
    troughs = []

    for i in range(2, n - 2):
        left = prices[i - 1]
        right = prices[i + 1]
        center = prices[i]

        # Peak: center > neighbors and beats them by prominence
        if center > left and center > right:
            if (center - max(left, right)) / max(left, right) >= prominence:
                peaks.append(i)

        # Trough: center < neighbors and below by prominence
        if center < left and center < right:
            if (min(left, right) - center) / min(left, right) >= prominence:
                troughs.append(i)

    return peaks, troughs



def _price_close_enough(a: float, b: float, tol: float = 0.05) -> bool:
    """Return True if two prices are within tol (2% default)."""
    if a == 0:
        return False
    return abs(a - b) / abs(a) <= tol


def add_structural_patterns(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Detect structural swing patterns over last `lookback` bars.
    Adds boolean columns:

    - pattern_struct_double_top
    - pattern_struct_double_bottom
    - pattern_struct_head_shoulders
    - pattern_struct_inverse_head_shoulders
    - pattern_struct_ascending_triangle
    - pattern_struct_descending_triangle
    - pattern_struct_sym_triangle
    - pattern_struct_rising_wedge
    - pattern_struct_falling_wedge
    - pattern_struct_bull_flag
    - pattern_struct_bear_flag

    We mark 1 ONLY on the last row if such a pattern exists in the
    lookback window and the last candle is near the breakout region.
    """
    df = df.copy()
    n = len(df)

    pattern_cols = [
        "pattern_struct_double_top",
        "pattern_struct_double_bottom",
        "pattern_struct_head_shoulders",
        "pattern_struct_inverse_head_shoulders",
        "pattern_struct_ascending_triangle",
        "pattern_struct_descending_triangle",
        "pattern_struct_sym_triangle",
        "pattern_struct_rising_wedge",
        "pattern_struct_falling_wedge",
        "pattern_struct_bull_flag",
        "pattern_struct_bear_flag",
    ]

    # Initialize all 0
    for col in pattern_cols:
        df[col] = 0

    # Not enough history → just return zeros
    if n < lookback + 10:
        return df

    sub = df.iloc[-lookback:].reset_index(drop=True)
    closes = sub["Close"]
    highs = sub["High"]
    lows = sub["Low"]

    # Correct: peaks from HIGH series, troughs from LOW series
    peak_idx, _ = _find_local_extrema(highs)
    _, trough_idx = _find_local_extrema(lows)

    last_idx_global = df.index[-1]

    # ---------- 1) Double top / Double bottom ----------
    if len(peak_idx) >= 2:
        p1, p2 = peak_idx[-2], peak_idx[-1]
        h1, h2 = highs.iloc[p1], highs.iloc[p2]
        if _price_close_enough(h1, h2, tol=0.02):
            # neckline = last trough between p1 and p2
            mid_troughs = [t for t in trough_idx if p1 < t < p2]
            if mid_troughs:
                neckline = lows.iloc[mid_troughs[-1]]
                last_close = closes.iloc[-1]
                if last_close <= neckline * 1.01:  # near/below neckline
                    df.loc[last_idx_global, "pattern_struct_double_top"] = 1

    if len(trough_idx) >= 2:
        t1, t2 = trough_idx[-2], trough_idx[-1]
        l1, l2 = lows.iloc[t1], lows.iloc[t2]
        if _price_close_enough(l1, l2, tol=0.02):
            mid_peaks = [p for p in peak_idx if t1 < p < t2]
            if mid_peaks:
                neckline = highs.iloc[mid_peaks[-1]]
                last_close = closes.iloc[-1]
                if last_close >= neckline * 0.99:  # near/above neckline
                    df.loc[last_idx_global, "pattern_struct_double_bottom"] = 1

    # ---------- 2) Head and shoulders / inverse H&S ----------
    if len(peak_idx) >= 3:
        p1, p2, p3 = peak_idx[-3], peak_idx[-2], peak_idx[-1]
        h1, h2, h3 = highs.iloc[p1], highs.iloc[p2], highs.iloc[p3]
        if h2 > h1 and h2 > h3 and _price_close_enough(h1, h3, tol=0.05):
            # shoulders don't need to be perfect
            if (p3 - p1) >= 3:  # spaced out
                df.loc[last_idx_global, "pattern_struct_head_shoulders"] = 1

    if len(trough_idx) >= 3:
        t1, t2, t3 = trough_idx[-3], trough_idx[-2], trough_idx[-1]
        l1, l2, l3 = lows.iloc[t1], lows.iloc[t2], lows.iloc[t3]
        if l2 < l1 and l2 < l3 and _price_close_enough(l1, l3, tol=0.03):
            df.loc[last_idx_global, "pattern_struct_inverse_head_shoulders"] = 1

    # ---------- 3) Triangles and wedges ----------
    x = np.arange(len(sub))
    x_mean = x.mean()

    def _slope(y: pd.Series) -> float:
        y_arr = y.values
        y_mean = y_arr.mean()
        num = ((x - x_mean) * (y_arr - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum() + 1e-9
        return num / den

    high_slope = _slope(highs)
    low_slope = _slope(lows)

    # Ascending triangle: highs ~ flat, lows rising
    if abs(high_slope) < 0.002 and low_slope > 0.0:
        df.loc[last_idx_global, "pattern_struct_ascending_triangle"] = 1

    # Descending triangle: lows ~ flat, highs falling
    if abs(low_slope) < 0.002 and high_slope < 0.0:
        df.loc[last_idx_global, "pattern_struct_descending_triangle"] = 1

    # Symmetrical triangle: highs trending down, lows trending up
    if high_slope < 0.0 and low_slope > 0.0:
        df.loc[last_idx_global, "pattern_struct_sym_triangle"] = 1

    # Wedges: both highs and lows trend in same direction, range contracts
    ranges = highs - lows
    range_slope = _slope(ranges)

    if high_slope > 0 and low_slope > 0 and range_slope < 0:
        df.loc[last_idx_global, "pattern_struct_rising_wedge"] = 1

    if high_slope < 0 and low_slope < 0 and range_slope < 0:
        df.loc[last_idx_global, "pattern_struct_falling_wedge"] = 1

    # ---------- 4) Flags ----------
    flag_window = min(15, lookback)
    if len(closes) > flag_window + 10:
        sub_flag = sub.iloc[-flag_window:]
        flag_returns = sub_flag["Close"].pct_change().cumsum()

        overall_ret = closes.iloc[-flag_window] / closes.iloc[-flag_window - 10] - 1.0
        inside_vol = flag_returns.std()

        if overall_ret > 0.08 and inside_vol < 0.03:
            df.loc[last_idx_global, "pattern_struct_bull_flag"] = 1
        if overall_ret < -0.08 and inside_vol < 0.03:
            df.loc[last_idx_global, "pattern_struct_bear_flag"] = 1

    return df

