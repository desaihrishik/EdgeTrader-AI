# # backend/src/agent_engine.py

# """
# Agentic swing-trading helper for NVDA.

# - Uses historical model BUY signals from nvda_daily_with_signals.csv
# - Computes forward returns for multiple holding horizons
# - Picks a *personalized* holding horizon based on:
#     * risk profile (low / medium / high)
#     * model confidence on latest signal
#     * current news sentiment
#     * budget (position size, lightly)
# - Builds base / best / worst projection curves for the UI
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Any

# import numpy as np
# import pandas as pd

# from src.signal_engine import get_latest_recommendation, RiskProfile
# from src.sentiment_engine import get_nvda_sentiment


# ROOT_DIR = Path(__file__).resolve().parents[1]  # backend/
# DATA_DIR = ROOT_DIR / "data"
# HIST_CSV = DATA_DIR / "nvda_daily_with_signals.csv"

# # Candidate holding horizons in trading days (roughly 1.5–5 weeks)
# # Kept away from extremes so it doesn't always become "30".
# CANDIDATE_HORIZONS = [7, 10, 12, 15, 18, 20, 22, 25]


# @dataclass
# class HorizonStats:
#     horizon: int
#     expected_return: float     # mean forward return
#     worst_case: float          # ~5th percentile
#     best_case: float           # ~95th percentile
#     win_rate: float            # fraction of BUY trades with >0 return
#     avg_win: float
#     avg_loss: float


# def _load_history() -> pd.DataFrame:
#     if not HIST_CSV.exists():
#         raise FileNotFoundError(
#             f"Historical signals file not found at {HIST_CSV}. "
#             "Run run_historical_signals.py first."
#         )

#     df = pd.read_csv(HIST_CSV, parse_dates=["Date"])
#     df = df.sort_values("Date").reset_index(drop=True)
#     return df


# def _compute_horizon_stats(df: pd.DataFrame) -> Dict[int, HorizonStats]:
#     """
#     For each candidate horizon H:
#       - compute forward H-day return for each row
#       - restrict to rows where model_signal == 1 (BUY)
#       - compute mean, percentiles, win-rate, avg win/loss
#     """
#     stats: Dict[int, HorizonStats] = {}

#     if "Close" not in df.columns or "model_signal" not in df.columns:
#         return stats

#     for h in CANDIDATE_HORIZONS:
#         col = f"ret_fwd_{h}d"
#         df[col] = df["Close"].shift(-h) / df["Close"] - 1.0

#         mask = (df["model_signal"] == 1) & df[col].notna()
#         samples = df.loc[mask, col].values.astype(float)

#         if len(samples) < 15:
#             # Not enough examples for this horizon → skip
#             continue

#         expected = float(np.mean(samples))
#         worst = float(np.percentile(samples, 5))
#         best = float(np.percentile(samples, 95))

#         win_rate = float((samples > 0).mean())
#         if (samples > 0).any():
#             avg_win = float(samples[samples > 0].mean())
#         else:
#             avg_win = 0.0
#         if (samples < 0).any():
#             avg_loss = float(samples[samples < 0].mean())
#         else:
#             avg_loss = 0.0

#         stats[h] = HorizonStats(
#             horizon=h,
#             expected_return=expected,
#             worst_case=worst,
#             best_case=best,
#             win_rate=win_rate,
#             avg_win=avg_win,
#             avg_loss=avg_loss,
#         )

#     return stats


# def _choose_horizon_risk_aware(
#     stats: Dict[int, HorizonStats],
#     risk_profile: RiskProfile,
#     confidence: float,
#     sentiment_score: float,
#     budget: float,
# ) -> HorizonStats:
#     """
#     Choose horizon using a risk-aware, agentic score:

#     For each horizon h:
#       score_base = w_ret * expected_return
#                  + w_win * (win_rate - 0.5)
#                  - w_dd * max(0, -worst_case)

#     Then adjust score based on:
#       - model confidence
#       - sentiment
#       - budget (very lightly)

#     The horizon with highest final score is chosen.
#     """
#     if not stats:
#         # Fallback dummy stats if nothing computed
#         return HorizonStats(
#             horizon=15,
#             expected_return=0.0,
#             worst_case=-0.1,
#             best_case=0.15,
#             win_rate=0.5,
#             avg_win=0.05,
#             avg_loss=-0.05,
#         )

#     # Weights depend on risk profile
#     if risk_profile == "low":
#         w_ret = 0.4   # care about return
#         w_win = 0.4   # care a lot about win rate
#         w_dd = 0.4    # strong penalty on drawdown
#     elif risk_profile == "medium":
#         w_ret = 0.6
#         w_win = 0.25
#         w_dd = 0.3
#     else:  # high
#         w_ret = 0.8   # mostly chase return
#         w_win = 0.15
#         w_dd = 0.2    # lighter downside penalty

#     confidence = max(0.0, min(1.0, float(confidence)))
#     sentiment_score = max(-1.0, min(1.0, float(sentiment_score)))

#     # Normalize budget into rough scale ~[0, 2]
#     budget_norm = min(budget / 5000.0, 2.0)

#     best_h: HorizonStats | None = None
#     best_score = -1e9

#     for h, st in stats.items():
#         exp_ret = st.expected_return          # e.g. 0.03
#         worst = st.worst_case                 # e.g. -0.08
#         win_rate = st.win_rate                # e.g. 0.55

#         # Base score: return + win rate – downside
#         base_score = (
#             w_ret * exp_ret
#             + w_win * (win_rate - 0.5) * 0.5  # 0.5 factor keeps it moderate
#             - w_dd * max(0.0, -worst)
#         )

#         # Confidence & sentiment modifiers:
#         # - confidence ∈ [0,1] → factor ~ [0.8, 1.2]
#         # - sentiment ∈ [-1,1] → factor ~ [0.7, 1.3]
#         conf_factor = 0.8 + 0.4 * confidence
#         sent_factor = 1.0 + 0.3 * sentiment_score

#         score = base_score * conf_factor * sent_factor

#         # Light budget effect:
#         # - for low risk, larger budget prefers *slightly shorter* horizons
#         # - for high risk, larger budget is okay with slightly longer ones
#         longness = h / max(CANDIDATE_HORIZONS)  # 0..1
#         if risk_profile == "low":
#             score -= 0.02 * budget_norm * longness  # small penalty
#         elif risk_profile == "high":
#             score += 0.01 * budget_norm * longness  # tiny bonus

#         if score > best_score:
#             best_score = score
#             best_h = st

#     assert best_h is not None
#     return best_h


# def get_agentic_projection(
#     budget: float,
#     risk_profile: RiskProfile,
# ) -> Dict[str, Any]:
#     """
#     Main entry point:

#     - Uses latest model recommendation
#     - Uses current sentiment
#     - Uses historical BUY stats
#     - Returns JSON-ready dict for frontend
#     """
#     df = _load_history()
#     stats_by_h = _compute_horizon_stats(df)

#     latest = get_latest_recommendation(budget=budget, risk_profile=risk_profile)
#     sentiment = get_nvda_sentiment()

#     action = latest["action"]
#     confidence = float(latest["confidence"])
#     entry_price = float(latest["latest_close"])
#     suggested_shares = int(latest["suggested_shares"])

#     # Approximate position value (used in projections)
#     if suggested_shares > 0:
#         position_value = suggested_shares * entry_price
#     else:
#         # Same risk fractions as compute_position_size
#         if risk_profile == "low":
#             frac = 0.25
#         elif risk_profile == "medium":
#             frac = 0.5
#         else:
#             frac = 0.9
#         position_value = budget * frac

#     if position_value <= 0:
#         position_value = max(budget * 0.3, 1.0)

#     chosen = _choose_horizon_risk_aware(
#         stats=stats_by_h,
#         risk_profile=risk_profile,
#         confidence=confidence,
#         sentiment_score=sentiment.score,
#         budget=budget,
#     )

#     H = chosen.horizon
#     exp_ret = chosen.expected_return
#     worst = chosen.worst_case
#     best = chosen.best_case

#     expected_final_value = position_value * (1.0 + exp_ret)
#     worst_final_value = position_value * (1.0 + worst)
#     best_final_value = position_value * (1.0 + best)

#     # Build simple linear projection curves for UI (0..H days)
#     days = list(range(0, H + 1))
#     expected_curve = [
#         position_value * (1.0 + exp_ret * (d / H)) for d in days
#     ]
#     best_curve = [
#         position_value * (1.0 + best * (d / H)) for d in days
#     ]
#     worst_curve = [
#         position_value * (1.0 + worst * (d / H)) for d in days
#     ]

#     # Decide if we actively recommend a new trade
#     can_trade = action == "BUY" and suggested_shares > 0

#     if can_trade:
#         summary = (
#             f"For a new BUY position of {suggested_shares} share(s) at around "
#             f"${entry_price:.2f}, the agent recommends a swing horizon of about "
#             f"{H} trading days. Based on similar historical setups, the expected "
#             f"return is ~{exp_ret * 100:.1f}%, with a worst-case (5th percentile) "
#             f"of ~{worst * 100:.1f}% and a best-case (95th percentile) of "
#             f"~{best * 100:.1f}% over that period."
#         )
#         reason = ""
#     else:
#         summary = (
#             "Model does not strongly recommend opening a new long position right now. "
#             f"Current action is {action}. The agent's typical swing horizon for "
#             f"BUY setups with your inputs is about {H} trading days, with an "
#             f"expected return of ~{exp_ret * 100:.1f}% over that period."
#         )
#         reason = (
#             "Model is not in a strong BUY state or suggested size is zero; "
#             "agent shows typical stats but does not recommend a fresh entry."
#         )

#     sentiment_news = [
#         {
#             "source": n.source,
#             "headline": n.headline,
#             "url": n.url,
#             "datetime": n.datetime,
#         }
#         for n in sentiment.news
#     ]

#     result: Dict[str, Any] = {
#         "can_trade": can_trade,
#         "reason": reason,
#         "entry_price": entry_price,
#         "shares": suggested_shares,
#         "position_value": position_value,
#         "recommended_holding_days": H,
#         "expected_return": exp_ret,
#         "worst_case": worst,
#         "best_case": best,
#         "expected_final_value": expected_final_value,
#         "worst_final_value": worst_final_value,
#         "best_final_value": best_final_value,
#         "win_rate": chosen.win_rate,
#         "avg_win": chosen.avg_win,
#         "avg_loss": chosen.avg_loss,
#         "sentiment_label": sentiment.label,
#         "sentiment_score": sentiment.score,
#         "sentiment_news": sentiment_news,
#         "projection_curve": {
#             "days": days,
#             "expected": expected_curve,
#             "best": best_curve,
#             "worst": worst_curve,
#         },
#         "summary_text": summary,
#     }

#     return result
