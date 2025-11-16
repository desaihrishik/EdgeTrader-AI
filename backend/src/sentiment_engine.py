"""
Simple sentiment helper for NVDA.

- Uses Finnhub company-news endpoint (free tier OK).
- Derives a rough sentiment score from recent headlines.
- If anything fails, returns neutral with empty news list.

This is a lightweight, hackathon-friendly module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any

from datetime import datetime, timedelta

import requests


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"


@dataclass
class SentimentNewsItem:
    source: str
    headline: str
    url: str
    datetime: str  # ISO-ish string


@dataclass
class SentimentResult:
    score: float         # -1.0..1.0 (bearish..bullish)
    label: str           # "bearish" | "neutral" | "bullish"
    raw: Dict[str, Any]
    news: List[SentimentNewsItem]


def _label_from_score(score: float) -> str:
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def _compute_headline_sentiment(news_data: List[Dict[str, Any]]) -> float:
    """
    Very naive headline-based sentiment:

    - Count simple positive / negative keywords in headline + summary.
    - Score in [-1, 1]. This is enough for a hackathon demo.
    """
    if not news_data:
        return 0.0

    positive_words = [
        "beat",
        "beats",
        "record",
        "strong",
        "surge",
        "rally",
        "gain",
        "gains",
        "bullish",
        "upgrade",
        "upgraded",
        "buy",
        "outperform",
        "growth",
        "accelerate",
        "optimistic",
        "raised",
        "raise",
    ]
    negative_words = [
        "miss",
        "misses",
        "cut",
        "cuts",
        "downgrade",
        "downgraded",
        "sell",
        "underperform",
        "bearish",
        "loss",
        "losses",
        "decline",
        "drop",
        "drops",
        "plunge",
        "plunges",
        "slump",
        "weak",
        "lawsuit",
        "investigation",
        "regulator",
    ]

    score = 0
    hits = 0

    for item in news_data:
        text = (
            (item.get("headline") or "") + " " + (item.get("summary") or "")
        ).lower()

        for w in positive_words:
            if w in text:
                score += 1
                hits += 1

        for w in negative_words:
            if w in text:
                score -= 1
                hits += 1

    if hits == 0:
        return 0.0

    # Normalize to [-1, 1] but keep some magnitude
    raw = score / hits  # between roughly -1 and 1 in practice
    return max(-1.0, min(1.0, raw))


def get_nvda_sentiment() -> SentimentResult:
    """
    Fetch simple sentiment for NVDA using only company news.

    - If FINNHUB_API_KEY is missing → neutral with no news.
    - If request fails → neutral with error info in raw.
    - Otherwise → uses recent 5 days of headlines to compute a rough score.
    """
    if not FINNHUB_API_KEY:
        return SentimentResult(
            score=0.0,
            label="neutral",
            raw={"reason": "FINNHUB_API_KEY not set"},
            news=[],
        )

    try:
        # --- Fetch company news for last ~5 days ---
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=5)

        resp = requests.get(
            FINNHUB_COMPANY_NEWS_URL,
            params={
                "symbol": "NVDA",
                "from": str(from_date),
                "to": str(to_date),
                "token": FINNHUB_API_KEY,
            },
            timeout=5,
        )
        resp.raise_for_status()
        news_data = resp.json()

        # Build list of news items (limit to 5 for display)
        news_items: List[SentimentNewsItem] = []
        for item in news_data[:5]:
            ts = item.get("datetime")
            if isinstance(ts, (int, float)):
                dt_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            else:
                dt_str = str(ts)

            news_items.append(
                SentimentNewsItem(
                    source=str(item.get("source", "")),
                    headline=str(item.get("headline", "")),
                    url=str(item.get("url", "")),
                    datetime=dt_str,
                )
            )

        # Derive sentiment from headlines (or neutral if none)
        score = _compute_headline_sentiment(news_data)
        label = _label_from_score(score)

        return SentimentResult(
            score=score,
            label=label,
            raw={
                "article_count": len(news_data),
                "method": "headline_keyword_heuristic",
            },
            news=news_items,
        )

    except Exception as e:
        # Graceful fallback: neutral sentiment, no news
        return SentimentResult(
            score=0.0,
            label="neutral",
            raw={"error": str(e)},
            news=[],
        )
