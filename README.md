EdgeTrader · AI Swing Trading Engine
ML signals • Agentic forecasting • LLM reasoning • Portfolio-ready architecture

EdgeTrader is a full-stack autonomous swing-trading system designed initially for NVDA.
It combines:

RandomForest ML trading signals

Candlestick & pattern detection

Sentiment scoring

Monte Carlo agentic forecasting

Local LLM evaluation (Ollama)

Full React + FastAPI implementation

Although the engine is currently tuned and optimized for NVDA, the architecture is fully extensible:

🚀 Upcoming goal: Expand EdgeTrader into a multi-asset intelligence platform with
AAPL · TSLA · SPY · QQQ · ETH · BTC…
and build a portfolio-position sizing + risk management layer for end-to-end trading.

📌 Features
🔥 1. ML-Powered Trading Signals

RandomForest classifier trained on NVDA historical data

Predicts BUY / SELL / HOLD

Confidence score + probability distribution

Provides pattern-aware explanations

📈 2. Candlestick Pattern Engine

Detects 20+ price-action patterns

Highlights strongest recent signals

Integrates with the LLM reasoning stage

🌤 3. Agentic Monte-Carlo Forecasting

Simulates future paths using volatility-adaptive sampling

Generates:

P10 (Worst Case)

P50 (Median Case)

P90 (Best Case)

Automatically selects horizon based on trend strength

Sentiment-adjusted

🧠 4. LLM Evaluator (Ollama)

Reads the full market context

Interprets ML signals, patterns, trend, sentiment

Produces a human-like market explanation

Helps validate/override ML signal using reasoning

💼 5. Portfolio Expansion (Upcoming)

Planned modules include:

Multi-ticker support

Dynamic position sizing

Entry-price tracking per ticker

Risk-weighted allocation

P&L monitoring

Portfolio-level LLM reporting

(The current version includes NVDA-only inference, but backend design is already ticker-agnostic.)
