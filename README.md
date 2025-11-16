##EdgeTrader AI

## 🚀 Inspiration

I wanted to build an intelligent trading assistant that doesn’t just output BUY or SELL signals, but actually reasons like a human analyst. Most trading tools only show indicators — I wanted a system that explains *why*. This led to building an agentic AI that blends ML signals, price action, sentiment analysis, and an LLM reasoning layer.

## 📈 What it does

EdgeTrader AI generates swing-trading recommendations using:
- RandomForest model predictions  
- Candlestick pattern recognition  
- Sentiment scoring  
- Dynamic holding horizon selection  
- Monte-Carlo forecasting  

It also uses an LLM layer to evaluate market context and deliver human-like reasoning for every recommendation.

Currently, it is tuned for **NVDA**, but the architecture is designed to expand to **all stocks** and eventually manage **entire trading portfolios**.

## 🛠️ How we built it

- **Backend:** FastAPI powering ML inference, pattern extraction, sentiment analysis, trend scoring, and simulation forecasts  
- **Frontend:** React + ApexCharts for candlestick charts, price projections, and agentic insights  
- **LLM Layer:** Local Ollama models for contextual evaluation and natural-language reasoning  
- **ML Models:** RandomForest-based BUY/HOLD/SELL classifier trained on NVDA historical data  

## ⚠️ Challenges we ran into

- Keeping LLM outputs consistent and non-hallucinatory  
- Managing complex React state across multiple async endpoints  
- Extracting meaningful patterns from noisy candlestick data  
- Designing a clean UI despite many insights  
- Ensuring responsiveness while running multiple models  

## 🏆 Accomplishments that we're proud of

- Blending ML forecasts, sentiment, price patterns, and LLM reasoning in one system  
- Achieving interpretable explanations rather than black-box outputs  
- Building a smooth, modern interface with data + reasoning visualized  
- Designing a modular pipeline that scales beyond NVDA  

## 📚 What we learned

- How ML + LLM hybrid systems improve interpretability and decision quality  
- Importance of robust preprocessing for stable predictions  
- Deep insights into candlestick structures and trend modeling  
- How to design an agentic workflow that feels interactive and intelligent  

## 🔮 What’s next for EdgeTrader AI

- Expanding beyond NVDA to all major stocks  
- Building a portfolio-level AI manager for allocations and rebalancing  
- Adding real-time market streaming + automated alerts  
- Integrating broker APIs for paper trading and later full execution  
- Training improved custom models tailored to each stock’s behavior  
