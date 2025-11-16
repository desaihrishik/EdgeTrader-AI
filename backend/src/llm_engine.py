# backend/src/llm_engine.py

"""
LLM engine stub.

Right now this is a simple rule-based placeholder so you can
wire up the API + frontend without having the real LLM installed.

Later, you will replace `generate()` with a call to your
Qualcomm-optimized Llama model (via Qualcomm AI Hub / QNN).
"""

from textwrap import dedent


def generate(prompt: str, max_new_tokens: int = 512) -> str:
    """
    TEMP IMPLEMENTATION:
    - Looks at the prompt and returns a simple canned response.
    - You will later swap this out for the real on-device LLM.
    """
    prompt_lower = prompt.lower()

    if "what action would you choose" in prompt_lower:
        # Initial observation-style answer
        return dedent(
            """
            From the recent candles, the price has been respecting a short-term uptrend,
            with higher lows and controlled pullbacks. The detected patterns suggest
            a healthy trend rather than a blow-off top.

            The current candlestick structure shows a constructive reaction near support,
            with buyers stepping in on above-average volume. That aligns reasonably well
            with the model's suggested action.

            Based on this setup, I see a balanced risk–reward profile, but it's not
            a guaranteed outcome — just a probabilistic edge.

            What action would YOU choose here? (Buy / Sell / Hold)
            """
        ).strip()

    if "the human chose" in prompt_lower:
        # Choice evaluation-style answer
        return dedent(
            """
            Your choice is understandable given the recent behaviour of the chart.

            • If you chose BUY: you're aligning with the trend and the model's signal,
              which is often a sensible approach when volatility is controlled.

            • If you chose SELL: that could make sense only if your risk tolerance is low
              or you expect a sharp reversal. In that case, it's important to define a
              clear re-entry plan if price bounces.

            • If you chose HOLD: this is a neutral stance that avoids over-trading.
              It's reasonable when signals are mixed or you're not fully convinced.

            The key is to be consistent with your own risk profile and time horizon.
            Would you like to proceed with this decision, or reconsider?
            """
        ).strip()

    if "explain more about this setup" in prompt_lower:
        # Deep explanation
        return dedent(
            """
            Let's break the setup down more clearly:

            1) Trend:
               Look at the sequence of highs and lows. Higher highs and higher lows
               indicate an uptrend; lower highs and lows indicate a downtrend.

            2) Candlestick patterns:
               Reversal patterns near key levels (supports/resistances) are more
               meaningful than the same patterns in the middle of a range.

            3) Volume:
               Rising volume on moves in the direction of the trend strengthens
               the signal; fading volume can signal exhaustion.

            4) Risk–reward:
               Before choosing Buy/Sell/Hold, think:
               - Where is my stop-loss?
               - Where is my first target?
               - Is the potential reward at least 2–3x the risk?

            Now that you’ve seen this breakdown, what would YOU choose here,
            given your risk tolerance? (Buy / Sell / Hold)
            """
        ).strip()

    # Fallback generic answer
    return (
        "I received your request and generated a generic response. "
        "Once the real LLM is connected, this will be replaced with "
        "a richer, context-aware explanation."
    )
