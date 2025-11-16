// src/App.tsx

import { useEffect, useState } from "react";
import ReactApexChart from "react-apexcharts";
import type { ApexOptions } from "apexcharts";

import {
  fetchCandles,
  fetchLatestSignal,
  fetchAgenticSignal,
  llmEvaluate,
} from "./api";

import type {
  Candle,
  LatestSignalResponse,
  RiskProfile,
  AgenticSignalResponse,
} from "./api";

import "./App.css";

function formatPercent(p: number) {
  return (p * 100).toFixed(1) + "%";
}

type RangeKey = "1M" | "3M" | "6M" | "1Y";

const RANGE_LIMITS: Record<RangeKey, number> = {
  "1M": 22,
  "3M": 66,
  "6M": 132,
  "1Y": 252,
};

function App() {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [signal, setSignal] = useState<LatestSignalResponse | null>(null);
  const [agentic, setAgentic] = useState<AgenticSignalResponse | null>(null);

  const [loading, setLoading] = useState(true);
  const [agenticLoading, setAgenticLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [agenticError, setAgenticError] = useState<string | null>(null);

  const [budget, setBudget] = useState(1000);
  const [risk, setRisk] = useState<RiskProfile>("medium");
  const [entryPrice, setEntryPrice] = useState<number | undefined>(undefined);

  const [range, setRange] = useState<RangeKey>("3M");

  // ----------- LLM State ---------------
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);
  const [llmOutput, setLlmOutput] = useState<string | null>(null);

  // ---------------------------------------------------------
  // Load candles + latest signal
  // ---------------------------------------------------------
  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError(null);

        const limit = RANGE_LIMITS[range];

        const [candleData, latestSignal] = await Promise.all([
          fetchCandles(limit),
          fetchLatestSignal(budget, risk),
        ]);

        setCandles(candleData);
        setSignal(latestSignal);
      } catch (err) {
        console.error(err);
        setError("Failed to load data from backend.");
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [budget, risk, range]);

  // ---------------------------------------------------------
  // Load agentic signal
  // ---------------------------------------------------------
  useEffect(() => {
    async function loadAgentic() {
      try {
        setAgenticLoading(true);
        setAgenticError(null);

        const res = await fetchAgenticSignal(budget, risk, entryPrice);
        setAgentic(res);
      } catch (err) {
        console.error(err);
        setAgenticError("Failed to load agentic signal.");
      } finally {
        setAgenticLoading(false);
      }
    }

    loadAgentic();
  }, [budget, risk, entryPrice]);

  const latestCandle = candles[candles.length - 1];

  // ---------------------------------------------------------
  // Candlestick chart
  // ---------------------------------------------------------
  const chartOptions: ApexOptions = {
    chart: {
      type: "candlestick",
      height: 400,
      background: "#020617",
      foreColor: "#e5e7eb",
      toolbar: {
        tools: {
          pan: true,
          zoom: true,
          zoomin: true,
          zoomout: true,
          reset: true,
        },
      },
    },
    grid: {
      borderColor: "#1f2933",
      strokeDashArray: 3,
    },
    xaxis: {
      type: "datetime",
      labels: { style: { colors: "#9ca3af" } },
      axisBorder: { color: "#1f2937" },
      axisTicks: { color: "#1f2937" },
    },
    yaxis: {
      tooltip: { enabled: true },
      labels: {
        formatter: (val) => `$${val.toFixed(0)}`,
        style: { colors: "#9ca3af" },
      },
    },
    plotOptions: {
      candlestick: {
        colors: { upward: "#22c55e", downward: "#ef4444" },
        wick: { useFillColor: true },
      },
    },
    tooltip: {
      theme: "dark",
      shared: true,
      custom: ({ dataPointIndex }) => {
        const candle = candles[dataPointIndex];
        if (!candle) return "";
        const patternText =
          candle.patterns?.length > 0
            ? `<div style="margin-top:4px;font-size:11px;color:#fbbf24">
                 Patterns: ${candle.patterns.join(", ")}
               </div>`
            : "";

        return `
          <div style="padding:8px;font-size:12px;color:#e5e7eb;background:#020617;">
            <div><strong>${candle.date}</strong></div>
            <div>O: ${candle.open.toFixed(2)}  
                 H: ${candle.high.toFixed(2)}  
                 L: ${candle.low.toFixed(2)}  
                 C: ${candle.close.toFixed(2)}</div>
            <div>Vol: ${candle.volume.toLocaleString()}</div>
            ${patternText}
          </div>
        `;
      },
    },
  };

  const chartSeries = [
    {
      name: "NVDA",
      data: candles.map((c) => ({
        x: new Date(c.date),
        y: [c.open, c.high, c.low, c.close],
      })),
    },
  ];

  // ---------------------------------------------------------
  // Agentic projection chart values
  // ---------------------------------------------------------
  const agenticCharts =
    agentic && latestCandle
      ? {
          expected: [latestCandle.close, agentic.forecast.p50],
          best: [latestCandle.close, agentic.forecast.p90],
          worst: [latestCandle.close, agentic.forecast.p10],
          days: [0, agentic.horizon_days],
        }
      : null;

  const unifiedOptions: ApexOptions = {
    chart: {
      type: "line",
      background: "#020617",
      foreColor: "#e5e7eb",
      zoom: { enabled: false },
    },
    colors: ["#4F8BFF", "#00C851", "#ff4444"],
    stroke: { curve: "smooth", width: 3 },
    xaxis: {
      categories: agenticCharts ? agenticCharts.days : [0, 1],
      title: { text: "Days" },
      labels: { style: { colors: "#9ca3af" } },
    },
    yaxis: {
      labels: {
        formatter: (val) => `$${val.toFixed(2)}`,
        style: { colors: "#9ca3af" },
      },
    },
    tooltip: {
      shared: true,
      intersect: false,
      theme: "dark",
      y: { formatter: (val) => `$${val.toFixed(2)}` },
    },
    legend: { position: "top" },
    grid: { borderColor: "#1f2933", strokeDashArray: 3 },
  };

  // ---------------------------------------------------------
  // LLM Evaluate
  // ---------------------------------------------------------
  const handleEvaluateClick = async () => {
    if (!latestCandle || !signal) return;

    try {
      setLlmLoading(true);
      setLlmError(null);
      setLlmOutput(null);

      const body = {
        symbol: "NVDA",
        last_close: latestCandle.close,
        horizon_days: agentic?.horizon_days ?? 20,
        patterns: latestCandle.patterns ?? [],
        sentiment_label: agentic?.sentiment_label ?? null,
        sentiment_score: agentic?.sentiment_score ?? null,
        model_action: signal.action,
        model_confidence: signal.confidence,
      };

      const res = await llmEvaluate(body);
      setLlmOutput(res.analysis);
    } catch (err) {
      console.error(err);
      setLlmError("Failed to contact LLM.");
    } finally {
      setLlmLoading(false);
    }
  };

  // ---------------------------------------------------------
  // UI
  // ---------------------------------------------------------
  return (
    <div className="app-root">
      <header className="app-header">
        <div>
          <h1>EdgeTrader · NVDA Swing AI</h1>
          <p className="app-subtitle">
            Edge-aware AI assistant for NVDA swing trading — uses candles,
            patterns, ML and sentiment to suggest entries & horizons.
          </p>
        </div>

        <div className="header-actions">
          <button
            className="evaluate-btn"
            onClick={handleEvaluateClick}
            disabled={!latestCandle || !signal || llmLoading}
          >
            {llmLoading ? "Evaluating…" : "Evaluate"}
          </button>

          <button className="btn-secondary" onClick={() => window.location.reload()}>
            Refresh
          </button>
        </div>
      </header>

      <main className="app-main">
        {/* ================= LEFT PANEL ================= */}
        <section className="left-panel">
          {/* Candles */}
          <div className="card">
            <div className="card-header">
              <h2>NVDA Candlestick Chart</h2>
              {latestCandle && (
                <span className="tag">
                  Last close: ${latestCandle.close.toFixed(2)} — {latestCandle.date}
                </span>
              )}
            </div>

            <ReactApexChart
              options={chartOptions}
              series={chartSeries}
              type="candlestick"
              height={420}
            />
          </div>

          {/* Signal */}
          <div className="card">
            <div className="card-header">
              <h2>Current NVDA Recommendation</h2>
            </div>

            <div className="controls-row">
              <div className="field">
                <label>Budget (USD)</label>
                <input
                  type="number"
                  value={budget}
                  min={100}
                  step={100}
                  onChange={(e) => setBudget(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label>Risk Profile</label>
                <select
                  value={risk}
                  onChange={(e) => setRisk(e.target.value as RiskProfile)}
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
            </div>

            {loading && <p className="muted">Loading signal…</p>}
            {error && <p className="error">{error}</p>}

            {signal && !loading && !error && (
              <>
                <div className="signal-row">
                  <div>
                    <div className="signal-label">Action</div>
                    <div className={`signal-action action-${signal.action}`}>
                      {signal.action}
                    </div>
                  </div>

                  <div>
                    <div className="signal-label">Confidence</div>
                    <div className="confidence-row">
                      <span className="confidence-value">
                        {formatPercent(signal.confidence)}
                      </span>
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{ width: `${signal.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="signal-row">
                  <div>
                    <div className="signal-label">Suggested quantity</div>
                    <div className="signal-value">
                      {signal.suggested_shares} share
                      {signal.suggested_shares !== 1 && "s"}
                    </div>
                    <div className="muted-small">
                      Capital used: ${signal.capital_used.toFixed(2)}
                    </div>
                  </div>

                  <div>
                    <div className="signal-label">Model price</div>
                    <div className="signal-value">
                      ${signal.latest_close.toFixed(2)}
                    </div>
                    <div className="muted-small">As of {signal.date}</div>
                  </div>
                </div>

                <div className="card-subsection">
                  <div className="signal-label">Explanation</div>
                  <p className="explanation-text">{signal.explanation}</p>
                </div>
              </>
            )}
          </div>
        </section>

        {/* ================= RIGHT PANEL ================= */}
        <section className="right-panel">
          {/* -------- LLM Output -------- */}
          <div className="card">
            <div className="card-header">
              <h2>LLM Evaluation</h2>
            </div>

            {llmError && <p className="error">{llmError}</p>}
            {llmLoading && <p className="muted">Evaluating with LLM…</p>}

            {!llmLoading && !llmOutput && (
              <p className="muted">Press Evaluate to run the LLM.</p>
            )}

            {llmOutput && (
              <div className="llm-bubble llm-bubble-primary" style={{ whiteSpace: "pre-line" }}>
                {llmOutput}
              </div>
            )}
          </div>

          {/* -------- Agentic Projection -------- */}
          <div className="card">
            <div className="card-header">
              <h2>Agentic Horizon & Risk Projection</h2>
            </div>

            <div className="controls-row">
              <div className="field">
                <label>Entry Price (USD, optional)</label>
                <input
                  type="number"
                  value={entryPrice ?? ""}
                  onChange={(e) =>
                    setEntryPrice(e.target.value === "" ? undefined : Number(e.target.value))
                  }
                />
              </div>
            </div>

            {agenticLoading && <p className="muted">Computing…</p>}
            {agenticError && <p className="error">{agenticError}</p>}

            {agentic && agenticCharts && (
              <>
                <div className="signal-row">
                  <div>
                    <div className="signal-label">Agent Action</div>
                    <div className={`signal-action action-${agentic.action}`}>
                      {agentic.action}
                    </div>
                    <div className="muted-small">
                      Horizon: ~{agentic.horizon_days} trading days
                    </div>
                  </div>

                  <div>
                    <div className="signal-label">Sentiment</div>
                    <div className="signal-value">
                      {agentic.sentiment_label} (
                      {formatPercent(agentic.sentiment_score)})
                    </div>
                  </div>
                </div>

                <div className="card-subsection">
                  <div className="signal-label">Price Projection</div>

                  <ReactApexChart
                    type="line"
                    height={300}
                    series={[
                      { name: "Expected (Median)", data: agenticCharts.expected },
                      { name: "Best Case (P90)", data: agenticCharts.best },
                      { name: "Worst Case (P10)", data: agenticCharts.worst },
                    ]}
                    options={unifiedOptions}
                  />
                </div>

                <div className="card-subsection">
                  <div className="signal-label">Agent Explanation</div>
                  <p className="explanation-text">{agentic.explanation}</p>
                </div>
              </>
            )}
          </div>
        </section>
      </main>

      {/* Bottom strip */}
      <section className="bottom-strip">
        <div className="card">
          <div className="card-header">
            <h2>Recent Pattern Candles</h2>
          </div>

          {candles
            .filter((c) => c.patterns?.length > 0)
            .slice(-8)
            .reverse()
            .map((c) => (
              <div key={c.date} className="pattern-strip-item">
                <div className="pattern-date">{c.date}</div>
                <div className="pattern-price">
                  Close: ${c.close.toFixed(2)}
                </div>
                <div className="pattern-tags">
                  {c.patterns.map((p) => (
                    <span key={p} className="pattern-tag">{p}</span>
                  ))}
                </div>
              </div>
            ))}
        </div>
      </section>
    </div>
  );
}

export default App;
