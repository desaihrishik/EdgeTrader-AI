// src/api.ts
import axios from "axios";

const BASE_URL = "http://localhost:8000";

export type RiskProfile = "low" | "medium" | "high";

export interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  patterns: string[];
  signal: number;
}

export interface LatestSignalResponse {
  date: string;
  latest_close: number;
  action: "BUY" | "SELL" | "HOLD";
  signal_value: -1 | 0 | 1;
  confidence: number;
  probas: Record<string, number>;
  suggested_shares: number;
  capital_used: number;
  patterns: string[];
  risk_profile: RiskProfile;
  budget: number;
  explanation: string;
}

export interface AgenticForecast {
  p10: number;
  p50: number;
  p90: number;
}

export interface SellAnalysis {
  current_return: number;
  potential_downside_if_hold: number;
  expected_future_ret: number;
  best_case_ret: number;
}

export interface AgenticSignalResponse {
  date: string;
  latest_close: number;
  action: "BUY" | "SELL" | "HOLD";
  signal_value: -1 | 0 | 1;
  confidence: number;
  probas: Record<string, number>;
  patterns: string[];
  pattern_strength: number;
  trend_strength: number;
  sentiment_label: string;
  sentiment_score: number;
  sentiment_strength: number;
  horizon_days: number;
  forecast: AgenticForecast;
  suggested_shares: number;
  capital_used: number;
  sell_analysis: SellAnalysis | null;
  entry_price_used: number | null;
  risk_profile: RiskProfile;
  budget: number;
  explanation: string;
}

export async function fetchCandles(limit = 120): Promise<Candle[]> {
  const res = await axios.get<{ candles: Candle[] }>(
    `${BASE_URL}/api/nvda/candles`,
    { params: { limit } }
  );
  return res.data.candles;
}

export async function fetchLatestSignal(
  budget: number,
  risk: RiskProfile
): Promise<LatestSignalResponse> {
  const res = await axios.get<LatestSignalResponse>(
    `${BASE_URL}/api/nvda/latest_signal`,
    { params: { budget, risk } }
  );
  return res.data;
}

export async function fetchAgenticSignal(
  budget: number,
  risk: RiskProfile,
  entryPrice?: number
): Promise<AgenticSignalResponse> {
  const res = await axios.get<AgenticSignalResponse>(
    `${BASE_URL}/api/nvda/agentic_signal`,
    {
      params: { budget, risk, entry_price: entryPrice ?? null },
    }
  );
  return res.data;
}

// LLM → observe/inference only
export async function llmEvaluate(body: object): Promise<{ analysis: string }> {
  const res = await axios.post(`${BASE_URL}/api/llm/observe`, body);
  return res.data; // { analysis: "..." }
}
