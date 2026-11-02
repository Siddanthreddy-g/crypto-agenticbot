"""Rule-based long-only strategy (transparent, deterministic, auditable).

Hypothesis (NOT validated, no edge is claimed):
  EMA9 > EMA21, close > EMA50, RSI in [rsi_min, rsi_max], close > session VWAP,
  volume > k * 20-candle average.
Stops/targets are ATR-based and the trade is rejected if round-trip costs eat
too much of the target (fees + slippage are modelled everywhere).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from .config import Config
from .indicators import enrich

MIN_CANDLES = 120


@dataclass(frozen=True)
class StrategyParams:
    rsi_min: float = 50.0
    rsi_max: float = 68.0
    volume_multiplier: float = 1.2
    atr_stop_mult: float = 1.5
    reward_risk: float = 2.0
    min_stop_pct: float = 0.008
    max_stop_pct: float = 0.03
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_cost_to_target: float = 0.35
    min_net_rr: float = 1.1

    @classmethod
    def from_config(cls, cfg: Config) -> "StrategyParams":
        return cls(
            rsi_min=cfg.rsi_min, rsi_max=cfg.rsi_max,
            volume_multiplier=cfg.volume_multiplier, atr_stop_mult=cfg.atr_stop_mult,
            reward_risk=cfg.reward_risk, min_stop_pct=cfg.min_stop_pct,
            max_stop_pct=cfg.max_stop_pct, fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate, max_cost_to_target=cfg.max_cost_to_target,
            min_net_rr=cfg.min_net_rr,
        )

    def round_trip_cost(self) -> float:
        return 2.0 * (self.fee_rate + self.slippage_rate)


@dataclass
class Signal:
    symbol: str
    candle_ts: int                # open time (ms) of the closed signal candle
    price: float                  # close of the signal candle
    should_buy: bool
    reason: str
    score: float = 0.0
    atr: float = 0.0
    stop_pct: float = 0.0
    target_pct: float = 0.0
    indicators: Dict[str, float] = field(default_factory=dict)


def _bad(*vals) -> bool:
    return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals)


def signal_from_values(symbol: str, candle_ts: int, close: float, ema_fast: float,
                       ema_slow: float, ema_trend: float, rsi: float, vwap: float,
                       volume: float, vol_avg: float, atr: float,
                       p: StrategyParams) -> Signal:
    """Pure function: one candle's indicator values -> Signal. Used by live + backtest."""
    ind = {"close": close, "ema_fast": ema_fast, "ema_slow": ema_slow, "ema_trend": ema_trend,
           "rsi": rsi, "vwap": vwap, "volume": volume, "vol_avg20": vol_avg, "atr": atr}
    if _bad(close, ema_fast, ema_slow, ema_trend, rsi, vwap, volume, vol_avg, atr) or close <= 0:
        return Signal(symbol, candle_ts, close, False, "indicators_not_ready", indicators=ind)

    vol_ratio = volume / vol_avg if vol_avg > 0 else 0.0
    ind["vol_ratio"] = vol_ratio
    checks = {
        "ema_trend": ema_fast > ema_slow,
        "above_ema50": close > ema_trend,
        "rsi_range": p.rsi_min <= rsi <= p.rsi_max,
        "above_vwap": close > vwap,
        "volume_confirm": vol_ratio > p.volume_multiplier,
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        return Signal(symbol, candle_ts, close, False, "filters_not_met:" + ",".join(failed),
                      atr=atr, indicators=ind)

    stop_pct = min(max(p.atr_stop_mult * atr / close, p.min_stop_pct), p.max_stop_pct)
    target_pct = stop_pct * p.reward_risk
    cost = p.round_trip_cost()
    if cost / target_pct > p.max_cost_to_target:
        return Signal(symbol, candle_ts, close, False,
                      f"costs_too_high:{cost / target_pct:.2f}", atr=atr, stop_pct=stop_pct,
                      target_pct=target_pct, indicators=ind)
    net_rr = (target_pct - cost) / (stop_pct + cost)
    ind["net_rr"] = net_rr
    if net_rr < p.min_net_rr:
        return Signal(symbol, candle_ts, close, False, f"net_rr_too_low:{net_rr:.2f}",
                      atr=atr, stop_pct=stop_pct, target_pct=target_pct, indicators=ind)

    trend_strength = (ema_fast / ema_slow - 1.0) * 100.0
    score = min(vol_ratio, 5.0) + max(trend_strength, 0.0) * 5.0
    return Signal(symbol, candle_ts, close, True,
                  f"EMA9>EMA21; close>EMA50; RSI {rsi:.1f}; close>VWAP; vol x{vol_ratio:.2f}",
                  score=score, atr=atr, stop_pct=stop_pct, target_pct=target_pct, indicators=ind)


def ts_ms(ts) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


def signal_for_row(symbol: str, enriched: pd.DataFrame, i: int, p: StrategyParams) -> Signal:
    r = enriched.iloc[i]
    return signal_from_values(
        symbol, ts_ms(r["timestamp"]), float(r["close"]), float(r["ema_fast"]),
        float(r["ema_slow"]), float(r["ema_trend"]), float(r["rsi"]), float(r["vwap"]),
        float(r["volume"]), float(r["vol_avg20"]), float(r["atr"]), p)


def evaluate(symbol: str, df: pd.DataFrame, p: StrategyParams) -> Optional[Signal]:
    """Evaluate the LAST (closed) candle of df."""
    if df is None or len(df) < MIN_CANDLES:
        return None
    e = enrich(df)
    return signal_for_row(symbol, e, len(e) - 1, p)
