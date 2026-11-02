"""Agent roster. Each agent has one narrow job; only the LLM agent is non-deterministic and it
can only veto / shrink. Every decision is written to the `decisions` table for audit.

  MarketScannerAgent   - picks the liquid universe
  StrategyAnalystAgent - deterministic signal on the last closed candle
  LLMReviewAgent       - bounded second opinion (approve / reduce / reject), rate-limited
  RiskSupervisorAgent  - deterministic sizing + vetoes + halts (final authority)
  PerformanceAgent     - metrics
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd

from .config import Config
from .journal import Journal
from .llm import LLMError, ReviewResult
from .risk import RiskGuard, RiskManager
from .stats import max_drawdown_pct, trade_stats
from .strategy import MIN_CANDLES, Signal, StrategyParams, evaluate

log = logging.getLogger("bot.agents")


class MarketScannerAgent:
    def __init__(self, data: Any, cfg: Config):
        self.data, self.cfg = data, cfg

    def run(self, tickers: Dict[str, Dict[str, float]]) -> List[str]:
        return self.data.liquid_symbols(self.cfg.min_quote_volume, self.cfg.max_symbols,
                                        self.cfg.watchlist, tickers=tickers)


class StrategyAnalystAgent:
    def __init__(self, data: Any, cfg: Config, params: StrategyParams):
        self.data, self.cfg, self.params = data, cfg, params

    def analyze(self, symbol: str) -> Tuple[Optional[Signal], pd.DataFrame]:
        df = self.data.candles(symbol, self.cfg.timeframe, self.cfg.candle_limit)
        if df is None or len(df) < MIN_CANDLES:
            return None, df
        return evaluate(symbol, df, self.params), df


@dataclass
class Verdict:
    approved: bool
    size_multiplier: float
    rationale: str
    source: str
    decision: str


class LLMReviewAgent:
    """Wraps a reviewer (LLM or rules-only) with rate-limit, confidence floor and fail mode."""

    def __init__(self, reviewer: Any, cfg: Config, journal: Journal):
        self.reviewer, self.cfg, self.j = reviewer, cfg, journal
        self._calls: Deque[float] = deque()

    def _rate_ok(self) -> bool:
        now = time.time()
        while self._calls and now - self._calls[0] > 3600:
            self._calls.popleft()
        return len(self._calls) < self.cfg.llm_max_calls_per_hour

    def review(self, sig: Signal, ctx: Dict[str, Any], now_iso: str) -> Verdict:
        is_rules = getattr(self.reviewer, "model", "") == "none"
        if not is_rules and not self._rate_ok():
            v = self._fail("rate_limited")
        else:
            try:
                if not is_rules:
                    self._calls.append(time.time())
                r: ReviewResult = self.reviewer.review(ctx)
                v = self._from_result(r)
            except LLMError as exc:
                log.warning("LLM review failed for %s: %s", sig.symbol, exc)
                v = self._fail(str(exc)[:200])
        self.j.add_decision(now_iso, sig.symbol, "llm_review", v.decision,
                            {"approved": v.approved, "mult": v.size_multiplier, "source": v.source,
                             "rationale": v.rationale, "signal": sig.reason, "score": round(sig.score, 3)})
        return v

    def _from_result(self, r: ReviewResult) -> Verdict:
        if r.source == "rules":
            return Verdict(True, 1.0, r.rationale, "rules", "approve")
        if r.decision == "reject":
            return Verdict(False, 0.0, r.rationale, "llm", "reject")
        if r.confidence < self.cfg.llm_min_confidence:
            return Verdict(False, 0.0, f"low_confidence({r.confidence:.2f}): {r.rationale}", "llm", "reject")
        mult = max(0.0, min(1.0, float(r.size_multiplier)))        # defence in depth: never > 1
        return Verdict(True, mult, r.rationale, "llm", r.decision)

    def _fail(self, why: str) -> Verdict:
        if self.cfg.llm_fail_mode == "approve":
            return Verdict(True, 1.0, f"LLM unavailable ({why}) - falling back to rules", "fallback", "approve")
        return Verdict(False, 0.0, f"LLM unavailable ({why}) - fail-closed", "fallback", "reject")


class RiskSupervisorAgent:
    def __init__(self, manager: RiskManager, guard: RiskGuard):
        self.manager, self.guard = manager, guard

    def size(self, **kw):
        return self.manager.size_position(**kw)


class PerformanceAgent:
    def __init__(self, journal: Journal):
        self.j = journal

    def summary(self) -> Dict[str, Any]:
        closed = self.j.get_closed_trades()
        s = trade_stats([t["pnl"] or 0.0 for t in closed])
        curve = [r["equity"] for r in self.j.equity_curve(20000)]
        s["max_drawdown_pct"] = max_drawdown_pct(curve)
        return s


# ---------------------------------------------------------------- LLM context
def _pct(a: float, b: float) -> float:
    return round((a / b - 1.0) * 100, 3) if b else 0.0


def build_review_context(sig: Signal, df: pd.DataFrame, cfg: Config, perf: Dict[str, Any],
                         open_positions: List[str], benchmark: Optional[Dict[str, Any]],
                         day_pnl_pct: float, now_iso: str) -> Dict[str, Any]:
    c = df["close"]
    n = len(c)

    def chg(k: int) -> Optional[float]:
        return _pct(float(c.iloc[-1]), float(c.iloc[-1 - k])) if n > k else None

    last = df.tail(8)
    candles = [[round(float(r.open), 8), round(float(r.high), 8), round(float(r.low), 8),
                round(float(r.close), 8), round(float(r.volume), 4)] for r in last.itertuples()]
    ind = {k: round(float(v), 6) for k, v in sig.indicators.items()}
    vw, ef = ind.get("vwap"), ind.get("ema_fast")
    return {
        "time_utc": now_iso, "symbol": sig.symbol, "timeframe": cfg.timeframe,
        "signal": {"reason": sig.reason, "score": round(sig.score, 3), "price": sig.price,
                   "stop_pct": round(sig.stop_pct * 100, 3), "target_pct": round(sig.target_pct * 100, 3),
                   "round_trip_cost_pct": round(2 * (cfg.fee_rate + cfg.slippage_rate) * 100, 3)},
        "indicators": ind,
        "extension": {"pct_above_vwap": _pct(sig.price, vw) if vw else None,
                      "pct_above_ema_fast": _pct(sig.price, ef) if ef else None},
        "price_change_pct": {"last_1_candle": chg(1), "last_12_candles": chg(12), "last_48_candles": chg(48)},
        "recent_candles_ohlcv_oldest_first": candles,
        "benchmark_regime": benchmark,
        "portfolio": {"open_positions": open_positions, "max_positions": cfg.max_open_positions,
                      "today_pnl_pct": round(day_pnl_pct, 3),
                      "closed_trades": perf.get("trades", 0),
                      "win_rate_pct": round(perf.get("win_rate_pct", 0.0), 1),
                      "profit_factor": None if perf.get("profit_factor") in (None, float("inf"))
                      else round(perf["profit_factor"], 2),
                      "max_drawdown_pct": round(perf.get("max_drawdown_pct", 0.0), 2)},
    }
