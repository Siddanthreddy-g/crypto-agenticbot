"""Paper broker (spot, long-only) with fees, slippage, gap-aware exits.

Exit rules (conservative):
  * Only candles that START AFTER the entry candle are used (no stale / look-ahead exits).
  * Every missed candle since entry is replayed in order, so a restart / outage cannot skip a stop.
  * If stop and target are both touched inside one candle, the stop is assumed first.
  * Gaps: stop fills at min(stop, open); target fills at max(target, open).
  * Time stop after `max_hold_candles`.
  * `evaluate_price` handles intra-candle stop/target checks using the live ticker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .journal import Journal


@dataclass
class Position:
    trade_id: int
    symbol: str
    qty: float
    entry_price: float
    stop_price: float
    target_price: float
    entry_fee: float
    entry_time: str
    entry_candle_ts: int


def df_to_candles(df: pd.DataFrame) -> List[Dict[str, float]]:
    out = []
    for r in df.itertuples(index=False):
        out.append({"ts": int(pd.Timestamp(r.timestamp).timestamp() * 1000), "open": float(r.open),
                    "high": float(r.high), "low": float(r.low), "close": float(r.close)})
    return out


class PaperBroker:
    def __init__(self, initial_cash: float, fee_rate: float, slippage_rate: float, journal: Journal):
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.journal = journal
        journal.initialize_portfolio(initial_cash)
        s = journal.load_portfolio()
        self.initial_cash = s["initial_cash"]
        self.cash = s["cash"]
        self.realized_pnl = s["realized_pnl"]
        self.closed_trades = s["closed_trades"]
        self.wins = s["wins"]
        self.losses = s["losses"]
        self.positions: Dict[str, Position] = {
            sym: Position(int(t["id"]), t["symbol"], float(t["qty"]), float(t["entry_price"]),
                          float(t["stop_price"]), float(t["target_price"]), float(t["entry_fee"]),
                          t["entry_time"], int(t["entry_candle_ts"] or 0))
            for sym, t in s["positions"].items()
        }

    # ------------------------------------------------------------- accounting
    def equity(self, marks: Dict[str, float]) -> float:
        return self.cash + sum(p.qty * marks.get(s, p.entry_price) for s, p in self.positions.items())

    def open_notional(self, marks: Dict[str, float]) -> float:
        return sum(p.qty * marks.get(s, p.entry_price) for s, p in self.positions.items())

    # ------------------------------------------------------------------ entry
    def enter_long(self, symbol: str, qty: float, reference_price: float, stop_pct: float,
                   target_pct: float, reason: str, now_iso: str, entry_candle_ts: int,
                   meta: Optional[Dict[str, Any]] = None, source: str = "rules") -> Optional[Position]:
        if symbol in self.positions or qty <= 0 or reference_price <= 0:
            return None
        fill = reference_price * (1 + self.slippage_rate)
        qty = min(qty, self.cash / (fill * (1 + self.fee_rate)))
        if qty <= 0:
            return None
        notional = qty * fill
        fee = notional * self.fee_rate
        total = notional + fee
        if total > self.cash + 1e-9:
            return None
        stop = fill * (1 - stop_pct)
        target = fill * (1 + target_pct)
        tid = self.journal.add_trade({
            "symbol": symbol, "entry_time": now_iso, "qty": qty, "entry_price": fill,
            "stop_price": stop, "target_price": target, "entry_fee": fee,
            "entry_candle_ts": entry_candle_ts, "signal_reason": reason, "meta": meta, "source": source})
        self.cash -= total
        pos = Position(tid, symbol, qty, fill, stop, target, fee, now_iso, entry_candle_ts)
        self.positions[symbol] = pos
        return pos

    # ------------------------------------------------------------------- exit
    def _close(self, p: Position, reference: float, why: str, now_iso: str) -> Dict[str, Any]:
        fill = reference * (1 - self.slippage_rate)
        gross = p.qty * fill
        fee = gross * self.fee_rate
        pnl = (gross - fee) - (p.qty * p.entry_price + p.entry_fee)
        # persist first, then mutate memory
        self.journal.close_trade(p.trade_id, now_iso, fill, fee, pnl, why)
        self.cash += gross - fee
        self.realized_pnl += pnl
        self.closed_trades += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        del self.positions[p.symbol]
        cost_basis = p.qty * p.entry_price + p.entry_fee
        return {"trade_id": p.trade_id, "symbol": p.symbol, "reason": why, "pnl": pnl,
                "pnl_pct": pnl / cost_basis if cost_basis else 0.0, "entry_price": p.entry_price,
                "exit_price": fill, "qty": p.qty, "exit_time": now_iso}

    def evaluate_candles(self, symbol: str, candles: List[Dict[str, float]], tf_ms: int,
                         max_hold_candles: int, now_iso: str) -> Optional[Dict[str, Any]]:
        """Replay closed candles newer than the entry candle. Returns closed-trade dict or None."""
        p = self.positions.get(symbol)
        if not p:
            return None
        for c in candles:
            if c["ts"] <= p.entry_candle_ts:
                continue
            res = self.evaluate_candle(symbol, c, tf_ms, max_hold_candles, now_iso)
            if res:
                return res
        return None

    def evaluate_candle(self, symbol: str, c: Dict[str, float], tf_ms: int,
                        max_hold_candles: int, now_iso: str) -> Optional[Dict[str, Any]]:
        p = self.positions.get(symbol)
        if not p or c["ts"] <= p.entry_candle_ts:
            return None
        o, lo, hi = float(c.get("open", p.stop_price)), float(c["low"]), float(c["high"])
        if lo <= p.stop_price:
            return self._close(p, min(p.stop_price, o), "stop_loss", now_iso)
        if hi >= p.target_price:
            return self._close(p, max(p.target_price, o), "take_profit", now_iso)
        held = (c["ts"] - p.entry_candle_ts) // tf_ms
        if max_hold_candles and held >= max_hold_candles:
            return self._close(p, float(c["close"]), "time_stop", now_iso)
        return None

    def evaluate_price(self, symbol: str, price: float, now_iso: str) -> Optional[Dict[str, Any]]:
        """Intra-candle check against the live ticker price."""
        p = self.positions.get(symbol)
        if not p or price <= 0:
            return None
        if price <= p.stop_price:
            return self._close(p, price, "stop_loss", now_iso)
        if price >= p.target_price:
            return self._close(p, p.target_price, "take_profit", now_iso)
        return None

    def close_at_market(self, symbol: str, price: float, why: str, now_iso: str) -> Optional[Dict[str, Any]]:
        p = self.positions.get(symbol)
        if not p or price <= 0:
            return None
        return self._close(p, price, why, now_iso)
