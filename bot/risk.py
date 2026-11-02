"""Deterministic risk layer. The LLM can never override anything in here.

RiskManager  - pure position sizing + per-entry veto rules.
RiskGuard    - persistent state: daily loss reference, peak equity, drawdown halt,
               consecutive-loss pause, per-symbol cooldown, kill switch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Tuple

from .config import Config
from .journal import Journal


@dataclass
class RiskDecision:
    qty: float
    approved: bool
    reason: str


class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def size_position(self, *, equity: float, cash: float, entry_price: float, stop_pct: float,
                      open_positions: int, open_notional: float, size_multiplier: float = 1.0,
                      round_qty: Optional[Callable[[float], float]] = None,
                      min_notional: Optional[float] = None) -> RiskDecision:
        c = self.cfg
        if min(equity, cash, entry_price) <= 0 or stop_pct <= 0:
            return RiskDecision(0.0, False, "invalid_inputs")
        if open_positions >= c.max_open_positions:
            return RiskDecision(0.0, False, "max_open_positions")
        # risk-based, notional cap, exposure cap, cash (with fee/slippage buffer)
        qty_risk = equity * c.risk_per_trade / (entry_price * stop_pct)
        qty_cap = equity * c.max_position_fraction / entry_price
        room = max(0.0, equity * c.max_total_exposure - open_notional)
        qty_exposure = room / entry_price
        qty_cash = cash / (entry_price * (1 + c.fee_rate + c.slippage_rate) * 1.001)
        qty = min(qty_risk, qty_cap, qty_exposure, qty_cash)
        qty *= max(0.0, min(1.0, size_multiplier))          # LLM may only shrink
        if round_qty is not None:
            qty = round_qty(qty)
        if qty <= 0:
            return RiskDecision(0.0, False, "size_zero_or_exposure_cap")
        floor = max(c.min_notional, min_notional or 0.0)
        if qty * entry_price < floor:
            return RiskDecision(0.0, False, f"below_min_notional({qty * entry_price:.2f}<{floor:.2f})")
        return RiskDecision(qty, True, "approved")


class RiskGuard:
    def __init__(self, cfg: Config, journal: Journal):
        self.cfg = cfg
        self.j = journal

    # ------------------------------------------------------------ halt state
    @property
    def halted(self) -> bool:
        return self.j.kv_get("halted", "0") == "1"

    @property
    def halt_reason(self) -> str:
        return self.j.kv_get("halt_reason", "") or ""

    def halt(self, reason: str) -> None:
        self.j.kv_set("halted", "1")
        self.j.kv_set("halt_reason", reason)

    def resume(self, reset_peak_to: Optional[float] = None) -> None:
        self.j.kv_set("halted", "0")
        self.j.kv_set("halt_reason", "")
        self.j.kv_set("consecutive_losses", 0)
        self.j.kv_set("pause_until", "")
        if reset_peak_to is not None:
            self.j.kv_set("peak_equity", reset_peak_to)

    # ---------------------------------------------------------- daily / peak
    def roll_day(self, now: datetime, equity: float) -> Tuple[bool, Optional[str]]:
        """Returns (rolled, previous_day). previous_day is None on the very first run."""
        day = now.astimezone(timezone.utc).date().isoformat()
        prev = self.j.kv_get("day")
        if prev != day:
            self.j.kv_set("day", day)
            self.j.kv_set("day_start_equity", equity)
            return True, prev
        return False, prev

    @property
    def day_start_equity(self) -> float:
        return self.j.kv_float("day_start_equity", 0.0)

    @property
    def peak_equity(self) -> float:
        return self.j.kv_float("peak_equity", 0.0)

    def update_equity(self, equity: float) -> Optional[str]:
        """Track peak; trip the max-drawdown halt. Returns halt reason if tripped now."""
        peak = max(self.peak_equity, equity)
        self.j.kv_set("peak_equity", peak)
        if not self.halted and peak > 0 and equity <= peak * (1 - self.cfg.max_drawdown):
            reason = f"max_drawdown_reached ({(1 - equity / peak) * 100:.1f}% from peak {peak:.2f})"
            self.halt(reason)
            return reason
        return None

    # ----------------------------------------------------- losses / cooldowns
    def record_close(self, symbol: str, pnl: float, now: datetime, tf_ms: int) -> None:
        n = int(self.j.kv_float("consecutive_losses", 0))
        n = n + 1 if pnl <= 0 else 0
        self.j.kv_set("consecutive_losses", n)
        if n >= self.cfg.max_consecutive_losses:
            until = now + timedelta(minutes=self.cfg.loss_pause_minutes)
            self.j.kv_set("pause_until", until.isoformat())
            self.j.kv_set("consecutive_losses", 0)
        cool = now + timedelta(milliseconds=tf_ms * self.cfg.symbol_cooldown_candles)
        self.j.kv_set(f"cooldown:{symbol}", cool.isoformat())

    def blocked_reason(self, now: datetime, equity: float, symbol: Optional[str] = None) -> Optional[str]:
        if self.halted:
            return f"HALTED: {self.halt_reason}"
        if self.cfg.kill_switch_file and os.path.exists(self.cfg.kill_switch_file):
            return "kill_switch_file_present"
        ds = self.day_start_equity
        if ds > 0 and equity <= ds * (1 - self.cfg.max_daily_loss):
            return "daily_loss_limit_reached"
        pu = self.j.kv_get("pause_until", "")
        if pu:
            try:
                if now < datetime.fromisoformat(pu):
                    return f"loss_streak_pause_until_{pu}"
            except ValueError:
                pass
        if symbol:
            cd = self.j.kv_get(f"cooldown:{symbol}", "")
            if cd:
                try:
                    if now < datetime.fromisoformat(cd):
                        return "symbol_cooldown"
                except ValueError:
                    pass
        return None
