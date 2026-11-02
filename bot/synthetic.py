"""Deterministic synthetic market (regime-switching random walks).

Used for: offline demo (`run_bot.py --demo`), unit/e2e tests, offline backtests.
It has NO predictive meaning - results on synthetic data prove the plumbing works,
not that a strategy makes money.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = {"BTC/USDT": 60000.0, "ETH/USDT": 3000.0, "SOL/USDT": 150.0,
                   "BNB/USDT": 600.0, "XRP/USDT": 0.55}
TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
         "1h": 3_600_000}


def _parse_tf(tf: str) -> int:
    if tf in TF_MS:
        return TF_MS[tf]
    unit = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[tf[-1]]
    return int(tf[:-1]) * unit


def generate_history(base_price: float, n: int, timeframe: str = "5m", seed: int = 0,
                     start: Optional[datetime] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tf_ms = _parse_tf(timeframe)
    start = start or datetime(2026, 3, 1, tzinfo=timezone.utc)
    t0 = int(start.timestamp() * 1000)
    # regimes: 0 chop, 1 up, 2 down ; sticky Markov chain
    P = np.array([[0.985, 0.0075, 0.0075], [0.01, 0.985, 0.005], [0.01, 0.005, 0.985]])
    drift = np.array([0.0, 0.00035, -0.00035])
    vol = np.array([0.0012, 0.0016, 0.0020])
    reg = np.zeros(n, dtype=int)
    for i in range(1, n):
        reg[i] = rng.choice(3, p=P[reg[i - 1]])
    rets = drift[reg] + vol[reg] * rng.standard_normal(n)
    close = base_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[base_price], close[:-1]])
    wick = np.abs(rng.standard_normal((n, 2))) * vol[reg][:, None] * 0.6
    high = np.maximum(open_, close) * (1 + wick[:, 0])
    low = np.minimum(open_, close) * (1 - wick[:, 1])
    base_q = rng.uniform(150_000, 400_000)                         # quote volume / candle
    burst = 1 + 6 * np.abs(rets) / vol[reg] * rng.uniform(0.2, 1.0, n) * (rng.random(n) < 0.15)
    volume = base_q / close * rng.lognormal(0, 0.35, n) * (burst + (rets > 0) * 0.15)
    ts = pd.to_datetime(t0 + np.arange(n) * tf_ms, unit="ms", utc=True)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


class SyntheticData:
    """Replays a pre-generated market one candle at a time (call `advance()`)."""

    def __init__(self, symbols: Optional[Dict[str, float]] = None, timeframe: str = "5m",
                 seed: int = 7, history: int = 400, total: int = 6000):
        self.timeframe = timeframe
        self.tf_ms = _parse_tf(timeframe)
        self.symbols = symbols or dict(DEFAULT_SYMBOLS)
        self.cursor = history                       # number of CLOSED candles visible
        self.total = total
        self.frames = {s: generate_history(p, total, timeframe, seed + i)
                       for i, (s, p) in enumerate(self.symbols.items())}

    # ---- sim control
    def advance(self, n: int = 1) -> bool:
        if self.cursor + n > self.total:
            return False
        self.cursor += n
        return True

    # ---- data-source interface
    def now(self) -> datetime:
        last_open = self.frames[next(iter(self.frames))]["timestamp"].iloc[self.cursor - 1]
        return (last_open + timedelta(milliseconds=self.tf_ms, seconds=3)).to_pydatetime()

    def timeframe_ms(self, timeframe: str) -> int:
        return _parse_tf(timeframe)

    def tickers(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for s, df in self.frames.items():
            win = df.iloc[max(0, self.cursor - 288):self.cursor]
            out[s] = {"last": float(win["close"].iloc[-1]),
                      "quoteVolume": float((win["close"] * win["volume"]).sum())}
        return out

    def liquid_symbols(self, min_quote_volume: float, limit: int, watchlist: Sequence[str] = (),
                       tickers: Optional[dict] = None) -> List[str]:
        tk = tickers if tickers is not None else self.tickers()
        if watchlist:
            return [s for s in watchlist if s in tk][:limit]
        ranked = sorted(((s, v["quoteVolume"]) for s, v in tk.items()
                         if v["quoteVolume"] >= min_quote_volume), key=lambda x: -x[1])
        return [s for s, _ in ranked[:limit]]

    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        df = self.frames[symbol]
        return df.iloc[max(0, self.cursor - limit):self.cursor].reset_index(drop=True).copy()

    def round_qty(self, symbol: str, qty: float) -> float:
        return float(np.floor(qty * 1e6) / 1e6)

    def min_notional(self, symbol: str) -> float:
        return 5.0
