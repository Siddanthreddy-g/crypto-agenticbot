"""Public spot market data through ccxt. No authenticated / private endpoints are used.

Data-source interface (duck-typed; SyntheticData implements the same):
    now() -> aware UTC datetime
    tickers() -> {symbol: {"last": float, "quoteVolume": float}}
    liquid_symbols(min_quote_volume, limit, watchlist) -> [symbol]
    candles(symbol, timeframe, limit) -> DataFrame of CLOSED candles
    timeframe_ms(timeframe) -> int
    round_qty(symbol, qty) -> float ; min_notional(symbol) -> float
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

log = logging.getLogger("bot.data")

STABLE_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "USDE", "USDS", "EUR", "EURI", "AEUR", "PYUSD",
                "UST", "BUSD", "GBP", "TRY", "BRL", "USD1"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")


def _is_tradeable_base(base: str) -> bool:
    if base in STABLE_BASES:
        return False
    return not (len(base) > 4 and base.endswith(LEVERAGED_SUFFIXES))


class ExchangeData:
    def __init__(self, exchange_id: str = "binance", quote_asset: str = "USDT", retries: int = 3):
        import ccxt  # imported lazily so tests / synthetic mode need no network libs

        self.ccxt = ccxt
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Unknown ccxt exchange id: {exchange_id}")
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "spot"}})
        self.quote = quote_asset.upper()
        self.retries = retries
        self._call(self.exchange.load_markets)
        self._check_clock()

    def _check_clock(self) -> None:
        """Warn if the local clock differs from the exchange (candle-closure logic depends on it)."""
        try:
            skew = abs(int(self._call(self.exchange.fetch_time)) - int(time.time() * 1000))
            if skew > 5000:
                log.warning("Local clock differs from exchange by %.1fs - sync NTP, candle timing may be off", skew / 1000)
        except Exception:
            pass

    # -------------------------------------------------------------- plumbing
    def _call(self, fn: Callable, *a, **kw):
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                return fn(*a, **kw)
            except (self.ccxt.NetworkError, self.ccxt.RateLimitExceeded,
                    self.ccxt.ExchangeNotAvailable, self.ccxt.RequestTimeout) as exc:
                if attempt == self.retries:
                    raise
                log.warning("exchange call failed (%s); retry in %.0fs", type(exc).__name__, delay)
                time.sleep(delay)
                delay *= 2

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def timeframe_ms(self, timeframe: str) -> int:
        return int(self.exchange.parse_timeframe(timeframe) * 1000)

    # --------------------------------------------------------------- tickers
    def _spot_symbol(self, symbol: str) -> bool:
        m = self.exchange.markets.get(symbol)
        return bool(m and m.get("spot") and m.get("active", True) and m.get("quote") == self.quote
                    and _is_tradeable_base(m.get("base", "")))

    def tickers(self) -> Dict[str, Dict[str, float]]:
        raw = self._call(self.exchange.fetch_tickers)
        out = {}
        for sym, t in raw.items():
            if not self._spot_symbol(sym):
                continue
            last = t.get("last") or t.get("close")
            if not last:
                continue
            qv = t.get("quoteVolume")
            if qv is None and t.get("baseVolume"):
                qv = float(t["baseVolume"]) * float(last)
            out[sym] = {"last": float(last), "quoteVolume": float(qv or 0.0)}
        return out

    def liquid_symbols(self, min_quote_volume: float, limit: int,
                       watchlist: Sequence[str] = (), tickers: Optional[dict] = None) -> List[str]:
        tk = tickers if tickers is not None else self.tickers()
        if watchlist:
            return [s for s in watchlist if s in tk][:limit]
        ranked = sorted(((s, v["quoteVolume"]) for s, v in tk.items()
                         if v["quoteVolume"] >= min_quote_volume), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:limit]]

    # --------------------------------------------------------------- candles
    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        raw = self._call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit + 1)
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        tf_ms = self.timeframe_ms(timeframe)
        now_ms = int(self.now().timestamp() * 1000)
        df = df[df["timestamp"] + tf_ms <= now_ms]            # closed candles only
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna().drop_duplicates("timestamp").reset_index(drop=True)

    # ------------------------------------------------------ order constraints
    def round_qty(self, symbol: str, qty: float) -> float:
        try:
            return float(self.exchange.amount_to_precision(symbol, qty))
        except Exception:
            return qty

    def min_notional(self, symbol: str) -> float:
        try:
            m = self.exchange.markets[symbol]["limits"]["cost"]["min"]
            return float(m or 0.0)
        except Exception:
            return 0.0

    def fetch_history(self, symbol: str, timeframe: str, since_ms: int, until_ms: int,
                      max_candles: int = 200_000) -> pd.DataFrame:
        """Paginated historical download for backtests (closed candles in [since, until))."""
        step = self.timeframe_ms(timeframe)
        cursor, rows = since_ms, []
        while cursor < until_ms and len(rows) < max_candles:
            batch = self._call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe,
                               since=cursor, limit=1000)
            if not batch:
                break
            batch = [c for c in batch if cursor <= c[0] < until_ms]
            if rows:
                batch = [c for c in batch if c[0] > rows[-1][0]]
            if not batch:
                break
            rows.extend(batch)
            nxt = batch[-1][0] + step
            if nxt <= cursor:
                break
            cursor = nxt
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
