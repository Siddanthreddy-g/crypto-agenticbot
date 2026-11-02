"""Causal technical indicators (value at row i only depends on rows <= i)."""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    out = 100 - 100 / (1 + rs)
    # no losses at all in the window -> RSI 100
    out = out.where(~((avg_loss == 0) & avg_gain.notna()), 100.0)
    return out


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets at 00:00 UTC."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = pd.to_numeric(df["volume"], errors="coerce")
    ts = pd.to_datetime(df["timestamp"], utc=True)
    session = ts.dt.floor("D")
    cum_vol = vol.groupby(session, sort=False).cumsum()
    cum_val = (typical * vol).groupby(session, sort=False).cumsum()
    return cum_val / cum_vol.replace(0.0, float("nan"))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns used by the strategy."""
    if df.empty:
        return df.copy()
    out = df.copy().reset_index(drop=True)
    out["ema_fast"] = ema(out["close"], 9)
    out["ema_slow"] = ema(out["close"], 21)
    out["ema_trend"] = ema(out["close"], 50)
    out["rsi"] = rsi_wilder(out["close"], 14)
    out["atr"] = atr_wilder(out["high"], out["low"], out["close"], 14)
    out["vwap"] = session_vwap(out)
    out["vol_avg20"] = out["volume"].rolling(20, min_periods=20).mean()
    return out
