import numpy as np
import pandas as pd

from bot.indicators import enrich, rsi_wilder, session_vwap
from bot.strategy import StrategyParams, evaluate, signal_for_row, signal_from_values
from bot.synthetic import generate_history

P = StrategyParams()


def good(**over):
    v = dict(symbol="X/USDT", candle_ts=0, close=100.0, ema_fast=100.5, ema_slow=100.0, ema_trend=99.0,
             rsi=58.0, vwap=99.5, volume=300.0, vol_avg=100.0, atr=0.4)
    v.update(over)
    return signal_from_values(v.pop("symbol"), v.pop("candle_ts"), v.pop("close"), v.pop("ema_fast"),
                              v.pop("ema_slow"), v.pop("ema_trend"), v.pop("rsi"), v.pop("vwap"),
                              v.pop("volume"), v.pop("vol_avg"), v.pop("atr"), P)


def test_all_filters_pass_gives_buy_with_atr_levels():
    s = good()
    assert s.should_buy and s.stop_pct >= P.min_stop_pct and abs(s.target_pct / s.stop_pct - P.reward_risk) < 1e-9


def test_each_filter_blocks():
    for over in (dict(ema_fast=99.0), dict(close=98.0, vwap=99.5, ema_trend=97.0), dict(rsi=45.0),
                 dict(rsi=75.0), dict(vwap=101.0), dict(volume=100.0), dict(ema_trend=101.0)):
        assert not good(**over).should_buy, over


def test_nan_indicators_never_buy():
    assert not good(rsi=float("nan")).should_buy


def test_cost_gate_rejects_when_fees_dominate():
    expensive = StrategyParams(fee_rate=0.01, slippage_rate=0.005)
    s = signal_from_values("X", 0, 100.0, 100.5, 100.0, 99.0, 58.0, 99.5, 300.0, 100.0, 0.4, expensive)
    assert not s.should_buy and "cost" in s.reason or "net_rr" in s.reason


def test_stop_target_geometry_net_of_costs_is_positive_expectancy_shape():
    s = good()
    cost = P.round_trip_cost()
    assert (s.target_pct - cost) / (s.stop_pct + cost) >= P.min_net_rr


def test_no_look_ahead():
    """Signal on candle i must not change when future candles are appended."""
    df = generate_history(100.0, 800, "5m", seed=5)
    full = enrich(df)
    for i in (300, 450, 600):
        part = enrich(df.iloc[: i + 1])
        a, b = signal_for_row("X", part, i, P), signal_for_row("X", full, i, P)
        assert (a.should_buy, round(a.price, 9), a.reason) == (b.should_buy, round(b.price, 9), b.reason)
        for k in ("ema_fast", "rsi", "vwap", "atr"):
            assert abs(a.indicators[k] - b.indicators[k]) < 1e-9


def test_rsi_bounds_and_all_up_is_100():
    s = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi_wilder(s).iloc[-1] == 100.0
    r = rsi_wilder(pd.Series(np.random.default_rng(1).normal(100, 1, 300)))
    assert r.dropna().between(0, 100).all()


def test_vwap_resets_at_utc_midnight():
    ts = pd.date_range("2026-03-01 23:00", periods=24, freq="5min", tz="UTC")   # crosses midnight
    df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": np.r_[np.full(12, 10.0), np.full(12, 20.0)], "volume": 1.0})
    df["high"], df["low"] = df["close"], df["close"]
    v = session_vwap(df)
    assert v.iloc[11] == 10.0 and v.iloc[12] == 20.0      # new session starts fresh


def test_too_few_candles_returns_none():
    assert evaluate("X", generate_history(100.0, 50), P) is None
