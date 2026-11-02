import os

from bot.broker import PaperBroker
from bot.journal import Journal

ISO = "2026-03-01T00:00:00+00:00"
TF = 300_000


def mk(cash=1000.0, fee=0.001, slip=0.0, path=":memory:"):
    j = Journal(path)
    return PaperBroker(cash, fee, slip, j), j


def enter(b, price=100.0, qty=1.0, ts=1_000_000):
    return b.enter_long("X/USDT", qty, price, 0.01, 0.02, "t", ISO, ts)


def candle(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c}


def test_round_trip_accounting_matches_ledger():
    b, j = mk()
    p = enter(b)
    assert abs(b.cash - (1000 - 100 - 0.1)) < 1e-9
    r = b.evaluate_candle("X/USDT", candle(1_000_000 + TF, 100, 103, 100, 102), TF, 48, ISO)
    assert r["reason"] == "take_profit"
    expected = (1 * 102 - 1 * 102 * 0.001) - (100 + 0.1)
    assert abs(r["pnl"] - expected) < 1e-9
    port = j.load_portfolio()
    assert abs(port["cash"] - b.cash) < 1e-9 and port["closed_trades"] == 1 and not port["positions"]


def test_signal_candle_and_older_are_ignored_regression():
    """BUG FIX: the old paper trader re-checked the pre-entry candle and could stop out instantly."""
    b, _ = mk()
    enter(b, ts=1_000_000)
    pre = candle(1_000_000, 100, 100, 50, 100)                       # signal candle, huge wick
    older = candle(700_000, 100, 100, 10, 100)
    assert b.evaluate_candles("X/USDT", [older, pre], TF, 48, ISO) is None
    assert "X/USDT" in b.positions


def test_missed_candles_are_replayed_in_order():
    """BUG FIX: after downtime the stop must still be honoured on the candle where it happened."""
    b, _ = mk()
    enter(b, ts=1_000_000)
    cs = [candle(1_000_000 + TF, 100, 100.5, 99.5, 100), candle(1_000_000 + 2 * TF, 100, 100, 98, 98.5),
          candle(1_000_000 + 3 * TF, 98.5, 105, 98.5, 104)]       # stop first, target later
    r = b.evaluate_candles("X/USDT", cs, TF, 48, ISO)
    assert r["reason"] == "stop_loss" and not b.positions


def test_stop_first_when_both_touched_and_gap_fills():
    b, _ = mk()
    enter(b)
    r = b.evaluate_candle("X/USDT", candle(1_000_000 + TF, 100, 105, 97, 101), TF, 48, ISO)
    assert r["reason"] == "stop_loss" and abs(r["exit_price"] - 99.0) < 1e-9
    b2, _ = mk()
    enter(b2)
    r = b2.evaluate_candle("X/USDT", candle(1_000_000 + TF, 94, 96, 93, 95), TF, 48, ISO)     # gap down
    assert r["exit_price"] == 94.0
    b3, _ = mk()
    enter(b3)
    r = b3.evaluate_candle("X/USDT", candle(1_000_000 + TF, 106, 108, 105, 107), TF, 48, ISO)  # gap up
    assert r["reason"] == "take_profit" and r["exit_price"] == 106.0


def test_time_stop():
    b, _ = mk()
    enter(b, ts=0)
    flat = [candle(TF * k, 100, 100.2, 99.8, 100.1) for k in range(1, 4)]
    r = b.evaluate_candles("X/USDT", flat, TF, 3, ISO)
    assert r["reason"] == "time_stop"


def test_live_tick_stop_and_target():
    b, _ = mk()
    enter(b)
    assert b.evaluate_price("X/USDT", 99.5, ISO) is None
    assert b.evaluate_price("X/USDT", 98.0, ISO)["reason"] == "stop_loss"
    b2, _ = mk()
    enter(b2)
    assert b2.evaluate_price("X/USDT", 102.5, ISO)["reason"] == "take_profit"


def test_slippage_and_fees_make_losers_lose_more():
    b, _ = mk(slip=0.0005)
    p = enter(b)
    assert p.entry_price > 100.0
    r = b.evaluate_candle("X/USDT", candle(1_000_000 + TF, 100, 100, 98, 99), TF, 48, ISO)
    assert r["pnl"] < -1.0                                  # worse than the raw -1% on 100 notional


def test_cannot_overspend_cash_or_double_enter():
    b, _ = mk(cash=50.0)
    p = b.enter_long("X/USDT", 10.0, 100.0, 0.01, 0.02, "t", ISO, 0)
    assert p and p.qty * p.entry_price <= 50.0 and b.cash >= -1e-9
    assert b.enter_long("X/USDT", 1.0, 100.0, 0.01, 0.02, "t", ISO, 0) is None


def test_restart_recovers_positions_cash_and_entry_candle(tmp_path):
    path = str(tmp_path / "r.db")
    b1, _ = mk(path=path)
    enter(b1, ts=123_000)
    b2 = PaperBroker(1000.0, 0.001, 0.0, Journal(path))
    assert abs(b2.cash - b1.cash) < 1e-9 and b2.positions["X/USDT"].entry_candle_ts == 123_000
    r = b2.evaluate_candle("X/USDT", candle(123_000 + TF, 100, 100, 98, 98.5), TF, 48, ISO)
    assert r["reason"] == "stop_loss"
    b3 = PaperBroker(1000.0, 0.001, 0.0, Journal(path))
    assert b3.closed_trades == 1 and b3.losses == 1 and not b3.positions
    assert abs(b3.cash - b2.cash) < 1e-9


def test_close_at_market():
    b, _ = mk()
    enter(b)
    r = b.close_at_market("X/USDT", 101.0, "manual", ISO)
    assert r["reason"] == "manual" and not b.positions
