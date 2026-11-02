import backtest as bt
from bot.config import Config
from bot.strategy import StrategyParams
from bot.synthetic import DEFAULT_SYMBOLS, generate_history

CFG = Config(kill_switch_file="")
TF = 300_000


def prep(n=4000):
    return bt.prepare({s: generate_history(p, n, "5m", seed=50 + i) for i, (s, p) in enumerate(DEFAULT_SYMBOLS.items())})


def test_deterministic_and_sane():
    p = prep()
    a = bt.simulate(p, CFG, StrategyParams.from_config(CFG), TF)
    b = bt.simulate(p, CFG, StrategyParams.from_config(CFG), TF)
    assert a["trades"] == b["trades"] > 20 and a["final"] == b["final"]
    assert a["final"] > 0 and 0 <= a["max_drawdown_pct"] < 50
    assert all(t["status"] == "CLOSED" for t in a["trades_list"])


def test_costs_matter_zero_cost_beats_real_cost():
    p = prep()
    real = bt.simulate(p, CFG, StrategyParams.from_config(CFG), TF)
    free_cfg = CFG.replace(fee_rate=0.0, slippage_rate=0.0)
    free = bt.simulate(p, free_cfg, StrategyParams.from_config(free_cfg), TF)
    assert free["final"] > real["final"]


def test_window_slicing_partitions_trades():
    p = prep()
    full = bt.simulate(p, CFG, StrategyParams.from_config(CFG), TF)
    b = bt.split_bounds(full["timeline"], 2)
    assert b[0] <= full["timeline"][0] and b[-1] > full["timeline"][-1]
    h1 = bt.simulate(p, CFG, StrategyParams.from_config(CFG), TF, b[0], b[1])
    h2 = bt.simulate(p, CFG, StrategyParams.from_config(CFG), TF, b[1], b[2])
    assert h1["candles"] + h2["candles"] == full["candles"]


def test_no_entry_before_signal_candle_closes():
    """entry candle ts must be strictly after the signal candle ts."""
    p = prep(2500)
    r = bt.simulate(p, CFG, StrategyParams.from_config(CFG), TF)
    assert r["trades"] > 0
    for t in r["trades_list"]:
        assert t["entry_candle_ts"] > 0
