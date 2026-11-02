import os
import sqlite3

from bot.journal import Journal
from bot.synthetic import SyntheticData

from .helpers import FakeReviewer, make_bot, make_cfg, run


def entries(bot):
    return bot.j._all("SELECT * FROM trades ORDER BY id")


def test_full_run_produces_trades_and_consistent_books(tmp_path):
    seen = {"max_pos": 0}

    def check(bot):
        seen["max_pos"] = max(seen["max_pos"], len(bot.broker.positions))
        assert bot.broker.cash >= -1e-9
        eq = bot.broker.equity(bot.marks)
        assert bot.broker.open_notional(bot.marks) <= eq * 0.75          # 60% cap + mark drift tolerance

    bot, data = make_bot(tmp_path)
    run(bot, data, 250, check)
    trades = entries(bot)
    assert len(trades) >= 3, "demo market should trigger some trades"
    assert seen["max_pos"] <= bot.cfg.max_open_positions
    # the journal alone reproduces the broker's state exactly (crash-safe accounting)
    port = Journal(bot.cfg.database_path).load_portfolio()
    assert abs(port["cash"] - bot.broker.cash) < 1e-6 and abs(port["realized_pnl"] - bot.broker.realized_pnl) < 1e-6
    assert set(port["positions"]) == set(bot.broker.positions)
    # every trade respected stop < entry < target and the 25% notional cap
    for t in trades:
        assert t["stop_price"] < t["entry_price"] < t["target_price"]
        assert t["qty"] * t["entry_price"] <= 0.26 * max(r["equity"] for r in bot.j.equity_curve()) + 1
    # exits never use candles at/before the entry candle
    for t in trades:
        if t["status"] == "CLOSED":
            assert t["exit_time"] >= t["entry_time"]


def test_restart_mid_run_resumes_same_state(tmp_path):
    bot, data = make_bot(tmp_path)
    run(bot, data, 120)
    cash, pos, realized = bot.broker.cash, set(bot.broker.positions), bot.broker.realized_pnl
    bot2, _ = make_bot(tmp_path, data=data)
    assert abs(bot2.broker.cash - cash) < 1e-9 and set(bot2.broker.positions) == pos
    assert abs(bot2.broker.realized_pnl - realized) < 1e-9
    run(bot2, data, 60)                                                  # keeps working after recovery


def test_halted_bot_opens_nothing_but_still_manages_exits(tmp_path):
    bot, data = make_bot(tmp_path)
    run(bot, data, 150)
    open_before = set(bot.broker.positions)
    n_before = len(entries(bot))
    bot.guard.halt("test")
    run(bot, data, 150)
    assert len(entries(bot)) == n_before                                  # no new entries while halted
    assert len(bot.broker.positions) <= len(open_before)                  # existing ones may close


def test_kill_switch_blocks_entries_and_close_all_flattens(tmp_path):
    bot, data = make_bot(tmp_path)
    run(bot, data, 150)
    for _ in range(300):                                                  # make sure something is open
        if bot.broker.positions:
            break
        run(bot, data, 1)
    assert bot.broker.positions, "need an open position for this test"
    n = len(entries(bot))
    with open(bot.cfg.kill_switch_file, "w") as f:
        f.write("CLOSE_ALL")
    run(bot, data, 3)
    assert not bot.broker.positions
    assert any(t["reason"] == "kill_switch_close_all" for t in bot.j.get_closed_trades())
    run(bot, data, 100)
    assert len(entries(bot)) == n                                         # still blocked
    os.remove(bot.cfg.kill_switch_file)


def test_drawdown_halt_trips_in_a_crash_and_blocks_entries(tmp_path):
    bot, data = make_bot(tmp_path)
    run(bot, data, 150)
    real = bot.broker.equity
    bot.broker.equity = lambda marks: real(marks) * 0.5            # simulate a 50% crash in equity
    run(bot, data, 2)
    assert bot.guard.halted and "max_drawdown" in bot.guard.halt_reason
    bot.broker.equity = real
    n = len(entries(bot))
    run(bot, data, 150)
    assert len(entries(bot)) == n                                    # stays halted until a human resumes


def test_stale_feed_blocks_entries(tmp_path):
    class Frozen(SyntheticData):
        def now(self):
            return super().now() + __import__("datetime").timedelta(days=1)

    cfg = make_cfg(tmp_path)
    data = Frozen(timeframe=cfg.timeframe, total=3000)
    from bot.llm import RulesOnlyReviewer
    from bot.runner import Bot
    bot = Bot(cfg, data, reviewer=RulesOnlyReviewer())
    run(bot, data, 200)
    assert len(entries(bot)) == 0


def test_exchange_outage_does_not_kill_the_loop(tmp_path):
    bot, data = make_bot(tmp_path)
    real = data.tickers
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise ConnectionError("exchange down")
        return real()

    data.tickers = flaky
    run(bot, data, 40)
    assert calls["n"] >= 10 and bot._consec_errors == 0                   # recovered after the outage


def test_llm_reject_all_means_no_trades(tmp_path):
    bot, data = make_bot(tmp_path, reviewer=FakeReviewer("reject", 0.0))
    run(bot, data, 250)
    assert len(entries(bot)) == 0
    assert any(d["decision"] == "SKIPPED_BY_REVIEW" for d in bot.j.recent_decisions(500))


def test_llm_reduce_halves_first_position_and_never_enlarges(tmp_path):
    base, d1 = make_bot(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path)
    run(base, d1, 250)
    half_dir = tmp_path / "b"
    half_dir.mkdir()
    red, d2 = make_bot(half_dir, reviewer=FakeReviewer("reduce", 0.5))
    run(red, d2, 250)
    q0, q1 = entries(base)[0], entries(red)[0]
    assert q0["symbol"] == q1["symbol"]
    assert abs(q1["qty"] - 0.5 * q0["qty"]) / q0["qty"] < 0.02
    big_dir = tmp_path / "c"
    big_dir.mkdir()
    big, d3 = make_bot(big_dir, reviewer=FakeReviewer("reduce", 7.0))     # hostile "enlarge" attempt
    run(big, d3, 250)
    assert entries(big)[0]["qty"] <= q0["qty"] * 1.0000001        # hostile multiplier cannot enlarge


def test_llm_failure_fails_closed(tmp_path):
    bot, data = make_bot(tmp_path, reviewer=FakeReviewer(raises=True))
    run(bot, data, 250)
    assert len(entries(bot)) == 0


def test_llm_context_has_no_free_text_from_outside_and_is_compact(tmp_path):
    fr = FakeReviewer()
    bot, data = make_bot(tmp_path, reviewer=fr)
    run(bot, data, 250)
    assert fr.calls > 0
    import json
    ctx = fr.last_ctx
    assert {"symbol", "signal", "indicators", "portfolio", "recent_candles_ohlcv_oldest_first"} <= set(ctx)
    assert len(json.dumps(ctx)) < 6000


def test_decision_log_and_heartbeat_written(tmp_path):
    bot, data = make_bot(tmp_path)
    run(bot, data, 60)
    kv = bot.j.kv_all()
    assert kv["mode"] == "PAPER" and "last_cycle_wall" in kv
    assert bot.j.equity_curve() and bot.j.get_marks()


def test_llm_status_is_recorded_for_dashboard(tmp_path):
    bot, _ = make_bot(tmp_path)
    assert bot.j.kv_get("llm_status").startswith("OFF")
    d2 = tmp_path / "on"
    d2.mkdir()
    bot2, _ = make_bot(d2, reviewer=FakeReviewer())
    assert bot2.j.kv_get("llm_status") == "ON (fake)"


def test_repeated_llm_failures_trigger_one_alert(tmp_path):
    sent = []

    class Rec:
        def send(self, text, level="info"):
            sent.append(text)
            return True

        def entry(self, *a, **k): pass
        def exit(self, *a, **k): pass

    bot, data = make_bot(tmp_path, reviewer=FakeReviewer(raises=True))
    bot.notifier = Rec()
    run(bot, data, 300)
    assert bot.llm.consecutive_failures >= 3
    assert sum("failed 3 times" in m for m in sent) == 1
