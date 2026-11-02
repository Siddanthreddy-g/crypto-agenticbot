import json
import math
import sqlite3

import pytest

from bot.config import Config
from bot.journal import Journal
from bot.llm import AnthropicReviewer, LLMError
from bot.scout import MAX_CANDIDATES, SCOUT_SCHEMA, build_snapshot, normalize_scout
from bot.strategy import StrategyParams, risk_geometry
from bot.synthetic import SyntheticData

from .helpers import FakeAnthropic, FakeResp, FakeScout, make_bot, run

UNI = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def cand(sym="ETH/USDT", conf=0.8, **kw):
    d = {"symbol": sym, "setup_type": "breakout", "confidence": conf, "thesis": "t", "key_risks": ["r"]}
    d.update(kw)
    return d


# ------------------------------------------------------------- validation
def test_normalize_drops_unknown_duplicate_and_malformed():
    raw = {"market_view": "v", "candidates": [cand("eth/usdt"), cand("ETH/USDT"), cand("DOGE/USDT"),
                                              {"symbol": "SOL/USDT"}, cand("BTC/USDT", conf=float("nan"))]}
    r = normalize_scout(raw, UNI)
    assert [c.symbol for c in r.candidates] == ["ETH/USDT"]              # case-insensitive map to the real symbol
    assert any("duplicate" in d for d in r.dropped) and any("unknown_symbol" in d for d in r.dropped)
    assert len(r.dropped) == 4


def test_normalize_clamps_sorts_and_limits():
    raw = {"market_view": "x" * 1000, "candidates": [cand("BTC/USDT", 0.7), cand("ETH/USDT", 9.0, setup_type="nonsense",
                                                                                      thesis="y" * 2000, key_risks=list("abcdefg")),
                                                      cand("SOL/USDT", -3)]}
    r = normalize_scout(raw, UNI + ["XRP/USDT"])
    assert [c.symbol for c in r.candidates] == ["ETH/USDT", "BTC/USDT", "SOL/USDT"]
    top = r.candidates[0]
    assert top.confidence == 1.0 and top.setup_type == "other" and len(top.thesis) <= 400 and len(top.key_risks) == 3
    assert r.candidates[-1].confidence == 0.0 and len(r.market_view) <= 400
    many = {"market_view": "", "candidates": [cand(s, 0.9 - i / 100) for i, s in enumerate(UNI + ["A/USDT", "B/USDT"])]}
    assert len(normalize_scout(many, UNI + ["A/USDT", "B/USDT"]).candidates) == MAX_CANDIDATES


@pytest.mark.parametrize("raw", [None, [], {}, {"candidates": "x"}, {"market_view": "v"}])
def test_normalize_rejects_bad_envelope(raw):
    with pytest.raises(ValueError):
        normalize_scout(raw, UNI)


def test_empty_candidates_is_valid_and_common():
    assert normalize_scout({"market_view": "flat", "candidates": []}, UNI).candidates == []


# --------------------------------------------------------------- snapshot
def test_snapshot_is_compact_numeric_and_flags_held():
    d = SyntheticData(timeframe="5m", total=1500)
    cfg = Config()
    frames = {s: d.candles(s, "5m", 300) for s in d.symbols}
    snap = build_snapshot(frames, d.tickers(), cfg, StrategyParams.from_config(cfg), ["ETH/USDT"], {"trades": 3}, -0.4, "t")
    assert len(snap["universe"]) == 5 and snap["min_confidence"] == cfg.llm_scout_min_confidence
    held = {r["symbol"]: r["already_held"] for r in snap["universe"]}
    assert held["ETH/USDT"] and not held["BTC/USDT"]
    assert all(r["rules_signal"] for r in snap["universe"])               # the LLM sees why the rules said no
    text = json.dumps(snap)
    assert len(text) < 9000 and "NaN" not in text
    for r in snap["universe"]:
        for k, v in r.items():
            assert v is None or isinstance(v, (int, float, str, bool)) and not (isinstance(v, float) and math.isnan(v))


# ------------------------------------------------------------ client shape
def test_anthropic_scout_request_shape_and_model_override():
    payload = json.dumps({"market_view": "v", "candidates": [cand("ETH/USDT")]})
    fake = FakeAnthropic(FakeResp(payload))
    rev = AnthropicReviewer("claude-opus-5-5", client=fake, scout_model="claude-sonnet-5-5")
    res = rev.scout({"universe": [{"symbol": "ETH/USDT"}], "min_confidence": 0.65})
    kw = fake.kwargs
    assert kw["model"] == "claude-sonnet-5-5" and res.candidates[0].symbol == "ETH/USDT"
    assert kw["output_config"]["format"]["schema"] == SCOUT_SCHEMA
    for banned in ("temperature", "top_p", "top_k", "thinking", "tool_choice"):
        assert banned not in kw
    assert kw["messages"][-1]["role"] == "user"


def test_anthropic_scout_filters_hallucinated_symbols_and_wraps_errors():
    payload = json.dumps({"market_view": "v", "candidates": [cand("MADEUP/USDT"), cand("ETH/USDT")]})
    res = AnthropicReviewer(client=FakeAnthropic(FakeResp(payload))).scout({"universe": [{"symbol": "ETH/USDT"}]})
    assert [c.symbol for c in res.candidates] == ["ETH/USDT"] and res.dropped
    for bad in (FakeResp("not json"), FakeResp(stop_reason="refusal", text="x"), FakeResp('{"market_view":"v"}')):
        with pytest.raises(LLMError):
            AnthropicReviewer(client=FakeAnthropic(bad)).scout({"universe": []})
    with pytest.raises(LLMError):
        AnthropicReviewer(client=FakeAnthropic(exc=TimeoutError("t"))).scout({"universe": []})


# ----------------------------------------------------------- shared geometry
def test_risk_geometry_matches_cost_gate():
    p = StrategyParams()
    s, t, rr, bad = risk_geometry(100.0, 0.1, p)                           # tiny ATR -> floor stop
    assert bad is None and s == p.min_stop_pct and abs(t / s - p.reward_risk) < 1e-9 and rr >= p.min_net_rr
    assert risk_geometry(100.0, 5.0, p)[0] == p.max_stop_pct              # huge ATR -> capped
    assert risk_geometry(100.0, 0.1, StrategyParams(fee_rate=0.01))[3] is not None


# ----------------------------------------------------------- journal changes
def test_journal_migrates_old_database_and_counts(tmp_path):
    path = str(tmp_path / "old.db")
    con = sqlite3.connect(path)
    con.executescript("""CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
        entry_time TEXT NOT NULL, exit_time TEXT, qty REAL NOT NULL, entry_price REAL NOT NULL, exit_price REAL,
        stop_price REAL NOT NULL, target_price REAL NOT NULL, entry_fee REAL NOT NULL, exit_fee REAL, pnl REAL,
        status TEXT NOT NULL, reason TEXT, entry_candle_ts INTEGER NOT NULL DEFAULT 0, signal_reason TEXT, meta TEXT);
        INSERT INTO trades (symbol, entry_time, qty, entry_price, stop_price, target_price, entry_fee, status)
        VALUES ('OLD/USDT','2026-01-01T00:00:00+00:00',1,1,0.9,1.1,0,'OPEN');""")
    con.commit()
    con.close()
    j = Journal(path)
    assert j.get_open_trades()[0]["source"] == "rules"
    j.add_trade({"symbol": "X/USDT", "entry_time": "2026-03-01T10:00:00+00:00", "qty": 1, "entry_price": 1,
                 "stop_price": 0.9, "target_price": 1.1, "entry_fee": 0, "source": "llm_scout"})
    assert j.count_trades(source="llm_scout") == 1 and j.count_trades(source="rules") == 1
    assert j.count_trades(source="llm_scout", since_iso="2026-03-02") == 0
    assert j.count_trades(status="OPEN") == 2


# ------------------------------------------------------------ end to end
def scout_bot(tmp_path, scout=None, **kw):
    scout = scout or FakeScout()
    bot, data = make_bot(tmp_path, reviewer=scout, llm_scout_enabled=True, **kw)
    return bot, data, scout


def scout_trades(bot):
    return [t for t in bot.j._all("SELECT * FROM trades ORDER BY id") if t["source"] == "llm_scout"]


def test_scout_trades_are_tagged_small_capped_and_code_sets_levels(tmp_path):
    peak = {"open": 0}

    def check(b):
        peak["open"] = max(peak["open"], b.j.count_trades(source="llm_scout", status="OPEN"))

    bot, data, scout = scout_bot(tmp_path)
    run(bot, data, 300, check)
    trades = scout_trades(bot)
    assert scout.scout_calls > 0 and len(trades) >= 2, "scout should be able to open paper trades"
    assert peak["open"] <= bot.cfg.llm_scout_max_concurrent
    per_day = {}
    for t in trades:
        per_day[t["entry_time"][:10]] = per_day.get(t["entry_time"][:10], 0) + 1
        notional = t["qty"] * t["entry_price"]
        assert notional <= 0.25 * bot.cfg.llm_scout_size_factor * 110 + 0.5        # half of the 25% cap
        stop_pct = 1 - t["stop_price"] / t["entry_price"]
        target_pct = t["target_price"] / t["entry_price"] - 1
        assert 0.007 <= stop_pct <= 0.031 and abs(target_pct / stop_pct - bot.cfg.reward_risk) < 0.01
        assert "LLM scout" in t["signal_reason"]
    assert max(per_day.values()) <= bot.cfg.llm_scout_max_trades_per_day
    assert any(d["agent"] == "llm_scout" and d["decision"] == "SCAN" for d in bot.j.recent_decisions(2000))


def test_scout_interval_is_respected(tmp_path):
    bot, data, scout = scout_bot(tmp_path, scout=FakeScout(conf=0.4))             # never trades -> asked every interval
    run(bot, data, 120)                                                          # 120 x 5min = 10h
    assert 1 <= scout.scout_calls <= 120 * 5 / 15 + 1


def test_low_confidence_nominations_never_trade(tmp_path):
    bot, data, scout = scout_bot(tmp_path, scout=FakeScout(conf=0.4))
    run(bot, data, 200)
    assert scout.scout_calls > 0 and scout_trades(bot) == []
    assert any(d["decision"] == "REJECTED" and "low_confidence" in d["detail"] for d in bot.j.recent_decisions(2000))


def test_hostile_nominations_are_rejected_by_the_runner_itself(tmp_path):
    """Even if a buggy/malicious scout returns a made-up symbol, the runner independently refuses it."""
    bot, data, scout = scout_bot(tmp_path, scout=FakeScout(symbols=["NOPE/USDT"]))
    run(bot, data, 120)
    assert scout.scout_calls > 0 and scout_trades(bot) == []
    assert any("no_fresh_data" in d["detail"] for d in bot.j.recent_decisions(2000) if d["agent"] == "llm_scout")


def test_scout_disabled_or_rules_only_makes_no_calls(tmp_path):
    scout = FakeScout()
    bot, data = make_bot(tmp_path, reviewer=scout)                                # llm_scout_enabled defaults False
    run(bot, data, 100)
    assert scout.scout_calls == 0
    d2 = tmp_path / "r"
    d2.mkdir()
    bot2, data2 = make_bot(d2, llm_scout_enabled=True)                            # rules-only reviewer
    run(bot2, data2, 100)
    assert scout_trades(bot2) == []


def test_halt_and_kill_switch_stop_scouting(tmp_path):
    bot, data, scout = scout_bot(tmp_path)
    bot.guard.halt("test")
    run(bot, data, 60)
    assert scout.scout_calls == 0
    d2 = tmp_path / "k"
    d2.mkdir()
    bot2, data2, scout2 = scout_bot(d2)
    open(bot2.cfg.kill_switch_file, "w").write("HALT")
    run(bot2, data2, 60)
    assert scout2.scout_calls == 0


def test_scout_failures_trade_nothing_alert_once_and_dont_hammer_api(tmp_path):
    sent = []

    class Rec:
        def send(self, text, level="info"):
            sent.append(text)
            return True

        def entry(self, *a, **k): pass
        def exit(self, *a, **k): pass

    bot, data, scout = scout_bot(tmp_path, scout=FakeScout(scout_raises=True))
    bot.notifier = Rec()
    run(bot, data, 200)
    assert scout_trades(bot) == [] and bot.scout_failures >= 3
    assert sum("scout failed 3 times" in m for m in sent) == 1
    assert scout.scout_calls <= 200 * 5 / 15 + 1                                   # interval applies even when failing


def test_restart_does_not_rescout_immediately(tmp_path):
    bot, data, scout = scout_bot(tmp_path, scout=FakeScout(conf=0.4))
    run(bot, data, 1)
    assert scout.scout_calls == 1
    bot2, _ = make_bot(tmp_path, reviewer=FakeScout(conf=0.4), data=data, llm_scout_enabled=True)
    run(bot2, data, 1)
    assert bot2.reviewer.scout_calls == 0


def test_scout_never_nominates_held_symbols_to_enter_twice(tmp_path):
    bot, data, scout = scout_bot(tmp_path, scout=FakeScout(symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
                                 llm_scout_max_concurrent=3, llm_scout_max_trades_per_day=20)
    seen = {}

    def check(b):
        for s in b.broker.positions:
            seen.setdefault(s, 0)

    run(bot, data, 250, check)
    open_rows = [t for t in bot.j._all("SELECT symbol, status FROM trades WHERE status='OPEN'")]
    assert len({t["symbol"] for t in open_rows}) == len(open_rows)                # one open position per symbol, always


def test_no_budget_means_no_llm_call(tmp_path):
    """With a daily cap of 0 the bot must not pay for scouting it could never act on."""
    bot, data, scout = scout_bot(tmp_path, llm_scout_max_trades_per_day=0)
    run(bot, data, 60)
    assert scout.scout_calls == 0
