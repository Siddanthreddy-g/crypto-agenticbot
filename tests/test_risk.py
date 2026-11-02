from datetime import datetime, timedelta, timezone

from bot.config import Config
from bot.journal import Journal
from bot.risk import RiskGuard, RiskManager

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def size(cfg=None, **kw):
    base = dict(equity=100.0, cash=100.0, entry_price=10.0, stop_pct=0.01, open_positions=0, open_notional=0.0)
    base.update(kw)
    return RiskManager(cfg or Config()).size_position(**base)


def test_position_is_capped_at_25pct_notional():
    d = size()
    assert d.approved and abs(d.qty * 10 - 25.0) < 1e-9


def test_risk_budget_binds_with_wide_stop():
    d = size(stop_pct=0.05)                    # 0.5% of 100 = 0.5 risk / 0.5 per unit = 1 unit
    assert abs(d.qty - 1.0) < 1e-9 or d.qty < 2.5


def test_llm_multiplier_can_only_shrink():
    base = size().qty
    assert abs(size(size_multiplier=2.0).qty - base) < 1e-9          # >1 clamped to 1
    assert abs(size(size_multiplier=0.5).qty - base * 0.5) < 1e-9
    assert not size(size_multiplier=0.0).approved
    assert size(size_multiplier=-1).qty == 0


def test_max_open_positions_and_exposure_cap():
    assert size(open_positions=3).reason == "max_open_positions"
    d = size(open_notional=55.0)                  # only 5 USDT of room under the 60% cap -> below min notional
    assert not d.approved
    d = size(open_notional=45.0)                  # 15 USDT room, 10 min notional -> allowed but shrunk
    assert d.approved and d.qty * 10 <= 15.0 + 1e-9


def test_min_notional_veto_and_rounding():
    assert "min_notional" in size(equity=20.0, cash=20.0).reason
    d = size(round_qty=lambda q: float(int(q * 10) / 10))
    assert abs(d.qty - 2.5) < 1e-9


def test_cash_is_never_exceeded_including_fees():
    d = size(equity=1000.0, cash=30.0)
    assert d.qty * 10 * (1 + 0.001 + 0.0005) <= 30.0


def test_drawdown_halt_persists_across_restart():
    j = Journal(":memory:")
    g = RiskGuard(Config(), j)
    assert g.update_equity(100.0) is None and g.update_equity(95.0) is None
    reason = g.update_equity(89.0)
    assert reason and g.halted
    g2 = RiskGuard(Config(), j)                                   # "restart"
    assert g2.halted and g2.blocked_reason(NOW, 120.0).startswith("HALTED")
    g2.resume(reset_peak_to=89.0)
    assert not g2.halted and g2.blocked_reason(NOW, 89.0) is None


def test_daily_loss_limit_blocks_until_next_day():
    j = Journal(":memory:")
    cfg = Config(kill_switch_file="")
    g = RiskGuard(cfg, j)
    g.roll_day(NOW, 100.0)
    assert g.blocked_reason(NOW, 99.0) is None
    assert g.blocked_reason(NOW, 97.9) == "daily_loss_limit_reached"
    rolled, prev = g.roll_day(NOW + timedelta(days=1), 97.9)
    assert rolled and prev == "2026-03-01" and g.blocked_reason(NOW + timedelta(days=1), 97.9) is None


def test_loss_streak_pause_and_symbol_cooldown():
    j = Journal(":memory:")
    cfg = Config(kill_switch_file="")
    g = RiskGuard(cfg, j)
    for _ in range(cfg.max_consecutive_losses):
        g.record_close("A/USDT", -1.0, NOW, 300_000)
    assert g.blocked_reason(NOW + timedelta(minutes=10), 100.0).startswith("loss_streak")
    assert g.blocked_reason(NOW + timedelta(minutes=cfg.loss_pause_minutes + 1), 100.0, "B/USDT") is None
    assert g.blocked_reason(NOW + timedelta(minutes=cfg.loss_pause_minutes + 1), 100.0, "A/USDT") is None   # 6*5m=30m passed
    later = NOW + timedelta(minutes=cfg.loss_pause_minutes + 1)          # pause is over
    g.record_close("C/USDT", 1.0, later, 300_000)
    assert g.blocked_reason(later + timedelta(minutes=5), 100.0, "C/USDT") == "symbol_cooldown"
    assert g.blocked_reason(later + timedelta(minutes=31), 100.0, "C/USDT") is None
    assert int(j.kv_float("consecutive_losses")) == 0                # a win resets the streak


def test_kill_switch_file(tmp_path):
    ks = tmp_path / "KS"
    g = RiskGuard(Config(kill_switch_file=str(ks)), Journal(":memory:"))
    assert g.blocked_reason(NOW, 100.0) is None
    ks.write_text("HALT")
    assert g.blocked_reason(NOW, 100.0) == "kill_switch_file_present"
