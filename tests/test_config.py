import pytest

from bot.config import Config


def test_live_mode_is_rejected():
    with pytest.raises(ValueError, match="paper"):
        Config(trading_mode="live").validate()


def test_env_parsing_and_aliases():
    c = Config.from_env({"INITIAL_CASH_USDT": "250", "MAX_OPEN_POSITIONS": "2", "LLM_ENABLED": "true",
                         "WATCHLIST": "btc/usdt, eth/usdt", "SCAN_INTERVAL_SECONDS": "45",
                         "TRADING_MODE": "PAPER"}, load_dotenv_file=False)
    assert c.initial_cash == 250 and c.max_open_positions == 2 and c.llm_enabled is True
    assert c.watchlist == ("BTC/USDT", "ETH/USDT") and c.scan_interval == 45 and c.trading_mode == "paper"
    c.validate()


@pytest.mark.parametrize("kw", [dict(risk_per_trade=0.5), dict(max_daily_loss=0.9), dict(rsi_min=70, rsi_max=60),
                                dict(llm_fail_mode="maybe"), dict(candle_limit=50), dict(initial_cash=0),
                                dict(min_stop_pct=0.1, max_stop_pct=0.05)])
def test_bad_values_rejected(kw):
    with pytest.raises(ValueError):
        Config(**kw).validate()


def test_env_example_matches_defaults():
    """Every key in .env.example must parse and validate."""
    import re
    env = {}
    for line in open(".env.example"):
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    Config.from_env(env, load_dotenv_file=False).validate()
