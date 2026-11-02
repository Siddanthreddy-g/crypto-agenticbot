import glob
import re

from bot.config import Config
from bot.notifier import Notifier


def test_no_live_trading_or_credential_code_in_package():
    """The distribution must stay paper-only: no order placement, no exchange credentials."""
    banned = re.compile(r"create_order|createOrder|create_market|apiKey|api_secret|\.withdraw|private_post")
    for f in glob.glob("bot/*.py") + ["run_bot.py", "backtest.py", "manage.py", "dashboard.py"]:
        assert not banned.search(open(f, encoding="utf-8").read()), f


def test_no_hardcoded_secrets_anywhere():
    tg = re.compile(r"\d{8,10}:AA[\w-]{30,}")
    sk = re.compile(r"sk-ant-[\w-]{20,}")
    for f in glob.glob("**/*.py", recursive=True) + glob.glob("*.md") + [".env.example"]:
        txt = open(f, encoding="utf-8").read()
        assert not tg.search(txt) and not sk.search(txt), f


def test_notifier_never_raises_and_noops_without_credentials():
    assert Notifier(Config()).send("x") is False

    class Boom:
        def post(self, *a, **k):
            raise ConnectionError("down")

    n = Notifier(Config(telegram_bot_token="123:abc", telegram_chat_id="1"), session=Boom())
    assert n.send("hello") is False


def test_notifier_posts_when_configured():
    sent = {}

    class Ok:
        def post(self, url, data=None, timeout=None):
            sent.update(url=url, data=data, timeout=timeout)
            return type("R", (), {"status_code": 200})()

    n = Notifier(Config(telegram_bot_token="123:abc", telegram_chat_id="42"), session=Ok())
    assert n.send("hello") and sent["data"]["chat_id"] == "42" and sent["timeout"]


def test_dashboard_renders_on_demo_db(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from .helpers import make_bot, run
    bot, data = make_bot(tmp_path)
    run(bot, data, 120)
    monkeypatch.setenv("DATABASE_PATH", bot.cfg.database_path)
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "dashboard.py"), default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Equity (USDT)" in labels and "Max drawdown" in labels


def test_dashboard_and_report_split_rules_vs_scout(tmp_path, monkeypatch):
    import os
    from streamlit.testing.v1 import AppTest
    from bot.reviewer import build_report, collect_stats
    from .helpers import FakeScout, make_bot, run
    bot, data = make_bot(tmp_path, reviewer=FakeScout(), llm_scout_enabled=True)
    run(bot, data, 250)
    st = collect_stats(bot.j)
    assert "llm_scout" in st["by_source"] or "rules" in st["by_source"]
    assert "llm scout" in open(build_report(bot.j, None, out_dir=str(tmp_path / "rep"))).read().lower()
    monkeypatch.setenv("DATABASE_PATH", bot.cfg.database_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "dashboard.py"), default_timeout=60).run()
    assert not at.exception
    assert "Rules vs LLM scout" in [s.value for s in at.subheader]
