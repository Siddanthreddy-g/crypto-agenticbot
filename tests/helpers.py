from types import SimpleNamespace

from bot.config import Config
from bot.llm import LLMError, ReviewResult
from bot.runner import Bot
from bot.synthetic import SyntheticData


def make_cfg(tmp_path, **kw) -> Config:
    base = dict(data_source="synthetic", database_path=str(tmp_path / "t.db"),
                kill_switch_file=str(tmp_path / "KS"), scan_interval=5)
    base.update(kw)
    return Config(**base).validate()


def make_bot(tmp_path, reviewer=None, data=None, **kw):
    cfg = make_cfg(tmp_path, **kw)
    data = data or SyntheticData(timeframe=cfg.timeframe, total=3000)
    from bot.llm import RulesOnlyReviewer
    bot = Bot(cfg, data, reviewer=reviewer or RulesOnlyReviewer())
    return bot, data


def run(bot, data, cycles, check=None):
    def cb():
        if check:
            check(bot)
        return data.advance()
    bot.run_forever(max_cycles=cycles, on_cycle=cb, sleep=False)


class FakeReviewer:
    """Deterministic stand-in for the LLM reviewer."""
    model = "fake"

    def __init__(self, decision="approve", mult=1.0, conf=0.9, raises=False):
        self.decision, self.mult, self.conf, self.raises = decision, mult, conf, raises
        self.calls = 0
        self.last_ctx = None

    def review(self, ctx):
        self.calls += 1
        self.last_ctx = ctx
        if self.raises:
            raise LLMError("boom")
        return ReviewResult(self.decision, self.mult, self.conf, "fake rationale", [], source="llm", model="fake")

    def narrate(self, stats):
        return "fake narrative"


class FakeResp:
    def __init__(self, text=None, stop_reason="end_turn", blocks=None):
        self.stop_reason = stop_reason
        self.content = blocks if blocks is not None else [SimpleNamespace(type="text", text=text)]


class FakeAnthropic:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.kwargs = resp, exc, None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return self.resp
