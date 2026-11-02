import json

import pytest

from bot.agents import LLMReviewAgent
from bot.config import Config
from bot.journal import Journal
from bot.llm import (REVIEW_SCHEMA, AnthropicReviewer, LLMError, RulesOnlyReviewer, make_reviewer,
                     normalize)
from bot.strategy import Signal

from .helpers import FakeAnthropic, FakeResp, FakeReviewer

SIG = Signal("X/USDT", 0, 100.0, True, "r", score=1.0, stop_pct=0.01, target_pct=0.02)


def ok(decision="approve", mult=1.0, conf=0.8):
    return json.dumps({"decision": decision, "size_multiplier": mult, "confidence": conf,
                       "rationale": "because", "key_risks": ["a", "b", "c", "d"]})


def test_normalize_forces_safe_multipliers():
    assert normalize({"decision": "approve", "size_multiplier": 5, "confidence": 2, "rationale": "r"}).size_multiplier == 1.0
    assert normalize({"decision": "approve", "size_multiplier": 5, "confidence": 2, "rationale": "r"}).confidence == 1.0
    assert normalize({"decision": "reject", "size_multiplier": 1, "confidence": 1, "rationale": ""}).size_multiplier == 0.0
    assert normalize({"decision": "reduce", "size_multiplier": 3, "confidence": 1, "rationale": ""}).size_multiplier == 0.9  # never grows
    assert normalize({"decision": "reduce", "size_multiplier": -2, "confidence": 1, "rationale": ""}).size_multiplier == 0.1
    assert len(normalize(json.loads(ok())).key_risks) == 3


@pytest.mark.parametrize("raw", [{}, {"decision": "buy_more"}, {"decision": "approve", "size_multiplier": "x"},
                                 {"decision": "approve", "size_multiplier": 1}])
def test_normalize_rejects_malformed(raw):
    with pytest.raises(LLMError):
        normalize(raw)


def test_anthropic_request_shape():
    fake = FakeAnthropic(FakeResp(ok()))
    r = AnthropicReviewer("claude-opus-5-5", client=fake).review({"symbol": "X"})
    kw = fake.kwargs
    assert r.decision == "approve" and kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["schema"] == REVIEW_SCHEMA and kw["output_config"]["effort"] == "low"
    # parameters removed on current models must not be sent
    for banned in ("temperature", "top_p", "top_k", "thinking", "tool_choice"):
        assert banned not in kw
    assert kw["messages"][-1]["role"] == "user"            # no assistant prefill


@pytest.mark.parametrize("resp", [FakeResp(stop_reason="refusal", text="x"), FakeResp(stop_reason="max_tokens", text="{"),
                                  FakeResp(text="not json"), FakeResp(blocks=[]), FakeResp(text='{"decision":"approve"}')])
def test_bad_responses_raise_llmerror(resp):
    with pytest.raises(LLMError):
        AnthropicReviewer(client=FakeAnthropic(resp)).review({})


def test_network_error_becomes_llmerror():
    with pytest.raises(LLMError):
        AnthropicReviewer(client=FakeAnthropic(exc=TimeoutError("t"))).review({})


def agent(reviewer, **cfg):
    return LLMReviewAgent(reviewer, Config(**cfg), Journal(":memory:"))


def test_fail_closed_by_default_and_fail_open_when_configured():
    assert not agent(FakeReviewer(raises=True)).review(SIG, {}, "t").approved
    v = agent(FakeReviewer(raises=True), llm_fail_mode="approve").review(SIG, {}, "t")
    assert v.approved and v.source == "fallback"


def test_low_confidence_is_rejected():
    assert not agent(FakeReviewer(conf=0.2), llm_min_confidence=0.5).review(SIG, {}, "t").approved
    assert agent(FakeReviewer(conf=0.9)).review(SIG, {}, "t").approved


def test_reduce_passes_multiplier_reject_blocks():
    v = agent(FakeReviewer("reduce", 0.4)).review(SIG, {}, "t")
    assert v.approved and v.size_multiplier == 0.4
    assert not agent(FakeReviewer("reject", 0.0)).review(SIG, {}, "t").approved


def test_rate_limit_fails_closed_without_calling_llm():
    fr = FakeReviewer()
    a = agent(fr, llm_max_calls_per_hour=2)
    results = [a.review(SIG, {}, "t").approved for _ in range(4)]
    assert results == [True, True, False, False] and fr.calls == 2


def test_every_review_is_logged():
    a = agent(FakeReviewer())
    a.review(SIG, {}, "t")
    assert a.j.recent_decisions(agent="llm_review")[0]["symbol"] == "X/USDT"


def test_make_reviewer_without_key_is_rules_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_reviewer(Config(llm_enabled=True)), RulesOnlyReviewer)
    assert isinstance(make_reviewer(Config(llm_enabled=False)), RulesOnlyReviewer)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(make_reviewer(Config(llm_enabled=True)), AnthropicReviewer)


def test_off_reason_and_status_text(monkeypatch):
    from bot.llm import llm_off_reason, status_text
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "LLM_ENABLED" in llm_off_reason(Config(llm_enabled=False))
    assert "ANTHROPIC_API_KEY" in llm_off_reason(Config(llm_enabled=True))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert llm_off_reason(Config(llm_enabled=True)) is None
    assert status_text(Config(), RulesOnlyReviewer()).startswith("OFF")
    assert status_text(Config(), FakeReviewer()) == "ON (fake)"
