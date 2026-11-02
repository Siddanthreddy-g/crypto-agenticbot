"""OpenAI-compatible provider (Ollama / Groq / Gemini / ...) tested against a local mock HTTP server that speaks
the same /chat/completions wire protocol."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bot.config import Config
from bot.llm import (AnthropicReviewer, LLMError, OpenAICompatReviewer, RulesOnlyReviewer, llm_off_reason,
                     make_reviewer, parse_json_object, status_text)

from .helpers import make_bot, run


class Mock:
    """Tiny OpenAI-style server. `script` is a list of (status, body_dict_or_text, headers) consumed in order;
    when exhausted, `default(request_json) -> (status, body)` is used."""

    def __init__(self, default=None):
        self.requests, self.script, self.default = [], [], default
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if outer.script:
                    status, payload, hdrs = outer.script.pop(0)
                else:
                    status, payload = outer.default(body)
                    hdrs = {}
                raw = payload if isinstance(payload, str) else json.dumps(payload)
                self.send_response(status)
                self.send_header("content-type", "application/json")
                for k, v in hdrs.items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(raw.encode())))
                self.end_headers()
                self.wfile.write(raw.encode())

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def chat(content, finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}]}


REVIEW_OK = {"decision": "approve", "size_multiplier": 1, "confidence": 0.8, "rationale": "ok", "key_risks": []}


@pytest.fixture
def mock():
    m = Mock(default=lambda b: (200, chat(json.dumps(REVIEW_OK))))
    yield m
    m.close()


def rev(mock, **kw):
    kw.setdefault("sleep", lambda s: None)
    return OpenAICompatReviewer("test-model", mock.url, kw.pop("api_key", "k-123"), **kw)


# ----------------------------------------------------------------------- wire
def test_request_shape_and_auth_header(mock):
    r = rev(mock).review({"symbol": "X/USDT"})
    assert r.decision == "approve" and r.source == "llm"
    req = mock.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer k-123"
    b = req["body"]
    assert b["model"] == "test-model" and b["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in b["messages"]] == ["system", "user"]
    assert "JSON Schema" in b["messages"][0]["content"] and "decision" in b["messages"][0]["content"]
    assert json.loads(b["messages"][1]["content"]) == {"symbol": "X/USDT"}


def test_no_auth_header_for_keyless_local_server(mock):
    rev(mock, api_key="").review({})
    assert "Authorization" not in mock.requests[0]["headers"]


@pytest.mark.parametrize("content", [
    json.dumps(REVIEW_OK),
    "```json\n" + json.dumps(REVIEW_OK) + "\n```",
    "Sure! Here is my answer:\n" + json.dumps(REVIEW_OK) + "\nHope that helps.",
    "<think>let me reason about this {not json}</think>\n" + json.dumps(REVIEW_OK),
])
def test_messy_model_output_is_parsed(mock, content):
    mock.default = lambda b: (200, chat(content))
    assert rev(mock).review({}).decision == "approve"


def test_validation_still_clamps_untrusted_model_output(mock):
    bad = dict(REVIEW_OK, decision="reduce", size_multiplier=50, confidence=7)
    mock.default = lambda b: (200, chat(json.dumps(bad)))
    r = rev(mock).review({})
    assert r.size_multiplier == 0.9 and r.confidence == 1.0               # a model can never enlarge a trade


@pytest.mark.parametrize("script,needle", [
    ([(401, "nope", {})], "API key rejected"),
    ([(404, "no such model", {})], "model or URL not found"),
    ([(500, "boom", {})], "HTTP 500"),
    ([(200, chat("{"), {})], "valid JSON"),
    ([(200, chat(json.dumps(REVIEW_OK), finish="length"), {})], "truncated"),
    ([(200, chat(""), {})], "empty"),
    ([(200, {"unexpected": 1}, {})], "unexpected response shape"),
    ([(200, chat(json.dumps({"decision": "approve"})), {})], "malformed review"),
])
def test_failures_become_llmerror_with_hints(mock, script, needle):
    mock.script = list(script)
    with pytest.raises(LLMError, match=needle):
        rev(mock).review({})


def test_rate_limit_retries_once_then_succeeds_or_fails(mock):
    slept = []
    mock.script = [(429, "slow down", {"retry-after": "2"})]
    r = rev(mock, sleep=slept.append).review({})
    assert r.decision == "approve" and slept == [2.0] and len(mock.requests) == 2
    mock.requests.clear()
    mock.script = [(429, "x", {}), (429, "x", {})]
    with pytest.raises(LLMError, match="rate limited"):
        rev(mock, sleep=lambda s: None).review({})
    assert len(mock.requests) == 2


def test_json_mode_is_dropped_if_server_rejects_it(mock):
    mock.script = [(400, "response_format json_object is not supported", {})]
    assert rev(mock).review({}).decision == "approve"
    assert "response_format" in mock.requests[0]["body"] and "response_format" not in mock.requests[1]["body"]


def test_timeout_and_connection_errors(mock):
    def slow(body):
        import time
        time.sleep(1.5)
        return 200, chat(json.dumps(REVIEW_OK))
    mock.default = slow
    with pytest.raises(LLMError):
        rev(mock, timeout=0.3).review({})
    with pytest.raises(LLMError):
        OpenAICompatReviewer("m", "http://127.0.0.1:9/v1", timeout=2).review({})     # nothing listening


def test_narrate_returns_plain_text_without_json_mode(mock):
    mock.default = lambda b: (200, chat("Looks fine overall."))
    assert rev(mock).narrate({"x": 1}) == "Looks fine overall."
    assert "response_format" not in mock.requests[0]["body"]


# ---------------------------------------------------------------------- scout
def scout_payload(sym, conf=0.9):
    return {"market_view": "calm", "candidates": [
        {"symbol": sym, "setup_type": "breakout", "confidence": conf, "thesis": "t", "key_risks": ["r"]}]}


def test_scout_over_compat_filters_hallucinations(mock):
    mock.default = lambda b: (200, chat(json.dumps(scout_payload("MADEUP/USDT"))))
    res = rev(mock).scout({"universe": [{"symbol": "ETH/USDT"}], "min_confidence": 0.65})
    assert res.candidates == [] and res.dropped
    mock.default = lambda b: (200, chat(json.dumps(scout_payload("eth/usdt"))))
    assert rev(mock).scout({"universe": [{"symbol": "ETH/USDT"}]}).candidates[0].symbol == "ETH/USDT"
    assert mock.requests[-1]["body"]["model"] == "test-model"


def test_scout_model_override(mock):
    mock.default = lambda b: (200, chat(json.dumps(scout_payload("ETH/USDT"))))
    rev(mock, scout_model="big-model").scout({"universe": [{"symbol": "ETH/USDT"}]})
    assert mock.requests[0]["body"]["model"] == "big-model"


# ------------------------------------------------------------- config / wiring
def test_parse_json_object_rejects_garbage():
    for bad in ("", "no json here", "[1,2,3]", "{broken"):
        with pytest.raises(LLMError):
            parse_json_object(bad)


def test_off_reasons_and_factory_for_compat_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    base = dict(llm_enabled=True, llm_provider="openai_compat")
    assert "LLM_BASE_URL" in llm_off_reason(Config(**base))
    assert "LLM_MODEL" in llm_off_reason(Config(**base, llm_base_url="http://localhost:11434/v1"))
    assert "LLM_API_KEY" in llm_off_reason(Config(**base, llm_base_url="https://api.groq.com/openai/v1",
                                                  llm_model="llama-3.3-70b-versatile"))
    ollama = Config(**base, llm_base_url="http://localhost:11434/v1", llm_model="qwen2.5:7b")
    assert llm_off_reason(ollama) is None                                  # local server needs no key
    r = make_reviewer(ollama)
    assert isinstance(r, OpenAICompatReviewer) and r.provider == "localhost"
    assert status_text(ollama, r).startswith("ON (localhost: qwen2.5:7b")
    groq = Config(**base, llm_base_url="https://api.groq.com/openai/v1", llm_model="m", llm_api_key="gsk")
    assert isinstance(make_reviewer(groq), OpenAICompatReviewer)
    assert isinstance(make_reviewer(Config(llm_enabled=False)), RulesOnlyReviewer)
    with pytest.raises(ValueError):
        Config(llm_provider="skynet").validate()


def test_no_paid_call_is_possible_without_an_anthropic_key(monkeypatch):
    """Free-only guarantee: the paid SDK is only constructed when provider=anthropic AND a key exists."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_reviewer(Config(llm_enabled=True, llm_provider="anthropic")), RulesOnlyReviewer)
    assert isinstance(make_reviewer(Config(llm_enabled=True, llm_provider="openai_compat",
                                           llm_base_url="http://localhost:11434/v1", llm_model="m")),
                      OpenAICompatReviewer)


# -------------------------------------------------------------- end to end
def test_full_bot_run_over_the_openai_protocol(tmp_path):
    """Rules review + scout both go through the mock server; trades get reviewed, scout trades are tagged."""
    def respond(body):
        system = body["messages"][0]["content"]
        if "scout in a PAPER-trading" in system:
            snap = json.loads(body["messages"][1]["content"])
            free = [r["symbol"] for r in snap["universe"] if not r["already_held"]]
            return 200, chat(json.dumps(scout_payload(free[0], 0.9) if free else {"market_view": "", "candidates": []}))
        return 200, chat(json.dumps(REVIEW_OK))

    m = Mock(default=respond)
    try:
        reviewer = OpenAICompatReviewer("local-model", m.url, "", sleep=lambda s: None)
        bot, data = make_bot(tmp_path, reviewer=reviewer, llm_scout_enabled=True)
        run(bot, data, 150)
        sources = {t["source"] for t in bot.j._all("SELECT source FROM trades")}
        assert "llm_scout" in sources and "rules" in sources
        assert len(m.requests) > 5
        assert all(r["path"] == "/v1/chat/completions" for r in m.requests)
        assert bot.j.kv_get("llm_status").startswith("ON (127.0.0.1: local-model")
    finally:
        m.close()
