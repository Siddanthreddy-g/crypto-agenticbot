"""Central configuration. Everything is read from environment variables / .env.

Safety invariant: this distribution is PAPER-ONLY. `validate()` rejects any
TRADING_MODE other than "paper" and there is no authenticated exchange client
anywhere in the code base.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Mapping, Optional

try:  # python-dotenv is optional at import time (tests inject env dicts)
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass(frozen=True)
class Config:
    # safety / data
    trading_mode: str = "paper"
    data_source: str = "exchange"          # exchange | synthetic
    exchange_id: str = "binance"
    quote_asset: str = "USDT"
    timeframe: str = "5m"
    candle_limit: int = 300
    scan_interval: int = 30
    max_symbols: int = 15
    min_quote_volume: float = 5_000_000.0
    watchlist: tuple = ()

    # paper account
    initial_cash: float = 100.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_notional: float = 10.0

    # risk
    risk_per_trade: float = 0.005
    max_position_fraction: float = 0.25
    max_total_exposure: float = 0.60
    max_open_positions: int = 3
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.10
    max_consecutive_losses: int = 3
    loss_pause_minutes: int = 60
    symbol_cooldown_candles: int = 6
    max_hold_candles: int = 48

    # strategy
    rsi_min: float = 50.0
    rsi_max: float = 68.0
    volume_multiplier: float = 1.2
    atr_stop_mult: float = 1.5
    reward_risk: float = 2.0
    min_stop_pct: float = 0.008
    max_stop_pct: float = 0.03
    max_cost_to_target: float = 0.35
    min_net_rr: float = 1.1

    # llm
    llm_enabled: bool = False
    llm_provider: str = "anthropic"        # anthropic | openai_compat (Ollama / Groq / Gemini / ... free options)
    llm_base_url: str = ""                 # for openai_compat, e.g. http://localhost:11434/v1
    llm_api_key: str = ""                  # for openai_compat; not needed for local Ollama
    llm_timeout_seconds: int = 90
    llm_model: str = "claude-opus-5-5"
    llm_fail_mode: str = "reject"          # reject | approve
    llm_max_calls_per_hour: int = 30
    llm_min_confidence: float = 0.5
    daily_llm_review: bool = True

    # llm scout (LLM proposes extra candidates; code sets stops/sizes/limits)
    llm_scout_enabled: bool = False
    llm_scout_model: str = ""              # empty -> same as LLM_MODEL
    llm_scout_interval_minutes: int = 15
    llm_scout_min_confidence: float = 0.65
    llm_scout_size_factor: float = 0.5     # scout trades use at most this fraction of normal size
    llm_scout_max_trades_per_day: int = 4
    llm_scout_max_concurrent: int = 1

    # telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # files / logging
    database_path: str = "trader.db"
    kill_switch_file: str = "KILL_SWITCH"
    log_level: str = "INFO"

    # ------------------------------------------------------------------ build
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, load_dotenv_file: bool = True) -> "Config":
        if env is None:
            if load_dotenv_file and load_dotenv is not None:
                load_dotenv()
            env = os.environ

        def get(name: str, default):
            raw = env.get(name)
            if raw is None or str(raw).strip() == "":
                return default
            return raw

        kw = {}
        for f in fields(cls):
            name = f.name.upper()
            default = f.default
            raw = get(name, None)
            if raw is None:
                continue
            if f.name == "watchlist":
                kw[f.name] = tuple(s.strip().upper() for s in str(raw).split(",") if s.strip())
            elif isinstance(default, bool):
                kw[f.name] = _bool(raw)
            elif isinstance(default, int):
                kw[f.name] = int(float(raw))
            elif isinstance(default, float):
                kw[f.name] = float(raw)
            else:
                kw[f.name] = str(raw).strip()
        # a few env names differ from field names
        aliases = {
            "initial_cash": "INITIAL_CASH_USDT",
            "min_quote_volume": "MIN_QUOTE_VOLUME_USDT",
            "min_notional": "MIN_NOTIONAL_USDT",
            "scan_interval": "SCAN_INTERVAL_SECONDS",
        }
        for fname, ename in aliases.items():
            raw = get(ename, None)
            if raw is not None:
                kw[fname] = type(getattr(cls, fname))(float(raw)) if isinstance(getattr(cls, fname), (int, float)) else raw
        if "trading_mode" in kw:
            kw["trading_mode"] = kw["trading_mode"].lower()
        return cls(**kw)

    # --------------------------------------------------------------- validate
    def validate(self) -> "Config":
        def rng(name, lo, hi, incl_lo=False):
            v = getattr(self, name)
            ok = (lo <= v if incl_lo else lo < v) and v <= hi
            if not ok:
                raise ValueError(f"{name.upper()}={v} must be in {'[' if incl_lo else '('}{lo}, {hi}]")

        if self.trading_mode != "paper":
            raise ValueError("Safety lock: this distribution supports TRADING_MODE=paper only.")
        if self.data_source not in {"exchange", "synthetic"}:
            raise ValueError("DATA_SOURCE must be 'exchange' or 'synthetic'")
        if self.llm_provider not in {"anthropic", "openai_compat"}:
            raise ValueError("LLM_PROVIDER must be 'anthropic' or 'openai_compat'")
        if self.llm_timeout_seconds < 5:
            raise ValueError("LLM_TIMEOUT_SECONDS must be >= 5")
        if self.llm_fail_mode not in {"reject", "approve"}:
            raise ValueError("LLM_FAIL_MODE must be 'reject' or 'approve'")
        if self.initial_cash <= 0:
            raise ValueError("INITIAL_CASH_USDT must be positive")
        if self.scan_interval < 5:
            raise ValueError("SCAN_INTERVAL_SECONDS must be >= 5")
        rng("llm_scout_size_factor", 0, 1)
        rng("llm_scout_min_confidence", 0, 1, incl_lo=True)
        if self.llm_scout_interval_minutes < 5:
            raise ValueError("LLM_SCOUT_INTERVAL_MINUTES must be >= 5")
        if self.llm_scout_max_trades_per_day < 0 or self.llm_scout_max_concurrent < 0:
            raise ValueError("scout caps must be >= 0")
        rng("risk_per_trade", 0, 0.02)
        rng("max_daily_loss", 0, 0.10)
        rng("max_drawdown", 0, 0.30)
        rng("max_position_fraction", 0, 1)
        rng("max_total_exposure", 0, 1)
        rng("fee_rate", 0, 0.01, incl_lo=True)
        rng("slippage_rate", 0, 0.01, incl_lo=True)
        rng("min_stop_pct", 0, 0.2)
        rng("max_stop_pct", 0, 0.5)
        if self.min_stop_pct > self.max_stop_pct:
            raise ValueError("MIN_STOP_PCT must be <= MAX_STOP_PCT")
        if self.reward_risk <= 0:
            raise ValueError("REWARD_RISK must be positive")
        if self.max_open_positions < 1:
            raise ValueError("MAX_OPEN_POSITIONS must be >= 1")
        if self.candle_limit < 120:
            raise ValueError("CANDLE_LIMIT must be >= 120 (indicators need warm-up)")
        if self.rsi_min >= self.rsi_max:
            raise ValueError("RSI_MIN must be < RSI_MAX")
        return self

    # --------------------------------------------------------------- helpers
    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    def replace(self, **kw) -> "Config":
        from dataclasses import replace
        return replace(self, **kw)
