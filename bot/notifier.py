"""Console + optional Telegram notifications. Credentials come ONLY from the environment."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

log = logging.getLogger("bot.notify")


class Notifier:
    def __init__(self, cfg, session: Optional[Any] = None):
        self.token = cfg.telegram_bot_token
        self.chat_id = cfg.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)
        self.http = session or requests
        self._warned = False

    def send(self, text: str, level: str = "info") -> bool:
        getattr(log, "warning" if level in {"warn", "error"} else "info")("NOTIFY %s", text.replace("\n", " | "))
        if not self.enabled:
            return False
        try:
            r = self.http.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                               data={"chat_id": self.chat_id, "text": text[:4000]}, timeout=10)
            if getattr(r, "status_code", 200) >= 400:
                if not self._warned:
                    log.warning("Telegram rejected message (HTTP %s) - check token / chat id", r.status_code)
                    self._warned = True
                return False
            return True
        except Exception as exc:                  # never let alerting crash trading
            if not self._warned:
                log.warning("Telegram send failed: %s", type(exc).__name__)
                self._warned = True
            return False

    # ---- formatted helpers
    def entry(self, sym: str, pos: Any, reason: str, verdict: Any) -> None:
        self.send(
            f"🟢 PAPER BUY {sym}\nqty {pos.qty:.6f} @ {pos.entry_price:.6g}\n"
            f"stop {pos.stop_price:.6g} | target {pos.target_price:.6g}\n"
            f"why: {reason}\nreview: {verdict.source}/{verdict.decision} "
            f"(x{verdict.size_multiplier:.2f}) {verdict.rationale[:200]}")

    def exit(self, c: Dict[str, Any]) -> None:
        icon = "✅" if c["pnl"] > 0 else "🔻"
        self.send(f"{icon} PAPER SELL {c['symbol']} ({c['reason']})\n"
                  f"exit {c['exit_price']:.6g} | PnL {c['pnl']:+.4f} USDT ({c['pnl_pct'] * 100:+.2f}%)")
