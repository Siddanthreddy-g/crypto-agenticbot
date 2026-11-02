"""Main orchestration loop (paper trading).

One cycle:
  1. manage exits for open positions (replay missed candles + live ticker check)
  2. mark-to-market, daily roll-over, drawdown halt
  3. scan universe -> deterministic signals on NEW closed candles -> rank
  4. risk pre-check -> LLM review (veto/shrink only) -> final sizing -> paper fill
  5. snapshot equity + heartbeat
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agents import (LLMReviewAgent, MarketScannerAgent, PerformanceAgent, RiskSupervisorAgent,
                     StrategyAnalystAgent, build_review_context)
from .broker import PaperBroker, df_to_candles
from .config import Config
from .journal import Journal
from .llm import make_reviewer
from .notifier import Notifier
from .risk import RiskGuard, RiskManager
from .strategy import Signal, StrategyParams

log = logging.getLogger("bot.runner")


class Bot:
    def __init__(self, cfg: Config, data: Any, journal: Optional[Journal] = None,
                 notifier: Optional[Notifier] = None, reviewer: Any = None):
        self.cfg = cfg.validate()
        self.data = data
        self.j = journal or Journal(cfg.database_path)
        self.broker = PaperBroker(cfg.initial_cash, cfg.fee_rate, cfg.slippage_rate, self.j)
        self.params = StrategyParams.from_config(cfg)
        self.guard = RiskGuard(cfg, self.j)
        self.risk = RiskSupervisorAgent(RiskManager(cfg), self.guard)
        self.scanner = MarketScannerAgent(data, cfg)
        self.analyst = StrategyAnalystAgent(data, cfg, self.params)
        self.reviewer = reviewer if reviewer is not None else make_reviewer(cfg)
        self.llm = LLMReviewAgent(self.reviewer, cfg, self.j)
        self.perf = PerformanceAgent(self.j)
        self.notifier = notifier or Notifier(cfg)
        self.tf_ms = data.timeframe_ms(cfg.timeframe)
        self.marks: Dict[str, float] = {}
        self.seen: Dict[str, int] = {}
        self._last_block: Optional[str] = None
        self._consec_errors = 0
        self._stop = False
        log.info("Recovered paper portfolio: cash=%.4f open=%d realized=%.4f closed=%d",
                 self.broker.cash, len(self.broker.positions), self.broker.realized_pnl,
                 self.broker.closed_trades)

    # ------------------------------------------------------------------ utils
    def stop(self) -> None:
        self._stop = True

    def _event(self, now_iso: str, symbol: Optional[str], agent: str, decision: str, detail: Any = "") -> None:
        self.j.add_decision(now_iso, symbol, agent, decision, detail)

    def _benchmark(self, tickers: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        sym = "BTC/USDT" if "BTC/USDT" in tickers else (max(tickers, key=lambda s: tickers[s]["quoteVolume"]) if tickers else None)
        if not sym:
            return None
        try:
            df = self.data.candles(sym, self.cfg.timeframe, 300)
            c = df["close"]
            f = lambda k: round((float(c.iloc[-1]) / float(c.iloc[-1 - k]) - 1) * 100, 3) if len(c) > k else None
            return {"symbol": sym, "change_pct_1h_ish_12c": f(12), "change_pct_4h_ish_48c": f(48),
                    "change_pct_24h_ish_288c": f(288)}
        except Exception as exc:
            log.debug("benchmark unavailable: %s", exc)
            return None

    # ------------------------------------------------------------------ cycle
    def run_cycle(self) -> Dict[str, Any]:
        cfg = self.cfg
        now: datetime = self.data.now()
        now_iso = now.astimezone(timezone.utc).isoformat()
        now_ms = int(now.timestamp() * 1000)
        tickers = self.data.tickers()
        universe = self.scanner.run(tickers)
        held = list(self.broker.positions)
        summary = {"exits": 0, "entries": 0, "signals": 0, "blocked": None}

        # ---- 1. exits
        for sym in held:
            try:
                closed = None
                df = self.data.candles(sym, cfg.timeframe, cfg.candle_limit)
                if df is not None and not df.empty:
                    closed = self.broker.evaluate_candles(sym, df_to_candles(df), self.tf_ms,
                                                          cfg.max_hold_candles, now_iso)
                if not closed and sym in tickers:
                    closed = self.broker.evaluate_price(sym, tickers[sym]["last"], now_iso)
                if closed:
                    self._on_close(closed, now)
                    summary["exits"] += 1
            except Exception as exc:
                log.warning("exit check failed for %s: %s", sym, exc)

        # ---- kill switch with CLOSE_ALL
        ks = cfg.kill_switch_file
        if ks and os.path.exists(ks):
            try:
                content = open(ks).read().strip().upper()
            except OSError:
                content = ""
            if content.startswith("CLOSE") and self.broker.positions:
                for sym in list(self.broker.positions):
                    px = tickers.get(sym, {}).get("last") or self.marks.get(sym)
                    if px:
                        closed = self.broker.close_at_market(sym, px, "kill_switch_close_all", now_iso)
                        if closed:
                            self._on_close(closed, now)
                            summary["exits"] += 1

        # ---- 2. marks / equity / day roll / halts
        for sym in set(universe) | set(self.broker.positions):
            if sym in tickers:
                self.marks[sym] = tickers[sym]["last"]
        equity = self.broker.equity(self.marks)
        rolled, prev_day = self.guard.roll_day(now, equity)
        if rolled and prev_day is not None:
            self._on_new_day(prev_day, equity)
        halted_now = self.guard.update_equity(equity)
        if halted_now:
            self._event(now_iso, None, "risk", "HALT", halted_now)
            self.notifier.send(f"🛑 TRADING HALTED: {halted_now}\nOpen positions still managed. "
                               f"Resume with: python manage.py resume", "warn")

        # ---- 3/4. entries
        blocked = self.guard.blocked_reason(now, equity)
        summary["blocked"] = blocked
        if blocked != self._last_block:
            if blocked:
                self._event(now_iso, None, "risk", "ENTRIES_BLOCKED", blocked)
                log.warning("Entries blocked: %s", blocked)
            elif self._last_block:
                log.info("Entries unblocked")
            self._last_block = blocked
        if not blocked:
            cands = self._collect_signals(universe, now, now_ms, now_iso)
            summary["signals"] = len(cands)
            cands.sort(key=lambda x: -x[0].score)
            for sig, df in cands:
                if len(self.broker.positions) >= cfg.max_open_positions:
                    break
                if self._try_enter(sig, df, tickers, now, now_iso):
                    summary["entries"] += 1

        # ---- 5. snapshot / heartbeat
        equity = self.broker.equity(self.marks)
        self.j.snapshot(now_iso, self.broker.cash, equity)
        self.j.set_marks({s: p for s, p in self.marks.items()
                          if s in self.broker.positions or s in universe}, now_iso)
        self.j.kv_set("last_cycle_sim", now_iso)
        self.j.kv_set("last_cycle_wall", datetime.now(timezone.utc).isoformat())
        self.j.kv_set("universe", ",".join(universe))
        self.j.kv_set("mode", "PAPER")
        log.info("equity=%.4f cash=%.4f open=%d realized=%.4f signals=%d%s", equity, self.broker.cash,
                 len(self.broker.positions), self.broker.realized_pnl, summary["signals"],
                 f" BLOCKED[{blocked}]" if blocked else "")
        summary["equity"] = equity
        return summary

    # ---------------------------------------------------------------- signals
    def _collect_signals(self, universe: List[str], now: datetime, now_ms: int,
                         now_iso: str) -> List[Tuple[Signal, Any]]:
        out: List[Tuple[Signal, Any]] = []
        equity = self.broker.equity(self.marks)
        for sym in universe:
            if sym in self.broker.positions:
                continue
            last_seen = self.seen.get(sym)
            if last_seen is not None and now_ms < last_seen + 2 * self.tf_ms:
                continue                                    # no new candle can have closed yet
            if self.guard.blocked_reason(now, equity, sym):
                continue
            try:
                sig, df = self.analyst.analyze(sym)
            except Exception as exc:
                log.warning("analysis failed for %s: %s", sym, exc)
                continue
            if sig is None:
                continue
            if now_ms - (sig.candle_ts + self.tf_ms) > 2 * self.tf_ms:
                log.debug("stale data for %s - skipped", sym)      # watchdog: never trade stale candles
                continue
            if self.seen.get(sym) == sig.candle_ts:
                continue
            self.seen[sym] = sig.candle_ts
            if sig.should_buy:
                out.append((sig, df))
        return out

    # ----------------------------------------------------------------- entry
    def _try_enter(self, sig: Signal, df: Any, tickers: Dict[str, Dict[str, float]],
                   now: datetime, now_iso: str) -> bool:
        cfg, sym = self.cfg, sig.symbol
        ref = tickers.get(sym, {}).get("last") or sig.price
        equity = self.broker.equity(self.marks)
        sizing = dict(equity=equity, cash=self.broker.cash, entry_price=ref * (1 + cfg.slippage_rate),
                      stop_pct=sig.stop_pct, open_positions=len(self.broker.positions),
                      open_notional=self.broker.open_notional(self.marks),
                      round_qty=lambda q: self.data.round_qty(sym, q),
                      min_notional=self.data.min_notional(sym))

        pre = self.risk.size(**sizing)                              # cheap pre-check before paying for LLM
        if not pre.approved:
            self._event(now_iso, sym, "risk", "VETO", pre.reason)
            return False

        ds = self.guard.day_start_equity
        ctx = None
        if getattr(self.reviewer, "model", "") != "none":
            ctx = build_review_context(
                sig, df, cfg, self.perf.summary(), list(self.broker.positions),
                self._benchmark(tickers), ((equity / ds - 1) * 100) if ds else 0.0, now_iso)
        verdict = self.llm.review(sig, ctx or {"symbol": sym}, now_iso)
        if not verdict.approved:
            self._event(now_iso, sym, "strategy", "SKIPPED_BY_REVIEW", verdict.rationale)
            log.info("SKIP %s (%s): %s", sym, verdict.source, verdict.rationale[:120])
            return False

        final = self.risk.size(**{**sizing, "size_multiplier": verdict.size_multiplier})
        if not final.approved:
            self._event(now_iso, sym, "risk", "VETO", final.reason)
            return False
        pos = self.broker.enter_long(sym, final.qty, ref, sig.stop_pct, sig.target_pct, sig.reason,
                                     now_iso, sig.candle_ts,
                                     meta={"score": sig.score, "indicators": sig.indicators,
                                           "review": verdict.__dict__})
        if not pos:
            self._event(now_iso, sym, "broker", "REJECTED", "insufficient cash or invalid qty")
            return False
        self._event(now_iso, sym, "broker", "BUY", {"qty": pos.qty, "price": pos.entry_price,
                                                    "stop": pos.stop_price, "target": pos.target_price})
        self.marks[sym] = ref
        log.info("PAPER BUY %s qty=%.6f @ %.6g stop=%.6g target=%.6g", sym, pos.qty, pos.entry_price,
                 pos.stop_price, pos.target_price)
        self.notifier.entry(sym, pos, sig.reason, verdict)
        return True

    # ------------------------------------------------------------------ close
    def _on_close(self, closed: Dict[str, Any], now: datetime) -> None:
        self.guard.record_close(closed["symbol"], closed["pnl"], now, self.tf_ms)
        self._event(closed["exit_time"], closed["symbol"], "broker", "SELL:" + closed["reason"],
                    {"pnl": closed["pnl"], "exit": closed["exit_price"]})
        log.info("PAPER SELL %s %s pnl=%+.4f", closed["symbol"], closed["reason"], closed["pnl"])
        self.notifier.exit(closed)

    def _on_new_day(self, prev_day: str, equity: float) -> None:
        stats = self.perf.summary()
        self.notifier.send(
            f"📅 Daily summary ({prev_day})\nequity {equity:.2f} | trades {stats['trades']} | "
            f"win {stats['win_rate_pct']:.0f}% | PnL {stats['total_pnl']:+.2f} | "
            f"maxDD {stats['max_drawdown_pct']:.1f}%")
        if self.cfg.daily_llm_review and getattr(self.reviewer, "model", "") != "none":
            try:
                from .reviewer import build_report
                path = build_report(self.j, self.reviewer)
                log.info("Daily review written: %s", path)
            except Exception as exc:
                log.warning("daily review failed: %s", exc)

    # ------------------------------------------------------------- main loop
    def run_forever(self, max_cycles: Optional[int] = None, on_cycle: Optional[Callable[[], bool]] = None,
                    sleep: bool = True) -> None:
        n = 0
        self.notifier.send(f"🤖 Bot started (PAPER) | equity {self.broker.equity(self.marks):.2f} | "
                           f"LLM {'on' if getattr(self.reviewer, 'model', 'none') != 'none' else 'off'}")
        while not self._stop and (max_cycles is None or n < max_cycles):
            started = time.time()
            try:
                self.run_cycle()
                self._consec_errors = 0
            except KeyboardInterrupt:
                break
            except Exception as exc:
                self._consec_errors += 1
                log.exception("Cycle failed (%d in a row)", self._consec_errors)
                if self._consec_errors == 5:
                    self.notifier.send(f"⚠️ Bot failing repeatedly: {type(exc).__name__}: {exc}", "error")
            n += 1
            if on_cycle is not None and on_cycle() is False:
                break
            if sleep:
                backoff = min(300, 5 * 2 ** min(self._consec_errors, 6)) if self._consec_errors else 0
                wait = max(1.0, self.cfg.scan_interval - (time.time() - started)) + backoff
                end = time.time() + wait
                while time.time() < end and not self._stop:       # interruptible sleep
                    time.sleep(min(1.0, end - time.time()))
        self.notifier.send("🛑 Bot stopped (paper positions persist in the journal)")
