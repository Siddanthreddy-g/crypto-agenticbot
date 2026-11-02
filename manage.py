#!/usr/bin/env python3
"""Operator commands for the paper bot.

    python manage.py status                # equity, open positions, halt state, heartbeat
    python manage.py halt "reason"         # block new entries (open positions still managed)
    python manage.py resume                # clear halt / loss-streak pause
    python manage.py review                # write reports/review-*.md (LLM narrative if enabled)
    python manage.py llmtest               # ONE real LLM review call on live data - verifies key / model / credits
    python manage.py killswitch [close]    # create KILL_SWITCH file (optionally: close all positions)
    python manage.py killswitch off        # remove KILL_SWITCH file
"""
import argparse
import os
import sys
from datetime import datetime, timezone

from bot.config import Config
from bot.journal import Journal
from bot.risk import RiskGuard


def llm_test(cfg) -> int:
    """Make one real review call on the most liquid symbol's current candles (no trade is placed)."""
    import time
    from bot.agents import build_review_context
    from bot.llm import AnthropicReviewer, LLMError, llm_off_reason
    from bot.strategy import StrategyParams, evaluate

    why = llm_off_reason(cfg)
    if why:
        print(f"LLM review is OFF: {why}.\n\nTo enable, edit .env (notepad .env) and set:\n"
              "  LLM_ENABLED=true\n  ANTHROPIC_API_KEY=<your key from console.anthropic.com>\nthen run this again.")
        return 1
    try:
        from bot.market_data import ExchangeData
        data = ExchangeData(cfg.exchange_id, cfg.quote_asset)
        src = f"live {cfg.exchange_id} data"
    except Exception as exc:
        from bot.synthetic import SyntheticData
        data, src = SyntheticData(timeframe=cfg.timeframe), f"synthetic data (exchange unreachable: {type(exc).__name__})"
    tk = data.tickers()
    sym = max(tk, key=lambda s: tk[s]["quoteVolume"])
    df = data.candles(sym, cfg.timeframe, cfg.candle_limit)
    sig = evaluate(sym, df, StrategyParams.from_config(cfg))
    ctx = build_review_context(sig, df, cfg, {}, [], None, 0.0, data.now().isoformat())
    print(f"Calling {cfg.llm_model} on {sym} using {src} (rule signal is "
          f"{'BUY' if sig.should_buy else 'NOT a buy: ' + sig.reason}; this is only a connectivity test)...")
    t0 = time.time()
    try:
        r = AnthropicReviewer(cfg.llm_model).review(ctx)
    except LLMError as exc:
        msg = str(exc)
        print(f"\nFAILED: {msg}")
        hints = {"401": "API key is invalid - create a new one at console.anthropic.com -> API keys.",
                 "authentication": "API key is invalid - create a new one at console.anthropic.com -> API keys.",
                 "credit": "Your API account has no credits - add a payment method/credits in the Console (Plans & Billing).",
                 "404": f"Model name not found: check LLM_MODEL (now '{cfg.llm_model}').",
                 "not_found": f"Model name not found: check LLM_MODEL (now '{cfg.llm_model}').",
                 "429": "Rate limited - wait a minute and retry.",
                 "Connection": "Network problem reaching api.anthropic.com (firewall/VPN/proxy?)."}
        for k, h in hints.items():
            if k.lower() in msg.lower():
                print("Hint:", h)
                break
        return 2
    print(f"\nOK in {time.time() - t0:.1f}s  (model {r.model})")
    print(f"decision   : {r.decision}  (size x{r.size_multiplier:.2f}, confidence {r.confidence:.2f})")
    print(f"rationale  : {r.rationale}")
    print(f"key risks  : {'; '.join(r.key_risks) or '-'}")
    print("\nThe LLM review is working. Start the bot with: python run_bot.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["status", "halt", "resume", "review", "killswitch", "llmtest"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--db")
    a = ap.parse_args()
    cfg = Config.from_env()
    if a.db:
        cfg = cfg.replace(database_path=a.db)
    if a.cmd == "llmtest":
        return llm_test(cfg)
    if not os.path.exists(cfg.database_path) and a.cmd != "killswitch":
        print(f"No database at {cfg.database_path} - start the bot first.")
        return 1

    if a.cmd == "killswitch":
        if a.arg == "off":
            if os.path.exists(cfg.kill_switch_file):
                os.remove(cfg.kill_switch_file)
            print("kill switch removed")
        else:
            with open(cfg.kill_switch_file, "w") as f:
                f.write("CLOSE_ALL" if a.arg.lower() == "close" else "HALT")
            print(f"kill switch set ({'close all positions' if a.arg.lower() == 'close' else 'block new entries'})")
        return 0

    j = Journal(cfg.database_path)
    g = RiskGuard(cfg, j)
    if a.cmd == "status":
        kv = j.kv_all()
        port = j.load_portfolio()
        marks = j.get_marks()
        eq = port["cash"] + sum(float(t["qty"]) * marks.get(s, float(t["entry_price"]))
                                for s, t in port["positions"].items())
        print(f"mode            : {kv.get('mode', '?')}")
        print(f"equity / cash   : {eq:.4f} / {port['cash']:.4f}  (start {port['initial_cash']:.2f})")
        print(f"realized PnL    : {port['realized_pnl']:+.4f}  trades={port['closed_trades']} "
              f"W/L={port['wins']}/{port['losses']}")
        print(f"halted          : {g.halted} {g.halt_reason}")
        print(f"kill switch file: {os.path.exists(cfg.kill_switch_file)}")
        lw = kv.get("last_cycle_wall")
        if lw:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(lw)).total_seconds()
            print(f"last heartbeat  : {age:.0f}s ago {'(STALE - is the bot running?)' if age > 5 * cfg.scan_interval else ''}")
        for s, t in port["positions"].items():
            px = marks.get(s, float(t["entry_price"]))
            print(f"  OPEN {s}: qty {t['qty']:.6g} entry {t['entry_price']:.6g} now {px:.6g} "
                  f"stop {t['stop_price']:.6g} target {t['target_price']:.6g}")
    elif a.cmd == "halt":
        g.halt(a.arg or "manual_halt")
        print("halted")
    elif a.cmd == "resume":
        eq = None
        g.resume(reset_peak_to=eq)
        print("halt/pause cleared (peak equity NOT reset; use with care after a drawdown halt)")
    elif a.cmd == "review":
        from bot.llm import make_reviewer
        from bot.reviewer import build_report
        print("wrote", build_report(j, make_reviewer(cfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
