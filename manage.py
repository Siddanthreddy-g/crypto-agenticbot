#!/usr/bin/env python3
"""Operator commands for the paper bot.

    python manage.py status                # equity, open positions, halt state, heartbeat
    python manage.py halt "reason"         # block new entries (open positions still managed)
    python manage.py resume                # clear halt / loss-streak pause
    python manage.py review                # write reports/review-*.md (LLM narrative if enabled)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["status", "halt", "resume", "review", "killswitch"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--db")
    a = ap.parse_args()
    cfg = Config.from_env()
    if a.db:
        cfg = cfg.replace(database_path=a.db)
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
