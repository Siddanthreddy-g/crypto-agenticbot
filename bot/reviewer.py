"""Post-trade reviewer: deterministic stats + optional LLM narrative -> reports/review-*.md.
It only WRITES A REPORT. Nothing here changes live parameters."""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .journal import Journal
from .llm import LLMError
from .stats import max_drawdown_pct, trade_stats


def collect_stats(j: Journal) -> Dict[str, Any]:
    closed = j.get_closed_trades()
    out: Dict[str, Any] = {"overall": trade_stats([t["pnl"] or 0 for t in closed])}
    by_reason, by_symbol = defaultdict(list), defaultdict(list)
    for t in closed:
        by_reason[t["reason"]].append(t["pnl"] or 0)
        by_symbol[t["symbol"]].append(t["pnl"] or 0)
    out["by_exit_reason"] = {k: trade_stats(v) for k, v in by_reason.items()}
    out["by_symbol"] = {k: trade_stats(v) for k, v in by_symbol.items()}
    out["max_drawdown_pct"] = max_drawdown_pct([r["equity"] for r in j.equity_curve(50000)])
    llm = j.recent_decisions(1000, agent="llm_review")
    out["llm_review_counts"] = {d: sum(1 for x in llm if x["decision"] == d) for d in {x["decision"] for x in llm}}
    return out


def _fmt(d: Dict[str, Any]) -> str:
    pf = d["profit_factor"]
    return (f"trades {d['trades']} | win {d['win_rate_pct']:.1f}% | PnL {d['total_pnl']:+.4f} | "
            f"expectancy {d['expectancy']:+.4f} | PF {'inf' if pf == float('inf') else f'{pf:.2f}'}")


def build_report(j: Journal, reviewer: Optional[Any] = None, out_dir: str = "reports") -> str:
    st = collect_stats(j)
    lines = [f"# Post-trade review - {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", "",
             "_Paper trading. Informational only; nothing here is applied automatically._", "",
             "## Overall", _fmt(st["overall"]) + f" | max drawdown {st['max_drawdown_pct']:.2f}%", "",
             "## By exit reason"]
    lines += [f"- **{k}**: {_fmt(v)}" for k, v in st["by_exit_reason"].items()] or ["- (no closed trades)"]
    lines += ["", "## By symbol"]
    lines += [f"- **{k}**: {_fmt(v)}" for k, v in st["by_symbol"].items()] or ["- (no closed trades)"]
    lines += ["", f"## LLM review decisions\n{st['llm_review_counts'] or 'none'}", ""]
    if reviewer is not None and getattr(reviewer, "model", "none") != "none" and st["overall"]["trades"] >= 5:
        try:
            lines += ["## Analyst notes (LLM)", reviewer.narrate(st), ""]
        except LLMError as exc:
            lines += [f"_LLM narrative unavailable: {exc}_", ""]
    elif st["overall"]["trades"] < 5:
        lines += ["_Fewer than 5 closed trades - too little data for conclusions._", ""]
    text = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"review-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
