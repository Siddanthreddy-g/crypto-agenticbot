"""Performance statistics shared by dashboard, backtester and post-trade reviewer."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


def trade_stats(pnls: Iterable[float]) -> Dict[str, Any]:
    p = [float(x) for x in pnls]
    n = len(p)
    wins = [x for x in p if x > 0]
    losses = [x for x in p if x <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": (len(wins) / n * 100) if n else 0.0,
        "total_pnl": sum(p),
        "avg_win": (gp / len(wins)) if wins else 0.0,
        "avg_loss": (-gl / len(losses)) if losses else 0.0,
        "expectancy": (sum(p) / n) if n else 0.0,
        "profit_factor": (gp / gl) if gl else (float("inf") if gp else 0.0),
    }


def max_drawdown_pct(equity: List[float]) -> float:
    peak, mdd = float("-inf"), 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak)
    return mdd * 100
