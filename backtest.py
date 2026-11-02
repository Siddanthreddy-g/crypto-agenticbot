#!/usr/bin/env python3
"""Portfolio-level, multi-symbol backtester that reuses the LIVE strategy / risk / broker code.

Realism rules
  * Signal on a CLOSED candle -> entry at the NEXT candle's open (+slippage, +fee).
  * Stop-first if stop and target are both touched in one candle; gap-aware fills; time stop.
  * Same position sizing, exposure caps, daily-loss limit, drawdown halt and cooldowns as live.
  * Indicators are causal and computed once (O(n)); no look-ahead.

Usage
  python backtest.py --demo                                   # offline synthetic data (plumbing check only)
  python backtest.py --download --symbols BTC/USDT,ETH/USDT --since 2026-06-01 --until 2026-10-01
  python backtest.py --csv btc.csv:BTC/USDT --csv eth.csv:ETH/USDT
  add  --folds 4              to report 4 consecutive out-of-sample periods
  add  --walk-forward 3       anchored walk-forward: tune on past data, test on the next unseen slice
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bot.broker import PaperBroker
from bot.config import Config
from bot.indicators import enrich
from bot.journal import Journal
from bot.risk import RiskGuard, RiskManager
from bot.stats import max_drawdown_pct, trade_stats
from bot.strategy import StrategyParams, signal_from_values
from bot.synthetic import DEFAULT_SYMBOLS, _parse_tf, generate_history


# ------------------------------------------------------------------ data prep
def prepare(frames: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for sym, df in frames.items():
        e = enrich(df)
        # robust ms conversion regardless of pandas datetime resolution
        ts = np.array([int(pd.Timestamp(t).timestamp() * 1000) for t in e["timestamp"]], dtype=np.int64)
        cols = {c: e[c].to_numpy(dtype=float) for c in
                ["open", "high", "low", "close", "volume", "ema_fast", "ema_slow", "ema_trend",
                 "rsi", "vwap", "vol_avg20", "atr"]}
        out[sym] = {"ts": ts, "idx": {int(t): i for i, t in enumerate(ts)}, **cols}
    return out


def simulate(prep: Dict[str, Dict[str, Any]], cfg: Config, params: StrategyParams, tf_ms: int,
             start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> Dict[str, Any]:
    cfg = cfg.replace(kill_switch_file="")
    j = Journal(":memory:")
    broker = PaperBroker(cfg.initial_cash, cfg.fee_rate, cfg.slippage_rate, j)
    guard, risk = RiskGuard(cfg, j), RiskManager(cfg)
    timeline = sorted({int(t) for d in prep.values() for t in d["ts"]
                       if (start_ms is None or t >= start_ms) and (end_ms is None or t < end_ms)})
    marks: Dict[str, float] = {}
    pending: List[Any] = []
    curve: List[float] = []
    in_market = 0
    first_close: Dict[str, float] = {}
    last_close: Dict[str, float] = {}

    def candle(sym, i, ts):
        d = prep[sym]
        return {"ts": ts, "open": d["open"][i], "high": d["high"][i], "low": d["low"][i], "close": d["close"][i]}

    for ts in timeline:
        now = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        iso = now.isoformat()
        # 1. exits of positions opened earlier
        for sym in list(broker.positions):
            i = prep[sym]["idx"].get(ts)
            if i is None:
                continue
            r = broker.evaluate_candle(sym, candle(sym, i, ts), tf_ms, cfg.max_hold_candles, iso)
            if r:
                guard.record_close(sym, r["pnl"], now, tf_ms)
        # 2. entries from last candle's signals -> fill at this candle's open
        equity = broker.equity(marks)
        for sig in sorted(pending, key=lambda s: -s.score):
            sym = sig.symbol
            i = prep[sym]["idx"].get(ts)
            if i is None or sym in broker.positions or len(broker.positions) >= cfg.max_open_positions:
                continue
            if guard.blocked_reason(now, equity, sym):
                continue
            ref = prep[sym]["open"][i]
            d = risk.size_position(equity=equity, cash=broker.cash, entry_price=ref * (1 + cfg.slippage_rate),
                                   stop_pct=sig.stop_pct, open_positions=len(broker.positions),
                                   open_notional=broker.open_notional(marks))
            if not d.approved:
                continue
            pos = broker.enter_long(sym, d.qty, ref, sig.stop_pct, sig.target_pct, sig.reason, iso,
                                    sig.candle_ts)
            if pos:
                marks[sym] = ref
                r = broker.evaluate_candle(sym, candle(sym, i, ts), tf_ms, cfg.max_hold_candles, iso)
                if r:
                    guard.record_close(sym, r["pnl"], now, tf_ms)
            equity = broker.equity(marks)
        pending = []
        # 3. mark to market, day roll, drawdown halt
        for sym, d in prep.items():
            i = d["idx"].get(ts)
            if i is not None:
                marks[sym] = d["close"][i]
                first_close.setdefault(sym, d["close"][i])
                last_close[sym] = d["close"][i]
        equity = broker.equity(marks)
        guard.roll_day(now, equity)
        guard.update_equity(equity)
        curve.append(equity)
        in_market += bool(broker.positions)
        # 4. new signals on candles that just closed
        if not guard.blocked_reason(now, equity):
            for sym, d in prep.items():
                i = d["idx"].get(ts)
                if i is None or sym in broker.positions:
                    continue
                sig = signal_from_values(sym, ts, d["close"][i], d["ema_fast"][i], d["ema_slow"][i],
                                         d["ema_trend"][i], d["rsi"][i], d["vwap"][i], d["volume"][i],
                                         d["vol_avg20"][i], d["atr"][i], params)
                if sig.should_buy:
                    pending.append(sig)

    # liquidate leftovers at last mark so every trade is counted
    end_iso = datetime.fromtimestamp((timeline[-1] if timeline else 0) / 1000, tz=timezone.utc).isoformat()
    for sym in list(broker.positions):
        broker.close_at_market(sym, marks.get(sym, broker.positions[sym].entry_price), "end_of_data", end_iso)
    final_equity = broker.cash
    if curve:
        curve[-1] = final_equity
    trades = j.get_closed_trades()
    s = trade_stats([t["pnl"] for t in trades])
    bh = [last_close[k] / first_close[k] - 1 for k in first_close] if first_close else []
    res = {
        "initial": cfg.initial_cash, "final": final_equity,
        "return_pct": (final_equity / cfg.initial_cash - 1) * 100,
        "max_drawdown_pct": max_drawdown_pct(curve), "time_in_market_pct": in_market / len(timeline) * 100 if timeline else 0,
        "buy_hold_eqweight_pct": (sum(bh) / len(bh) * 100) if bh else 0.0,
        "candles": len(timeline), "trades_list": trades, "equity_curve": curve, "timeline": timeline,
        **s,
    }
    j.close()
    return res


# ------------------------------------------------------------------- reports
def fmt(r: Dict[str, Any], title: str) -> str:
    pf = "inf" if r["profit_factor"] == float("inf") else f"{r['profit_factor']:.2f}"
    return (f"{title:<22} ret {r['return_pct']:+7.2f}% | trades {r['trades']:4d} | win {r['win_rate_pct']:5.1f}% | "
            f"PF {pf:>5} | exp {r['expectancy']:+.4f} | maxDD {r['max_drawdown_pct']:5.2f}% | "
            f"mkt {r['time_in_market_pct']:4.1f}% | B&H {r['buy_hold_eqweight_pct']:+6.2f}%")


def split_bounds(timeline: List[int], n: int) -> List[int]:
    step = len(timeline) / n
    return [timeline[int(round(step * k))] if k < n else timeline[-1] + 1 for k in range(n + 1)]


GRID = {"volume_multiplier": [1.2, 1.8], "atr_stop_mult": [1.0, 1.5, 2.0], "reward_risk": [1.5, 2.0, 2.5]}


def objective(r: Dict[str, Any], min_trades: int = 10) -> float:
    if r["trades"] < min_trades:
        return -1e9
    pf = r["profit_factor"] if math.isfinite(r["profit_factor"]) else 10.0
    return pf + r["return_pct"] / 100.0


def walk_forward(prep, cfg, base: StrategyParams, tf_ms, n_folds: int):
    from dataclasses import replace
    timeline = sorted({int(t) for d in prep.values() for t in d["ts"]})
    bounds = split_bounds(timeline, n_folds + 1)
    oos_trades, rets = [], []
    print(f"\nANCHORED WALK-FORWARD: {n_folds} folds (tune on all earlier data, test on next unseen slice)")
    for k in range(1, n_folds + 1):
        best, best_p = -1e18, base
        for combo in itertools.product(*GRID.values()):
            p = replace(base, **dict(zip(GRID.keys(), combo)))
            tr = simulate(prep, cfg, p, tf_ms, bounds[0], bounds[k])
            o = objective(tr)
            if o > best:
                best, best_p = o, p
        te = simulate(prep, cfg, best_p, tf_ms, bounds[k], bounds[k + 1])
        tuned = {g: getattr(best_p, g) for g in GRID}
        print(fmt(te, f"fold {k} OOS") + f"\n{'':22} tuned on train: {tuned}")
        oos_trades += [t["pnl"] for t in te["trades_list"]]
        rets.append(te["return_pct"])
    s = trade_stats(oos_trades)
    print(f"\nOUT-OF-SAMPLE TOTAL: trades {s['trades']} | win {s['win_rate_pct']:.1f}% | "
          f"PF {s['profit_factor']:.2f} | mean fold return {np.mean(rets):+.2f}%")


# ---------------------------------------------------------------------- main
def load_csv(spec: str) -> (str, pd.DataFrame):
    path, _, sym = spec.partition(":")
    df = pd.read_csv(path)
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        df["timestamp"] = pd.to_datetime(ts, unit="ms" if ts.abs().median() > 1e11 else "s", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (sym or path.rsplit("/", 1)[-1].split(".")[0].upper()), df.dropna().reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="synthetic data (offline)")
    ap.add_argument("--candles", type=int, default=12000, help="synthetic candles per symbol (--demo)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--exchange-id", default=None)
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    ap.add_argument("--csv", action="append", default=[], help="path[:SYMBOL] (repeatable)")
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--since"), ap.add_argument("--until")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--walk-forward", type=int, default=0)
    ap.add_argument("--out", default="backtest")
    a = ap.parse_args()

    cfg = Config.from_env().validate()
    tf = a.timeframe or cfg.timeframe
    tf_ms = _parse_tf(tf)
    cfg = cfg.replace(timeframe=tf)
    frames: Dict[str, pd.DataFrame] = {}
    if a.demo:
        for i, (s, p) in enumerate(DEFAULT_SYMBOLS.items()):
            frames[s] = generate_history(p, a.candles, tf, seed=100 + i)
        print("NOTE: synthetic data - proves the plumbing, says NOTHING about real-market profitability.\n")
    elif a.download:
        from bot.market_data import ExchangeData
        if not (a.since and a.until):
            ap.error("--download needs --since and --until (UTC dates)")
        md = ExchangeData(a.exchange_id or cfg.exchange_id, cfg.quote_asset)
        s0 = int(pd.Timestamp(a.since, tz="UTC").timestamp() * 1000)
        s1 = int(pd.Timestamp(a.until, tz="UTC").timestamp() * 1000)
        for s in a.symbols.split(","):
            frames[s.strip()] = md.fetch_history(s.strip(), tf, s0, s1)
            print(f"downloaded {len(frames[s.strip()])} candles for {s.strip()}")
    elif a.csv:
        for spec in a.csv:
            s, df = load_csv(spec)
            frames[s] = df
    else:
        ap.error("choose --demo, --download or --csv")
    frames = {s: d for s, d in frames.items() if len(d) >= 200}
    if not frames:
        print("No usable data (need >= 200 candles per symbol).")
        return 1

    params = StrategyParams.from_config(cfg)
    prep = prepare(frames)
    full = simulate(prep, cfg, params, tf_ms)
    print(fmt(full, "FULL PERIOD"))
    if full["trades"] < 30:
        print("  ! fewer than 30 trades: statistically meaningless, collect more data.")
    if a.folds > 1:
        b = split_bounds(full["timeline"], a.folds)
        print(f"\nCONSECUTIVE FOLDS ({a.folds}) - consistency check, fixed parameters")
        for k in range(a.folds):
            print(fmt(simulate(prep, cfg, params, tf_ms, b[k], b[k + 1]), f"fold {k + 1}"))
    if a.walk_forward > 0:
        walk_forward(prep, cfg, params, tf_ms, a.walk_forward)

    pd.DataFrame(full["trades_list"]).to_csv(f"{a.out}_trades.csv", index=False)
    pd.DataFrame({"timestamp": pd.to_datetime(full["timeline"], unit="ms", utc=True),
                  "equity": full["equity_curve"]}).to_csv(f"{a.out}_equity.csv", index=False)
    print(f"\nWrote {a.out}_trades.csv and {a.out}_equity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
