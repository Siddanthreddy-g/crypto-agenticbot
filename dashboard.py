"""Streamlit dashboard (read-only view of the bot's SQLite journal).

    streamlit run dashboard.py
    DATABASE_PATH=demo.db streamlit run dashboard.py
"""
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from bot.journal import Journal
from bot.stats import max_drawdown_pct, trade_stats

st.set_page_config(page_title="Agentic Crypto Bot (PAPER)", layout="wide", page_icon="📈")

default_db = os.environ.get("DATABASE_PATH", "trader.db")
with st.sidebar:
    st.header("Settings")
    db_path = st.text_input("Database path", default_db)
    refresh = st.slider("Auto-refresh (seconds, 0 = off)", 0, 120, 15)
    st.caption("Read-only view. Control the bot with `python manage.py ...`")

st.title("📈 Agentic Crypto Bot")
st.warning("PAPER TRADING ONLY - simulated fills, no real orders. Past/simulated results do not predict future returns.", icon="⚠️")

if not os.path.exists(db_path):
    st.info(f"No database found at `{db_path}`. Start the bot (`python run_bot.py` or `python run_bot.py --demo`) first.")
    st.stop()


def render(path: str) -> None:
    j = Journal(path, readonly=True)
    try:
        kv = j.kv_all()
        port = j.load_portfolio()
        marks = j.get_marks()
        closed = j.get_closed_trades()
        opens = j.get_open_trades()
        curve = j.equity_curve(20000)
        decisions = j.recent_decisions(300)
    except Exception as exc:
        st.error(f"Could not read journal: {exc}")
        return
    finally:
        j.close()

    # ---- status
    wall = kv.get("last_cycle_wall")
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(wall)).total_seconds() if wall else None
    halted = kv.get("halted", "0") == "1"
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Bot heartbeat", "never" if age is None else f"{age:.0f}s ago",
              delta=None if age is None else ("running" if age < 180 else "STALE / stopped"),
              delta_color="normal" if (age is not None and age < 180) else "inverse")
    s2.metric("Trading state", "HALTED" if halted else "ACTIVE")
    s3.metric("Mode", kv.get("mode", "PAPER"))
    s4.metric("LLM review", kv.get("llm_status", "unknown").split(" (")[0],
              help=kv.get("llm_status", "") + " - the LLM can only approve, shrink or reject rule-based signals")
    st.caption(f"Watching {len([x for x in kv.get('universe', '').split(',') if x])} symbols  |  LLM: {kv.get('llm_status', 'unknown')}")
    if halted:
        st.error(f"Halt reason: {kv.get('halt_reason', '')}  -  resume with `python manage.py resume`")

    # ---- KPIs
    unreal = sum((marks.get(t["symbol"], t["entry_price"]) - t["entry_price"]) * t["qty"] for t in opens)
    mtm = sum(marks.get(t["symbol"], t["entry_price"]) * t["qty"] for t in opens)
    equity = port["cash"] + mtm
    st_ = trade_stats([t["pnl"] or 0 for t in closed])
    mdd = max_drawdown_pct([r["equity"] for r in curve]) if curve else 0.0
    k = st.columns(6)
    k[0].metric("Equity (USDT)", f"{equity:,.2f}", f"{(equity / port['initial_cash'] - 1) * 100:+.2f}%")
    k[1].metric("Realized PnL", f"{port['realized_pnl']:+.2f}")
    k[2].metric("Unrealized PnL", f"{unreal:+.2f}")
    k[3].metric("Win rate", f"{st_['win_rate_pct']:.1f}%", f"{st_['wins']}W / {st_['losses']}L", delta_color="off")
    pf = st_["profit_factor"]
    k[4].metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
    k[5].metric("Max drawdown", f"{mdd:.2f}%")
    if st_["trades"] < 30:
        st.caption(f"Only {st_['trades']} closed trades - statistics are not yet meaningful (aim for 100+).")

    # ---- equity curve
    st.subheader("Equity curve")
    if curve:
        df = pd.DataFrame(curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        st.line_chart(df.set_index("timestamp")[["equity"]], height=280)
    else:
        st.info("No equity snapshots yet.")

    # ---- positions
    st.subheader("Open positions")
    if opens:
        rows = []
        for t in opens:
            px = marks.get(t["symbol"], t["entry_price"])
            rows.append({"symbol": t["symbol"], "qty": t["qty"], "entry": t["entry_price"], "price": px,
                         "stop": t["stop_price"], "target": t["target_price"],
                         "unrealized": (px - t["entry_price"]) * t["qty"],
                         "unrealized_%": (px / t["entry_price"] - 1) * 100, "opened": t["entry_time"]})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No open positions.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Closed trades (latest 100)")
        if closed:
            cdf = pd.DataFrame(closed)[["id", "symbol", "entry_time", "exit_time", "entry_price", "exit_price",
                                        "qty", "pnl", "reason"]].tail(100).iloc[::-1]
            st.dataframe(cdf, width="stretch", hide_index=True)
        else:
            st.info("No closed trades yet.")
    with right:
        st.subheader("PnL by exit reason")
        if closed:
            g = pd.DataFrame(closed).groupby("reason")["pnl"].agg(["count", "sum", "mean"]).round(4)
            st.bar_chart(g["sum"])
            st.dataframe(g, width="stretch")

    # ---- agent log
    st.subheader("Agent decision log")
    if decisions:
        ddf = pd.DataFrame(decisions)[["ts", "symbol", "agent", "decision", "detail"]]
        agents = ["all"] + sorted(ddf["agent"].unique().tolist())
        pick = st.selectbox("Filter by agent", agents)
        st.dataframe(ddf if pick == "all" else ddf[ddf["agent"] == pick], width="stretch", hide_index=True)
    else:
        st.info("No agent decisions logged yet.")


if refresh > 0:
    st.fragment(run_every=refresh)(render)(db_path)
else:
    render(db_path)
