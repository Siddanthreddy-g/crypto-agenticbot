"""SQLite persistence: trades ledger, equity curve, marks, kv state, agent decisions.

* One shared connection guarded by a lock (works for ':memory:' in backtests).
* WAL mode so the Streamlit dashboard can read while the bot writes.
* Cash / open positions are *reconstructed from the trade ledger* on restart.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    entry_fee REAL NOT NULL,
    exit_fee REAL,
    pnl REAL,
    status TEXT NOT NULL,
    reason TEXT,
    entry_candle_ts INTEGER NOT NULL DEFAULT 0,
    signal_reason TEXT,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cash REAL NOT NULL,
    equity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS portfolio_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    initial_cash REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS marks (symbol TEXT PRIMARY KEY, price REAL, updated_at TEXT);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT,
    agent TEXT NOT NULL,
    decision TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
"""


class Journal:
    def __init__(self, path: str, readonly: bool = False):
        self.path = path
        self.readonly = readonly
        self._lock = threading.RLock()
        if readonly:
            self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        else:
            self.con = sqlite3.connect(path, check_same_thread=False)
            if path != ":memory:":
                self.con.execute("PRAGMA journal_mode=WAL")
                self.con.execute("PRAGMA synchronous=NORMAL")
            self.con.executescript(SCHEMA)
            self.con.commit()
        self.con.execute("PRAGMA busy_timeout=5000")
        self.con.row_factory = sqlite3.Row

    # ------------------------------------------------------------- utilities
    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self.con.execute(sql, params)
            self.con.commit()
            return cur

    def _all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.con.execute(sql, params).fetchall()]

    def close(self):
        with self._lock:
            self.con.close()

    # -------------------------------------------------------------------- kv
    def kv_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self.con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: Any) -> None:
        self._exec("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, str(value)))

    def kv_float(self, key: str, default: float = 0.0) -> float:
        v = self.kv_get(key)
        try:
            return float(v) if v is not None else default
        except ValueError:
            return default

    def kv_all(self) -> Dict[str, str]:
        return {r["key"]: r["value"] for r in self._all("SELECT key, value FROM kv")}

    # ------------------------------------------------------------- portfolio
    def initialize_portfolio(self, initial_cash: float) -> None:
        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive.")
        self._exec("INSERT OR IGNORE INTO portfolio_state (id, initial_cash) VALUES (1, ?)",
                   (float(initial_cash),))

    def load_portfolio(self) -> Dict[str, Any]:
        with self._lock:
            state = self.con.execute("SELECT initial_cash FROM portfolio_state WHERE id=1").fetchone()
            if state is None:
                raise RuntimeError("Portfolio not initialized.")
            rows = [dict(r) for r in self.con.execute("SELECT * FROM trades ORDER BY id").fetchall()]
        initial = float(state["initial_cash"])
        cash, realized, closed, wins, losses = initial, 0.0, 0, 0, 0
        positions: Dict[str, Dict[str, Any]] = {}
        for t in rows:
            qty, entry, efee = float(t["qty"]), float(t["entry_price"]), float(t["entry_fee"])
            cash -= qty * entry + efee
            if t["status"] == "OPEN":
                if t["symbol"] in positions:
                    raise RuntimeError(f"Multiple open trades found for {t['symbol']}.")
                positions[t["symbol"]] = t
            elif t["status"] == "CLOSED":
                if t["exit_price"] is None:
                    raise RuntimeError(f"Closed trade {t['id']} has no exit price.")
                cash += qty * float(t["exit_price"]) - float(t["exit_fee"] or 0.0)
                pnl = float(t["pnl"] or 0.0)
                realized += pnl
                closed += 1
                wins += pnl > 0
                losses += pnl <= 0
            else:
                raise RuntimeError(f"Unknown trade status: {t['status']}")
        if cash < -1e-6:
            raise RuntimeError(f"Reconstructed cash is negative: {cash:.8f}")
        return {"initial_cash": initial, "cash": max(0.0, cash), "positions": positions,
                "realized_pnl": realized, "closed_trades": closed, "wins": int(wins),
                "losses": int(losses)}

    # ---------------------------------------------------------------- trades
    def add_trade(self, v: Dict[str, Any]) -> int:
        cur = self._exec(
            """INSERT INTO trades (symbol, entry_time, qty, entry_price, stop_price, target_price,
               entry_fee, status, reason, entry_candle_ts, signal_reason, meta)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', NULL, ?, ?, ?)""",
            (v["symbol"], v["entry_time"], v["qty"], v["entry_price"], v["stop_price"],
             v["target_price"], v["entry_fee"], int(v.get("entry_candle_ts", 0)),
             v.get("signal_reason"), json.dumps(v.get("meta") or {}, default=str)))
        return int(cur.lastrowid)

    def close_trade(self, trade_id: int, exit_time: str, exit_price: float, exit_fee: float,
                    pnl: float, reason: str) -> None:
        cur = self._exec(
            """UPDATE trades SET exit_time=?, exit_price=?, exit_fee=?, pnl=?, status='CLOSED', reason=?
               WHERE id=? AND status='OPEN'""",
            (exit_time, exit_price, exit_fee, pnl, reason, trade_id))
        if cur.rowcount != 1:
            raise ValueError(f"Trade {trade_id} was not found as an open trade.")

    def get_open_trades(self) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM trades WHERE status='OPEN' ORDER BY id")

    def get_closed_trades(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit:
            rows = self._all("SELECT * FROM trades WHERE status='CLOSED' ORDER BY id DESC LIMIT ?", (limit,))
            return rows[::-1]
        return self._all("SELECT * FROM trades WHERE status='CLOSED' ORDER BY id")

    # ---------------------------------------------------------- equity/marks
    def snapshot(self, timestamp: str, cash: float, equity: float) -> None:
        self._exec("INSERT INTO equity_snapshots (timestamp, cash, equity) VALUES (?, ?, ?)",
                   (timestamp, cash, equity))

    def equity_curve(self, limit: int = 5000) -> List[Dict[str, Any]]:
        rows = self._all("SELECT timestamp, cash, equity FROM equity_snapshots ORDER BY id DESC LIMIT ?", (limit,))
        return rows[::-1]

    def set_marks(self, marks: Dict[str, float], ts: str) -> None:
        with self._lock:
            self.con.executemany(
                "INSERT OR REPLACE INTO marks (symbol, price, updated_at) VALUES (?, ?, ?)",
                [(s, float(p), ts) for s, p in marks.items()])
            self.con.commit()

    def get_marks(self) -> Dict[str, float]:
        return {r["symbol"]: r["price"] for r in self._all("SELECT symbol, price FROM marks")}

    # ------------------------------------------------------------- decisions
    def add_decision(self, ts: str, symbol: Optional[str], agent: str, decision: str,
                     detail: Any = None) -> None:
        d = detail if isinstance(detail, str) else json.dumps(detail, default=str)
        self._exec("INSERT INTO decisions (ts, symbol, agent, decision, detail) VALUES (?, ?, ?, ?, ?)",
                   (ts, symbol, agent, decision, d))

    def recent_decisions(self, limit: int = 200, agent: Optional[str] = None) -> List[Dict[str, Any]]:
        if agent:
            return self._all("SELECT * FROM decisions WHERE agent=? ORDER BY id DESC LIMIT ?", (agent, limit))
        return self._all("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))
