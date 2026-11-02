# Agentic Crypto Bot (paper-trading)

An autonomous intraday crypto trading bot that runs **24/7 with no human in the loop**, built from the
`crypto-intraday-alert-engine` and `agentic-paper-trader` projects.

> **Paper trading only.** Fills are simulated (fees + slippage modelled). There is no exchange API key
> support and no order-placement code anywhere in this package — a test (`test_no_live_trading_or_credential_code_in_package`)
> fails if that ever changes. No strategy is guaranteed profitable; results on synthetic data prove the
> plumbing works, not that the strategy makes money.

## What it does

```
            ┌────────────────────────── every SCAN_INTERVAL seconds ──────────────────────────┐
 exchange   │                                                                                  │
 (ccxt,     ▼                                                                                  │
 public) ─► Scanner ─► Strategy Analyst ─► rank ─► Risk pre-check ─► LLM Review ─► Risk sizing ─► Paper Broker
            liquid     EMA/RSI/VWAP/vol/    by      (cheap veto,     approve /      (final        fees, slippage,
            USDT       ATR stops, cost      score   before paying    reduce /       authority,    gap-aware exits,
            pairs      gate (deterministic)         for an LLM call) reject only    can only       time stop
                                                                     (optional)     shrink)            │
                                                                                                       ▼
        Telegram ◄── alerts ◄── Journal (SQLite, WAL): trades, equity, marks, agent decisions, risk state
        Streamlit dashboard ◄─────────────┘     Post-trade Reviewer ─► reports/review-*.md (never auto-applied)
```

| Agent | Type | Authority |
|---|---|---|
| Market Scanner | deterministic | picks liquid USDT spot pairs (no stables / leveraged tokens) |
| Strategy Analyst | deterministic | proposes a long signal on the last **closed** candle |
| **LLM Scout** (optional) | LLM | **nominates** extra long candidates only; code sets stops/size and enforces every limit |
| **LLM Review Agent** | LLM, optional | **approve / reduce / reject only.** Cannot create trades, enlarge size, move stops, or touch limits |
| Risk Supervisor + Guard | deterministic | **final say.** Sizing, exposure cap, daily-loss lock, drawdown halt, loss-streak pause, cooldowns, kill switch |
| Paper Broker | deterministic | simulated fills, stops, targets, time stop |
| Post-trade Reviewer | deterministic (+ LLM narrative) | writes a report; changes nothing |

**Design rule: the LLM proposes vetoes, deterministic code decides.** If the LLM is down, rate-limited,
refuses, or returns malformed output, the trade is skipped (`LLM_FAIL_MODE=reject`, the default).
Multipliers above 1 are clamped to 1 in three separate places (schema normalizer, review agent, risk sizer).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                   # edit if you want Telegram / LLM

python -m pytest                                       # 80+ tests, ~80 s
python run_bot.py --demo                               # offline simulated market, ~8 s, no keys/network
DATABASE_PATH=demo.db streamlit run dashboard.py       # look at the demo results

python run_bot.py                                      # LIVE public market data, paper fills
streamlit run dashboard.py                             # in a second terminal
```

If Binance is blocked in your region (HTTP 451/403), set `EXCHANGE_ID=binanceus` (or `bybit`, `okx`,
`kraken`, `kucoin`, ...) in `.env`. No API key is needed — only public market data is used.

### Telegram alerts (optional)
1. **Revoke the old bot token** in @BotFather (`/revoke`) — it was hardcoded in your old repo.
2. Create/refresh the token, then put `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` (never in code).
3. You get: start/stop, entries, exits, halts, repeated-failure alerts, and a daily summary.

### Using the LLM review agent
It is on by default in `.env.example` (`LLM_ENABLED=true`) but needs an API key, otherwise the bot runs
rule-based only and says so at startup and on the dashboard ("LLM review: OFF (reason)").

1. Create an API key at https://console.anthropic.com -> API keys, and make sure the account has credits
   (a chat subscription is separate from API credits).
2. `notepad .env` and set `LLM_ENABLED=true` and `ANTHROPIC_API_KEY=...` (never commit `.env`; it is git-ignored).
3. **Verify it:** `python manage.py llmtest` makes one real review call on live candles and prints the verdict,
   or a specific hint if the key / credits / model name is wrong.
4. `python run_bot.py` — the first log lines show `LLM review: ON model=...`.

```
LLM_ENABLED=true
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-opus-5-5        # or claude-sonnet-5-5 / claude-haiku-5-5 for lower cost
```
Role: for every candidate that already passed the strategy **and** the risk pre-check, the LLM gets ~2-4 KB of
numbers (indicators, last 8 candles, benchmark regime, portfolio stats) and returns schema-constrained JSON:
**approve / reduce / reject** plus a rationale. It cannot originate trades, enlarge size or change stops/limits.
Calls are capped at `LLM_MAX_CALLS_PER_HOUR`; a signal is only reviewed when it is already tradeable, so cost is
roughly cents per day. If the LLM fails 3 times in a row you get a Telegram alert, and (with the default
`LLM_FAIL_MODE=reject`) trades are skipped rather than taken unreviewed. Every verdict is stored in the
`decisions` table and shown in the dashboard, so you can measure whether it adds value over the plain rules.

### LLM scout: finding setups the rules miss
With `LLM_SCOUT_ENABLED=true` (and a working API key) the LLM also **nominates extra long candidates**. Every
`LLM_SCOUT_INTERVAL_MINUTES` (15 by default), after the rules have had first pick, it receives a compact table for
the whole universe (price changes, RSI, ATR %, volume ratio, distance to VWAP/EMAs/20-candle high-low, 24 h volume,
and *why the rules rejected each symbol*) and may return up to 3 candidates with a setup type, confidence and thesis.
Returning nothing is normal and expected.

What the LLM can and cannot do:

| LLM decides | Code decides (the LLM cannot touch these) |
|---|---|
| which symbols look interesting, setup type, confidence, thesis | symbol must be in the live universe, not held, not in cooldown |
| | stop and target (ATR geometry) and the fee/slippage cost gate |
| | position size: normal risk sizing x `LLM_SCOUT_SIZE_FACTOR` (0.5) |
| | min confidence (0.65), max `LLM_SCOUT_MAX_TRADES_PER_DAY` (4), max `LLM_SCOUT_MAX_CONCURRENT` (1) |
| | all global limits: exposure cap, daily loss, drawdown halt, loss-streak pause, kill switch |

Scout trades are tagged `llm_scout` in the journal; the dashboard has a **Rules vs LLM scout** table and
`manage.py review` reports both separately. Try it first without trading: `python manage.py llmtest scout`.

**Cost (rough, check your Console usage).** Each scout call is ~3-4 K input tokens plus reasoning/output. Estimated
at current list prices: ~$0.03-0.05/call on `claude-opus-5-5`, about half on `claude-sonnet-5-5`, a few cents a day
on `claude-haiku-5-5`. At the 15-minute default that is up to ~$3-5/day on Opus if it runs around the clock, so
consider `LLM_SCOUT_MODEL=claude-sonnet-5-5` or a 30-60 minute interval. The scout makes no call when it has no
budget left (daily cap reached, a scout position already open, positions full, bot halted).

**How to judge it - and why it cannot be backtested.** A language model may have seen historical prices in
training, so replaying history through it would leak the future and flatter the result. The only valid test is
forward paper trading: let it run for weeks, then compare the `llm_scout` row against `rules` after **50+ scout
trades** of its own. If it does not beat the plain rules after costs (including its API bill), switch it off with
`LLM_SCOUT_ENABLED=false`. It is an experiment, not a promise of edge.

## Operating it

| Task | Command |
|---|---|
| Status / heartbeat / halt state | `python manage.py status` |
| Block new entries (keeps managing exits) | `python manage.py halt "reason"` or `python manage.py killswitch` |
| Flatten everything (paper) and block | `python manage.py killswitch close` |
| Resume after a halt | `python manage.py killswitch off` then `python manage.py resume` |
| Write a performance review | `python manage.py review` |
| Test the LLM key/model with one real call | `python manage.py llmtest` |
| See what the scout would nominate (no trade) | `python manage.py llmtest scout` |
| Run one cycle and exit | `python run_bot.py --once` |

The bot survives restarts: cash and open positions are rebuilt from the trade ledger, and the daily-loss
reference, peak equity, halt flag, loss streak and cooldowns are persisted. Stops are replayed over every
candle missed during downtime.

**Keep it running 24/7** (Linux example, `/etc/systemd/system/cryptobot.service`):
```
[Service]
WorkingDirectory=/opt/crypto-agentic-bot
ExecStart=/opt/crypto-agentic-bot/.venv/bin/python run_bot.py
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
```
Make sure the host clock is NTP-synced (the bot warns at startup if it differs from the exchange by >5 s).

## Risk controls (all enforced in code, all configurable in `.env`)

| Control | Default |
|---|---|
| Risk per trade (equity × stop distance) | 0.5 % |
| Max position size / total exposure | 25 % / 60 % of equity |
| Max open positions | 3 |
| Min order notional | 10 USDT (and exchange minimum when known) |
| Daily loss lock (new entries) | −2 % from UTC-day start |
| **Max drawdown halt** (persistent, needs human `resume`) | −10 % from peak |
| Loss-streak pause | 3 losses → 60 min |
| Per-symbol cooldown after an exit | 6 candles |
| Time stop | 48 candles |
| Stale-data guard | signal candle must be ≤ 2 candles old |
| Cost gate | skip if round-trip fees+slippage > 35 % of the target, or net R:R < 1.1 |

## Backtesting & validation

```bash
python backtest.py --download --symbols BTC/USDT,ETH/USDT,SOL/USDT --since 2026-01-01 --until 2026-10-01 --folds 4
python backtest.py --download --symbols BTC/USDT,ETH/USDT,SOL/USDT --since 2025-06-01 --until 2026-10-01 --walk-forward 3
python backtest.py --csv btc.csv:BTC/USDT --csv eth.csv:ETH/USDT
python backtest.py --demo --folds 3          # offline plumbing check on synthetic data
```
It reuses the **live** strategy, risk and broker code (portfolio-level, multi-symbol, next-candle-open
entries, stop-first, gap-aware, same exposure/daily/drawdown rules), prints buy-and-hold for comparison, and
`--walk-forward N` tunes a small parameter grid on past data and reports only the **next unseen slice**.

**Suggested path before you trust it with anything:**
1. Backtest ≥ 6–12 months on your target pairs with `--folds 4`; the result must be positive net of fees in most folds, not just overall.
2. Run `walk-forward`; out-of-sample profit factor should stay > 1.2 or so.
3. Paper-trade live for **4–8+ weeks / 100+ trades**; compare to the backtest. Use the dashboard + `manage.py review`.
4. Only then think about a live adapter (see below).

## Going live — deliberately not included
Real-money execution needs, at minimum: exchange order/fill reconciliation, partial fills, precision &
min-notional handling, idempotent order IDs, exchange-side stop orders, outage handling, a least-privilege
API key (**trade-only, withdrawals disabled, IP-restricted**), and weeks of exchange-testnet runs. Ask for
that as a separate, explicit step; start with an amount you can afford to lose entirely.

## What changed vs. the two original projects

| Problem found in the originals | Fix |
|---|---|
| Telegram bot token + chat id hardcoded in source | Env-only; test fails if a token pattern appears in code |
| Alert engine: BEARISH trades never exited | Spot long-only; single, tested exit path |
| Dashboard read a `trades` table nothing wrote (always zeros) | One journal; dashboard reads real data |
| `session_breakout` could never fire (compared close to a max that included itself) | Removed; replaced by an ATR/volume-based design |
| WebSocket URL had no stream subscription (likely received no data) | Replaced by closed-candle polling via ccxt (simple, robust, exchange-agnostic) |
| Fees/slippage ignored (a "1.8 RR" was really ~1:1) | Fees+slippage everywhere + cost gate |
| 6 settings never read (`MAX_OPEN_TRADES`, `TOTAL_CAPITAL`, ...) | Every setting is enforced and validated |
| Paper trader re-checked the pre-entry candle → bogus instant stop-outs | Exits use only candles **after** the entry candle (regression test) |
| Paper trader checked only the latest candle → stops skipped after downtime | All missed candles replayed in order (regression test) |
| Backtest O(n²), single symbol | O(n), portfolio-level, multi-symbol, folds + walk-forward |
| Print-only "tests" | 80+ real tests incl. end-to-end bot runs, safety invariants, LLM fail-closed |

## Known limitations (read these)
- **No proven edge.** The EMA/RSI/VWAP/volume hypothesis is a starting point; random-walk backtests give ≈0 after costs by design. Validate on real history before believing anything.
- Paper fills assume you can trade at the ticker/open ± slippage; real order books, latency and partial fills are worse, especially for small-cap alts.
- Intra-candle stop checks use the polled ticker (every `SCAN_INTERVAL`), so a fast wick between polls is only caught at the next candle's low.
- Spot long-only: it cannot profit from falling markets, it can only stand aside.
- The LLM sees no news/sentiment feed (by design: no scraped text → no prompt-injection surface). It cannot see anything the numbers don't show.
- `ExchangeData` is unit-tested against a mocked ccxt exchange and the LLM client against a stub client. Run `python run_bot.py --once` against your real exchange (and one real LLM review) before relying on it.
- Not tax or financial advice.

## Layout
```
run_bot.py  manage.py  backtest.py  dashboard.py  .env.example  requirements.txt
bot/  config  indicators  strategy  market_data  synthetic  journal  broker  risk
      agents  llm  runner  notifier  reviewer  stats
tests/
```
