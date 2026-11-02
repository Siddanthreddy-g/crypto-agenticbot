#!/usr/bin/env python3
"""Start the paper-trading agentic bot.

    python run_bot.py              # live public market data, paper fills
    python run_bot.py --demo       # offline synthetic market (no network, no API keys)
    python run_bot.py --once       # run a single cycle and exit
"""
import argparse
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler

from bot.config import Config
from bot.journal import Journal
from bot.runner import Bot


def setup_logging(level: str) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in (logging.StreamHandler(), RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=3,
                                                           encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="offline synthetic market, accelerated")
    ap.add_argument("--demo-cycles", type=int, default=600, help="candles to simulate in --demo (default 600)")
    ap.add_argument("--demo-delay", type=float, default=0.0, help="seconds between simulated candles")
    ap.add_argument("--once", action="store_true", help="run one cycle and exit")
    ap.add_argument("--db", help="override DATABASE_PATH")
    args = ap.parse_args()

    cfg = Config.from_env()
    if args.demo:
        cfg = cfg.replace(data_source="synthetic",
                          database_path=args.db or "demo.db", scan_interval=5)
    elif args.db:
        cfg = cfg.replace(database_path=args.db)
    cfg.validate()
    setup_logging(cfg.log_level)
    log = logging.getLogger("run_bot")
    log.warning("PAPER MODE ONLY: no authenticated exchange client or live-order path exists.")

    if cfg.data_source == "synthetic":
        from bot.synthetic import SyntheticData
        data = SyntheticData(timeframe=cfg.timeframe)
        log.info("DEMO: synthetic market, results are meaningless for strategy validation")
    else:
        from bot.market_data import ExchangeData
        try:
            data = ExchangeData(cfg.exchange_id, cfg.quote_asset)
        except Exception as exc:
            log.error("Could not reach exchange '%s': %s\n"
                      "If it is geo-blocked in your region, set EXCHANGE_ID to another ccxt spot exchange "
                      "(binanceus, bybit, okx, kraken, ...) or try `python run_bot.py --demo`.",
                      cfg.exchange_id, exc)
            return 2

    bot = Bot(cfg, data, Journal(cfg.database_path))
    signal.signal(signal.SIGINT, lambda *_: bot.stop())
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())

    if args.once:
        print(bot.run_cycle())
        return 0
    if args.demo:
        left = {"n": args.demo_cycles}

        def advance() -> bool:
            left["n"] -= 1
            if args.demo_delay:
                import time
                time.sleep(args.demo_delay)
            return data.advance() and left["n"] > 0

        bot.run_forever(on_cycle=advance, sleep=False)
        stats = bot.perf.summary()
        print("\nDEMO FINISHED:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()})
        print(f"Journal: {cfg.database_path}   (view with: DATABASE_PATH={cfg.database_path} streamlit run dashboard.py)")
    else:
        bot.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
