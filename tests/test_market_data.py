"""ExchangeData is exercised against a fake ccxt exchange (no network available in CI)."""
from types import SimpleNamespace

import pandas as pd

from bot.market_data import ExchangeData, _is_tradeable_base

TF = 300_000
NOW_MS = 10_000 * TF + 90_000                 # 90s into candle #10000


def fake_exchange(candles):
    def market(base, quote="USDT", spot=True, active=True):
        return {"base": base, "quote": quote, "spot": spot, "active": active}

    markets = {"BTC/USDT": market("BTC"), "ETH/USDT": market("ETH"), "USDC/USDT": market("USDC"),
               "BTCUP/USDT": market("BTCUP"), "SOL/USDT": market("SOL"), "DEAD/USDT": market("DEAD", active=False),
               "BTC/EUR": market("BTC", "EUR"), "PERP/USDT": market("PERP", spot=False), "TINY/USDT": market("TINY")}
    tickers = {s: {"last": 10.0, "quoteVolume": v} for s, v in
               {"BTC/USDT": 9e8, "ETH/USDT": 5e8, "USDC/USDT": 7e8, "BTCUP/USDT": 6e8, "SOL/USDT": 1e8,
                "DEAD/USDT": 9e9, "BTC/EUR": 9e9, "PERP/USDT": 9e9, "TINY/USDT": 1e3}.items()}
    return SimpleNamespace(
        markets=markets, fetch_tickers=lambda: tickers, parse_timeframe=lambda tf: TF / 1000,
        fetch_ohlcv=lambda sym, timeframe="5m", limit=None, since=None: candles(since, limit),
        load_markets=lambda: None, amount_to_precision=lambda s, q: f"{q:.3f}",
        fetch_time=lambda: NOW_MS)


def make(candles):
    d = ExchangeData.__new__(ExchangeData)
    import ccxt
    d.ccxt, d.exchange, d.quote, d.retries = ccxt, fake_exchange(candles), "USDT", 0
    d.now = lambda: __import__("datetime").datetime.fromtimestamp(NOW_MS / 1000, tz=__import__("datetime").timezone.utc)
    return d


def test_universe_excludes_stables_leveraged_inactive_nonspot_and_other_quotes():
    d = make(lambda s, l: [])
    assert d.liquid_symbols(5e6, 10) == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]      # ranked by volume
    assert d.liquid_symbols(5e6, 2) == ["BTC/USDT", "ETH/USDT"]
    assert d.liquid_symbols(5e6, 10, watchlist=["SOL/USDT", "NOPE/USDT"]) == ["SOL/USDT"]


def test_base_filter():
    assert not _is_tradeable_base("USDC") and not _is_tradeable_base("ETHDOWN") and _is_tradeable_base("SOL")
    assert _is_tradeable_base("UP")                           # a real coin named UP is not a leveraged token (len<=4)


def test_candles_drop_the_forming_candle():
    rows = [[(9998 + k) * TF, 1, 2, 0.5, 1.5, 10] for k in range(3)]       # 9998, 9999 closed ; 10000 still forming
    d = make(lambda since, limit: rows)
    df = d.candles("BTC/USDT", "5m", 50)
    assert len(df) == 2 and int(df["timestamp"].iloc[-1].timestamp() * 1000) == 9999 * TF
    assert pd.api.types.is_float_dtype(df["close"])


def test_candles_deduplicate_and_drop_bad_rows():
    rows = [[9990 * TF, 1, 2, 0.5, 1.5, 10], [9990 * TF, 1, 2, 0.5, 1.5, 10], [9991 * TF, None, 2, 0.5, 1.5, 10]]
    df = make(lambda s, l: rows).candles("BTC/USDT", "5m", 50)
    assert len(df) == 1


def test_fetch_history_paginates_without_duplicates_or_gaps():
    store = [[k * TF, 1, 2, 0.5, 1.5, 10] for k in range(0, 2500)]

    def page(since, limit):
        return [c for c in store if c[0] >= since][:limit or 1000]

    df = make(page).fetch_history("BTC/USDT", "5m", 100 * TF, 2300 * TF)
    ts = (df["timestamp"].astype("int64") // 10 ** 6 if df["timestamp"].dt.unit == "ns"
          else df["timestamp"].map(lambda t: int(t.timestamp() * 1000)))
    assert len(df) == 2200 and ts.is_monotonic_increasing and ts.nunique() == 2200


def test_round_qty_and_min_notional_fallbacks():
    d = make(lambda s, l: [])
    assert d.round_qty("BTC/USDT", 0.123456) == 0.123
    assert d.min_notional("BTC/USDT") == 0.0                  # missing limits -> 0, config floor applies


def test_all_documented_exchange_ids_exist_in_ccxt():
    import ccxt
    for ex in ("binance", "binanceus", "bybit", "okx", "kraken", "coinbase", "kucoin"):
        assert hasattr(ccxt, ex)
