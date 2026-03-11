"""
StockDataCache — cache-first data access layer for historical price data.

Sits between BobsBrain and both the database and Alpaca, implementing a
read-through cache:
  1. Query the DB for the requested (symbols, date range).
  2. Identify symbols or date ranges missing from the DB result.
  3. Fetch the gaps from Alpaca.
  4. Write the gap data back to the DB.
  5. Return a unified DataFrame.

Neither BobsBrain nor any other caller needs to know whether data came from
the cache or the API — the interface is always the same DataFrame.

Key performance benefit: the O(N²) pair scan in before_market_opens() shifts
from a bulk Alpaca API call (network-bound, ~seconds) to a single DB query
(~milliseconds for 60 rows × 100 tickers locally).
"""

from datetime import datetime, timedelta

import pandas as pd

from AlpacaClient import AlpacaClient
from DatabaseClient import DatabaseClient


class StockDataCache:
    """
    Cache-first data access layer for price history.

    Instantiate once with a DatabaseClient and an AlpacaClient, then pass
    to BobsBrain. warm_cache() is intended for a nightly cron job so the
    market-open scan always finds a fully warm cache.
    """

    def __init__(self, db: DatabaseClient, alpaca: AlpacaClient):
        self._db = db
        self._alpaca = alpaca

    def get_prices(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Return daily close prices as a DataFrame with a DatetimeIndex and
        symbol columns.

        Data is read from the DB when available; only missing symbols or
        date ranges are fetched from Alpaca and written back.

        Returns an empty DataFrame if no data is available for any symbol.
        """
        if not symbols:
            return pd.DataFrame()

        cached = self._db.get_prices(symbols, start, end)
        missing_symbols = self._find_missing_symbols(symbols, cached, start, end)

        if missing_symbols:
            fresh = self._alpaca.get_historical_bars(missing_symbols, start, end)
            if not fresh.empty:
                self._db.upsert_prices(fresh)
                cached = _merge(cached, fresh)

        return cached

    def warm_cache(self, symbols: list[str], days: int = 60) -> None:
        """
        Pre-load the last N days of price data for a list of symbols.

        Intended to be called from a nightly cron job so before_market_opens()
        always finds a warm cache and never needs to call Alpaca at runtime.
        Fetches full OHLCV records (not just close) via get_ohlcv_records().
        """
        if not symbols:
            return

        end = datetime.now()
        start = end - timedelta(days=days)

        records = self._alpaca.get_ohlcv_records(symbols, start, end)
        if records:
            self._db.upsert_ohlcv(records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_missing_symbols(
        self,
        requested: list[str],
        cached: pd.DataFrame,
        start: datetime,
        end: datetime,
    ) -> list[str]:
        """
        Return symbols that are either absent from the cached DataFrame or
        whose cached data does not cover the requested date range.
        """
        if cached.empty:
            return list(requested)

        missing = []
        for symbol in requested:
            if symbol not in cached.columns:
                missing.append(symbol)
                continue

            series = cached[symbol].dropna()
            if series.empty:
                missing.append(symbol)
                continue

            # Check date coverage — treat a gap of >1 trading day as a miss.
            # This deliberately errs on the side of re-fetching to keep data fresh.
            earliest = pd.Timestamp(series.index.min())
            latest = pd.Timestamp(series.index.max())
            start_ts = pd.Timestamp(start).normalize()
            end_ts = pd.Timestamp(end).normalize()

            if earliest > start_ts + timedelta(days=2) or latest < end_ts - timedelta(days=2):
                missing.append(symbol)

        return missing


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _merge(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine two close-price DataFrames, preferring fresh data on overlap."""
    if cached.empty:
        return fresh
    if fresh.empty:
        return cached

    combined = pd.concat([cached, fresh])
    # Keep the last value for any (date, symbol) duplicates (fresh wins).
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()
