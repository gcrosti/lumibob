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

import logging
import time
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

from AlpacaClient import AlpacaClient
from DatabaseClient import DatabaseClient

# A fetch batch at least this large that returns zero rows is treated as an
# infrastructure failure (raise) rather than per-symbol data gaps (mark).
_MASS_FAILURE_MIN_BATCH = 50


class StockDataCache:
    """
    Cache-first data access layer for price history.

    Instantiate once with a DatabaseClient and an AlpacaClient, then pass
    to BobsBrain. warm_cache() is intended for a nightly cron job so the
    market-open scan always finds a fully warm cache.
    """

    def __init__(
        self,
        db: DatabaseClient,
        alpaca: AlpacaClient,
        cache_only: bool = False,
    ):
        """
        cache_only=True disables all Alpaca access: symbols without DB rows
        for the requested window are simply absent from the result, and no
        failed_tickers rows are written. Used by tuning studies — the price
        cache is fully backfilled, so the only thing live fetches can add
        during a study is rate-limit exposure and failure-marking risk
        (2026-07-14 incident).
        """
        self._db = db
        self._alpaca = alpaca
        self._cache_only = cache_only
        # Ensure the table exists / columns are up-to-date (idempotent).
        db.migrate_failed_tickers()
        # Per-window failed-symbol cache: keyed by (window_start, window_end)
        # dates.  Populated lazily on first get_prices call for a given window
        # so that a symbol that failed a 2022 fetch is not incorrectly excluded
        # from a 2023 fetch.
        self._failed: dict[tuple, set[str]] = {}

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
        date ranges are fetched from Alpaca and written back.  Symbols that
        previously returned no data from Alpaca are recorded in the
        failed_tickers table and skipped on all subsequent calls, eliminating
        repeated API round-trips for delisted or invalid tickers.

        Returns an empty DataFrame if no data is available for any symbol.
        """
        if not symbols:
            return pd.DataFrame()

        cached = self._db.get_prices(symbols, start, end)

        if self._cache_only:
            return cached

        missing_symbols = self._find_missing_symbols(symbols, cached, start, end)

        # Filter out symbols already known to be unfetchable for this window.
        window_failed = self._get_failed_for_window(start, end)
        fetchable = [s for s in missing_symbols if s not in window_failed]

        if fetchable:
            fresh = self._fetch_with_retry(fetchable, start, end)
            if not fresh.empty:
                self._db.upsert_prices(fresh)
                cached = _merge(cached, fresh)

            # Any symbol we requested but didn't get back → mark failed for this window.
            returned = set(fresh.columns) if not fresh.empty else set()
            ws = start.date() if hasattr(start, 'date') else start
            we = end.date()   if hasattr(end,   'date') else end

            # Circuit breaker: a large batch returning NOTHING is an
            # infrastructure failure (rate limit, network, auth) — not
            # evidence about each individual symbol. Recording it as
            # thousands of per-symbol failures poisons every later run
            # that overlaps this window (2026-07-14: 4,607 symbols marked
            # in 13 s). Fail the run loudly instead.
            if len(fetchable) >= _MASS_FAILURE_MIN_BATCH and not returned:
                raise RuntimeError(
                    f'StockDataCache: Alpaca returned no data for ALL '
                    f'{len(fetchable)} requested symbols ({ws} → {we}) — '
                    f'treating as infrastructure failure, not marking '
                    f'symbols failed.'
                )

            for symbol in fetchable:
                if symbol not in returned:
                    window_failed.add(symbol)
                    self._db.mark_ticker_failed(symbol, "no data from Alpaca", ws, we)

        return cached

    def _fetch_with_retry(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        max_retries: int = 3,
        backoff_seconds: float = 15.0,
    ) -> pd.DataFrame:
        """
        Call AlpacaClient.get_historical_bars with exponential-backoff retry
        on transient network errors (timeouts, connection resets).

        Returns an empty DataFrame if all retries are exhausted.
        """
        for attempt in range(1, max_retries + 1):
            try:
                return self._alpaca.get_historical_bars(symbols, start, end)
            except Exception as exc:
                if attempt < max_retries:
                    wait = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        'StockDataCache: Alpaca fetch failed (attempt %d/%d): %s — '
                        'retrying in %.0fs',
                        attempt, max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        'StockDataCache: Alpaca fetch failed after %d attempts: %s — '
                        'returning empty DataFrame',
                        max_retries, exc,
                    )
        return pd.DataFrame()

    def warm_cache(self, symbols: list[str], days: int = 126) -> None:
        """
        Pre-load the last N days of price data for a list of symbols.

        Intended to be called from a nightly cron job so before_market_opens()
        always finds a warm cache and never needs to call Alpaca at runtime.
        Fetches full OHLCV records (not just close) via get_ohlcv_records().

        The default of 126 days (~6 trading months) covers TickerClusterer’s
        clustering window. Strategy scoring uses lookback_window (default 130
        calendar days in BobsBrain); use warm_cache(days=...) ≥ max(126, 130)
        if you want one cron job to cover both.
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

    def _get_failed_for_window(self, start: datetime, end: datetime) -> set[str]:
        """
        Return (and cache in-process) the set of symbols that previously failed
        for a window overlapping [start, end].  The set is mutable — callers
        add new failures to it directly so in-process deduplication is free.
        """
        ws = start.date() if hasattr(start, 'date') else start
        we = end.date()   if hasattr(end,   'date') else end
        key = (ws, we)
        if key not in self._failed:
            self._failed[key] = set(self._db.get_failed_tickers_for_window(ws, we))
        return self._failed[key]

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
            # Strip timezone from all four timestamps so tz-aware DB values
            # (UTC) compare cleanly with potentially tz-aware Lumibot datetimes
            # and with each other regardless of origin.
            earliest = _to_naive(pd.Timestamp(series.index.min()))
            latest = _to_naive(pd.Timestamp(series.index.max()))
            start_ts = _to_naive(pd.Timestamp(start))
            end_ts = _to_naive(pd.Timestamp(end))

            if earliest > start_ts + timedelta(days=2) or latest < end_ts - timedelta(days=2):
                missing.append(symbol)

        return missing


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a tz-naive, date-only Timestamp regardless of whether ts is tz-aware."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


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
