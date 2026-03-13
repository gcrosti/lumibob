"""
Unit tests for StockDataCache.

Both DatabaseClient and AlpacaClient are fully mocked — no live DB or API
calls required.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, call
import pandas as pd
import pytest

from StockDataCache import StockDataCache, _merge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

START = datetime(2025, 1, 1)
END = datetime(2025, 3, 1)


def _make_cache(
    db_prices: pd.DataFrame = None,
    alpaca_bars: pd.DataFrame = None,
) -> tuple[StockDataCache, MagicMock, MagicMock]:
    mock_db = MagicMock()
    mock_alpaca = MagicMock()

    mock_db.get_prices.return_value = db_prices if db_prices is not None else pd.DataFrame()
    mock_alpaca.get_historical_bars.return_value = alpaca_bars if alpaca_bars is not None else pd.DataFrame()

    cache = StockDataCache(mock_db, mock_alpaca)
    return cache, mock_db, mock_alpaca


def _close_df(symbols: list[str], dates: list[str], values: list[float] = None) -> pd.DataFrame:
    """Build a close-price DataFrame matching the shape get_prices() returns."""
    idx = pd.to_datetime(dates)
    data = {}
    for i, sym in enumerate(symbols):
        data[sym] = [values[i]] * len(dates) if values else [100.0 + i] * len(dates)
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# get_prices — cache hit
# ---------------------------------------------------------------------------

class TestGetPricesCacheHit:
    def test_returns_db_data_when_all_symbols_covered(self):
        dates = pd.date_range(START, END, freq="B").strftime("%Y-%m-%d").tolist()
        cached = _close_df(["AAPL", "MSFT"], dates)
        cache, mock_db, mock_alpaca = _make_cache(db_prices=cached)

        result = cache.get_prices(["AAPL", "MSFT"], START, END)

        mock_alpaca.get_historical_bars.assert_not_called()
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns

    def test_does_not_upsert_when_no_gaps(self):
        dates = pd.date_range(START, END, freq="B").strftime("%Y-%m-%d").tolist()
        cached = _close_df(["AAPL"], dates)
        cache, mock_db, mock_alpaca = _make_cache(db_prices=cached)

        cache.get_prices(["AAPL"], START, END)

        mock_db.upsert_prices.assert_not_called()

    def test_returns_empty_dataframe_for_empty_symbols(self):
        cache, _, _ = _make_cache()
        result = cache.get_prices([], START, END)
        assert result.empty


# ---------------------------------------------------------------------------
# get_prices — cache miss
# ---------------------------------------------------------------------------

class TestGetPricesCacheMiss:
    def test_fetches_from_alpaca_when_db_empty(self):
        fresh = _close_df(["AAPL"], ["2025-01-02"])
        cache, mock_db, mock_alpaca = _make_cache(
            db_prices=pd.DataFrame(), alpaca_bars=fresh
        )

        result = cache.get_prices(["AAPL"], START, END)

        mock_alpaca.get_historical_bars.assert_called_once_with(["AAPL"], START, END)
        mock_db.upsert_prices.assert_called_once()

    def test_fetches_only_missing_symbols(self):
        dates = pd.date_range(START, END, freq="B").strftime("%Y-%m-%d").tolist()
        cached = _close_df(["AAPL"], dates)
        fresh = _close_df(["MSFT"], ["2025-01-02"])
        cache, mock_db, mock_alpaca = _make_cache(db_prices=cached, alpaca_bars=fresh)

        cache.get_prices(["AAPL", "MSFT"], START, END)

        args = mock_alpaca.get_historical_bars.call_args[0]
        assert args[0] == ["MSFT"]

    def test_merges_cached_and_fresh_data(self):
        dates = pd.date_range(START, END, freq="B").strftime("%Y-%m-%d").tolist()
        cached = _close_df(["AAPL"], dates)
        fresh = _close_df(["MSFT"], ["2025-01-02"])
        cache, _, _ = _make_cache(db_prices=cached, alpaca_bars=fresh)

        result = cache.get_prices(["AAPL", "MSFT"], START, END)

        assert "AAPL" in result.columns
        assert "MSFT" in result.columns

    def test_does_not_call_alpaca_when_fresh_but_empty(self):
        # Alpaca returns empty — should not call upsert
        cache, mock_db, mock_alpaca = _make_cache(
            db_prices=pd.DataFrame(), alpaca_bars=pd.DataFrame()
        )

        cache.get_prices(["AAPL"], START, END)

        mock_db.upsert_prices.assert_not_called()


# ---------------------------------------------------------------------------
# warm_cache
# ---------------------------------------------------------------------------

class TestWarmCache:
    def test_calls_get_ohlcv_records_and_upsert(self):
        cache, mock_db, mock_alpaca = _make_cache()
        mock_alpaca.get_ohlcv_records.return_value = [
            {"time": datetime(2025, 1, 2), "symbol": "AAPL", "close": 150.0}
        ]

        cache.warm_cache(["AAPL"], days=60)

        mock_alpaca.get_ohlcv_records.assert_called_once()
        mock_db.upsert_ohlcv.assert_called_once()

    def test_no_op_for_empty_symbols(self):
        cache, mock_db, mock_alpaca = _make_cache()

        cache.warm_cache([], days=60)

        mock_alpaca.get_ohlcv_records.assert_not_called()
        mock_db.upsert_ohlcv.assert_not_called()

    def test_passes_correct_day_window(self):
        cache, _, mock_alpaca = _make_cache()
        mock_alpaca.get_ohlcv_records.return_value = []

        cache.warm_cache(["AAPL"], days=30)

        _, start_arg, end_arg = mock_alpaca.get_ohlcv_records.call_args[0]
        assert (end_arg - start_arg).days == 30


# ---------------------------------------------------------------------------
# get_prices — failed ticker filtering
# ---------------------------------------------------------------------------

class TestFailedTickerFiltering:
    def test_known_bad_symbol_is_not_sent_to_alpaca(self):
        cache, mock_db, mock_alpaca = _make_cache()
        cache._failed = {"BAD"}

        cache.get_prices(["BAD"], START, END)

        mock_alpaca.get_historical_bars.assert_not_called()

    def test_symbol_not_returned_by_alpaca_is_added_to_failed_set(self):
        # Alpaca returns AAPL but silently drops MSFT
        fresh = _close_df(["AAPL"], ["2025-01-02"])
        cache, mock_db, mock_alpaca = _make_cache(
            db_prices=pd.DataFrame(), alpaca_bars=fresh
        )

        cache.get_prices(["AAPL", "MSFT"], START, END)

        assert "MSFT" in cache._failed

    def test_symbol_not_returned_by_alpaca_is_persisted_to_db(self):
        fresh = _close_df(["AAPL"], ["2025-01-02"])
        cache, mock_db, mock_alpaca = _make_cache(
            db_prices=pd.DataFrame(), alpaca_bars=fresh
        )

        cache.get_prices(["AAPL", "MSFT"], START, END)

        mock_db.mark_ticker_failed.assert_called_once_with("MSFT", "no data from Alpaca")

    def test_symbol_in_failed_set_skipped_on_subsequent_call(self):
        fresh = _close_df(["AAPL"], ["2025-01-02"])
        cache, mock_db, mock_alpaca = _make_cache(
            db_prices=pd.DataFrame(), alpaca_bars=fresh
        )
        # First call marks MSFT as failed
        cache.get_prices(["AAPL", "MSFT"], START, END)
        mock_alpaca.get_historical_bars.reset_mock()

        # Second call should not request MSFT from Alpaca at all
        cache.get_prices(["MSFT"], START, END)

        mock_alpaca.get_historical_bars.assert_not_called()

    def test_failed_symbols_loaded_from_db_at_init(self):
        mock_db = MagicMock()
        mock_alpaca = MagicMock()
        mock_db.get_failed_tickers.return_value = ["T.PRA", "ACHR.WS"]
        mock_alpaca.get_historical_bars.return_value = pd.DataFrame()

        cache = StockDataCache(mock_db, mock_alpaca)

        assert "T.PRA" in cache._failed
        assert "ACHR.WS" in cache._failed

    def test_all_failed_when_alpaca_returns_empty(self):
        cache, mock_db, mock_alpaca = _make_cache(
            db_prices=pd.DataFrame(), alpaca_bars=pd.DataFrame()
        )

        cache.get_prices(["AAPL", "MSFT"], START, END)

        assert "AAPL" in cache._failed
        assert "MSFT" in cache._failed
        assert mock_db.mark_ticker_failed.call_count == 2


# ---------------------------------------------------------------------------
# _find_missing_symbols
# ---------------------------------------------------------------------------

class TestFindMissingSymbols:
    def test_all_missing_when_cached_empty(self):
        cache, _, _ = _make_cache()
        result = cache._find_missing_symbols(["AAPL", "MSFT"], pd.DataFrame(), START, END)
        assert result == ["AAPL", "MSFT"]

    def test_symbol_missing_when_not_in_cached_columns(self):
        # Use a narrow range that AAPL's single data point fully covers.
        narrow_start = datetime(2025, 1, 2)
        narrow_end = datetime(2025, 1, 2)
        cached = _close_df(["AAPL"], ["2025-01-02"])
        cache, _, _ = _make_cache(db_prices=cached)
        result = cache._find_missing_symbols(["AAPL", "MSFT"], cached, narrow_start, narrow_end)
        assert "MSFT" in result
        assert "AAPL" not in result

    def test_symbol_missing_when_date_range_not_covered(self):
        # Cache only has data at the very end of the requested range
        cached = _close_df(["AAPL"], ["2025-02-28"])
        cache, _, _ = _make_cache(db_prices=cached)
        # The start of the range (2025-01-01) is not covered
        result = cache._find_missing_symbols(["AAPL"], cached, START, END)
        assert "AAPL" in result


# ---------------------------------------------------------------------------
# _merge helper
# ---------------------------------------------------------------------------

class TestMerge:
    def test_returns_fresh_when_cached_empty(self):
        fresh = _close_df(["AAPL"], ["2025-01-02"])
        result = _merge(pd.DataFrame(), fresh)
        assert "AAPL" in result.columns

    def test_returns_cached_when_fresh_empty(self):
        cached = _close_df(["AAPL"], ["2025-01-02"])
        result = _merge(cached, pd.DataFrame())
        assert "AAPL" in result.columns

    def test_fresh_wins_on_overlap(self):
        dates = ["2025-01-02"]
        cached = _close_df(["AAPL"], dates, values=[100.0])
        fresh = _close_df(["AAPL"], dates, values=[999.0])
        result = _merge(cached, fresh)
        assert result.loc[result.index[0], "AAPL"] == 999.0

    def test_result_is_sorted_by_date(self):
        cached = _close_df(["AAPL"], ["2025-01-03"])
        fresh = _close_df(["AAPL"], ["2025-01-02"])
        result = _merge(cached, fresh)
        assert result.index[0] < result.index[-1]
