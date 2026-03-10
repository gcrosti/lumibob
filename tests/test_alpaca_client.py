"""
Unit tests for AlpacaClient.

All alpaca-py SDK clients are fully mocked — no live API calls or credentials
required.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import pytest

from AlpacaClient import AlpacaClient, _ensure_utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(mode: str = "backtest") -> tuple[AlpacaClient, MagicMock, MagicMock]:
    """Return an AlpacaClient with both SDK clients replaced by mocks."""
    with patch("AlpacaClient.StockHistoricalDataClient") as mock_data_cls, \
         patch("AlpacaClient.TradingClient") as mock_trade_cls:

        mock_data = MagicMock()
        mock_trade = MagicMock()
        mock_data_cls.return_value = mock_data
        mock_trade_cls.return_value = mock_trade

        client = AlpacaClient("key", "secret", paper=True, mode=mode)
        client._data_client = mock_data
        client._trading_client = mock_trade
        return client, mock_data, mock_trade


def _make_asset(symbol: str, exchange, tradable: bool = True):
    asset = MagicMock()
    asset.symbol = symbol
    asset.exchange = exchange
    asset.tradable = tradable
    return asset


def _make_bars_df(rows: list[dict]) -> MagicMock:
    """Build a mock bars response whose .df is a MultiIndex DataFrame."""
    if not rows:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index(["symbol", "timestamp"])

    mock_bars = MagicMock()
    mock_bars.df = df
    return mock_bars


# ---------------------------------------------------------------------------
# get_tradeable_assets
# ---------------------------------------------------------------------------

class TestGetTradeableAssets:
    def test_returns_only_nyse_and_nasdaq_symbols(self):
        from alpaca.trading.enums import AssetExchange

        client, _, mock_trade = _make_client()
        mock_trade.get_all_assets.return_value = [
            _make_asset("AAPL", AssetExchange.NASDAQ),
            _make_asset("F", AssetExchange.NYSE),
            _make_asset("BTC", MagicMock()),    # non-equity exchange — excluded
        ]

        result = client.get_tradeable_assets()

        assert "AAPL" in result
        assert "F" in result
        assert "BTC" not in result

    def test_excludes_non_tradable_assets(self):
        from alpaca.trading.enums import AssetExchange

        client, _, mock_trade = _make_client()
        mock_trade.get_all_assets.return_value = [
            _make_asset("AAPL", AssetExchange.NASDAQ, tradable=True),
            _make_asset("HALT", AssetExchange.NYSE, tradable=False),
        ]

        result = client.get_tradeable_assets()

        assert "HALT" not in result
        assert "AAPL" in result

    def test_returns_empty_list_when_no_assets(self):
        client, _, mock_trade = _make_client()
        mock_trade.get_all_assets.return_value = []

        assert client.get_tradeable_assets() == []


# ---------------------------------------------------------------------------
# get_historical_bars
# ---------------------------------------------------------------------------

class TestGetHistoricalBars:
    def test_returns_empty_dataframe_for_empty_symbols(self):
        client, _, _ = _make_client()
        result = client.get_historical_bars([], datetime(2025, 1, 1), datetime(2025, 1, 31))
        assert result.empty

    def test_returns_empty_dataframe_when_api_returns_no_data(self):
        client, mock_data, _ = _make_client()
        mock_data.get_stock_bars.return_value = _make_bars_df([])

        result = client.get_historical_bars(
            ["AAPL"], datetime(2025, 1, 1), datetime(2025, 1, 31)
        )
        assert result.empty

    def test_pivots_to_symbol_columns_with_naive_datetimeindex(self):
        client, mock_data, _ = _make_client()
        mock_data.get_stock_bars.return_value = _make_bars_df([
            {"symbol": "AAPL", "timestamp": "2025-01-02", "close": 150.0,
             "open": 148.0, "high": 151.0, "low": 147.0, "volume": 1000},
            {"symbol": "MSFT", "timestamp": "2025-01-02", "close": 300.0,
             "open": 298.0, "high": 302.0, "low": 297.0, "volume": 2000},
        ])

        result = client.get_historical_bars(
            ["AAPL", "MSFT"], datetime(2025, 1, 1), datetime(2025, 1, 31)
        )

        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        assert result.index.tz is None  # timezone-naive

    def test_close_prices_are_correct(self):
        client, mock_data, _ = _make_client()
        mock_data.get_stock_bars.return_value = _make_bars_df([
            {"symbol": "AAPL", "timestamp": "2025-01-02", "close": 150.0,
             "open": 148.0, "high": 151.0, "low": 147.0, "volume": 1000},
        ])

        result = client.get_historical_bars(
            ["AAPL"], datetime(2025, 1, 1), datetime(2025, 1, 31)
        )

        assert result.iloc[0]["AAPL"] == 150.0

    def test_passes_date_range_to_api(self):
        client, mock_data, _ = _make_client()
        mock_data.get_stock_bars.return_value = _make_bars_df([])
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)

        client.get_historical_bars(["AAPL"], start, end)

        call_args = mock_data.get_stock_bars.call_args[0][0]
        # StockBarsRequest may store the datetime with or without tzinfo depending
        # on the alpaca-py version; compare the naive values to stay version-agnostic.
        def _naive(dt):
            return dt.replace(tzinfo=None) if dt else dt
        assert _naive(call_args.start) == _naive(start.replace(tzinfo=timezone.utc))
        assert _naive(call_args.end) == _naive(end.replace(tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# get_ohlcv_records
# ---------------------------------------------------------------------------

class TestGetOhlcvRecords:
    def test_returns_flat_list_of_dicts(self):
        client, mock_data, _ = _make_client()
        mock_data.get_stock_bars.return_value = _make_bars_df([
            {"symbol": "AAPL", "timestamp": "2025-01-02", "close": 150.0,
             "open": 148.0, "high": 151.0, "low": 147.0, "volume": 1000},
        ])

        result = client.get_ohlcv_records(
            ["AAPL"], datetime(2025, 1, 1), datetime(2025, 1, 31)
        )

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["close"] == 150.0
        assert "time" in result[0]

    def test_returns_empty_list_for_empty_symbols(self):
        client, _, _ = _make_client()
        assert client.get_ohlcv_records([], datetime(2025, 1, 1), datetime(2025, 1, 31)) == []


# ---------------------------------------------------------------------------
# submit_order
# ---------------------------------------------------------------------------

class TestSubmitOrder:
    def test_no_op_in_backtest_mode(self):
        client, _, mock_trade = _make_client(mode="backtest")

        client.submit_order("AAPL", 10, "buy")

        mock_trade.submit_order.assert_not_called()

    def test_submits_buy_order_in_paper_mode(self):
        client, _, mock_trade = _make_client(mode="paper")

        client.submit_order("AAPL", 10, "buy")

        assert mock_trade.submit_order.called
        request = mock_trade.submit_order.call_args[0][0]
        from alpaca.trading.enums import OrderSide
        assert request.side == OrderSide.BUY
        assert request.qty == 10

    def test_submits_sell_order_in_paper_mode(self):
        client, _, mock_trade = _make_client(mode="paper")

        client.submit_order("AAPL", 5, "sell")

        assert mock_trade.submit_order.called
        request = mock_trade.submit_order.call_args[0][0]
        from alpaca.trading.enums import OrderSide
        assert request.side == OrderSide.SELL


# ---------------------------------------------------------------------------
# _ensure_utc helper
# ---------------------------------------------------------------------------

class TestEnsureUtc:
    def test_attaches_utc_to_naive_datetime(self):
        dt = datetime(2025, 1, 1)
        result = _ensure_utc(dt)
        assert result.tzinfo == timezone.utc

    def test_preserves_existing_timezone(self):
        from datetime import timezone as tz
        import pytz
        eastern = pytz.timezone("US/Eastern")
        dt = eastern.localize(datetime(2025, 1, 1, 9, 30))
        result = _ensure_utc(dt)
        assert result.tzinfo is not None
        assert result == dt
