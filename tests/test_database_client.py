"""
Unit tests for DatabaseClient.

psycopg2 and its connection pool are fully mocked — no live database required.
"""

from datetime import datetime, date
from unittest.mock import MagicMock, patch, call
import pandas as pd
import pytest

from DatabaseClient import DatabaseClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client():
    """Return a DatabaseClient whose connection pool is fully mocked."""
    with patch("DatabaseClient.pool.SimpleConnectionPool") as mock_pool_cls:
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool

        client = DatabaseClient("postgresql://fake/db")
        client._pool = mock_pool
        return client, mock_pool


def _mock_conn(mock_pool, fetchall_return=None, fetchone_return=None):
    """Wire a mock connection/cursor onto the pool."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)
    if fetchall_return is not None:
        mock_cur.fetchall.return_value = fetchall_return
    if fetchone_return is not None:
        mock_cur.fetchone.return_value = fetchone_return

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_pool.getconn.return_value = mock_conn
    return mock_conn, mock_cur


# ---------------------------------------------------------------------------
# get_prices
# ---------------------------------------------------------------------------

class TestGetPrices:
    def test_returns_empty_dataframe_when_no_rows(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[])

        result = client.get_prices(["AAPL"], datetime(2025, 1, 1), datetime(2025, 1, 31))

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_pivots_rows_into_symbol_columns(self):
        client, mock_pool = _make_client()
        rows = [
            (datetime(2025, 1, 2), "AAPL", 150.0),
            (datetime(2025, 1, 2), "MSFT", 300.0),
            (datetime(2025, 1, 3), "AAPL", 152.0),
            (datetime(2025, 1, 3), "MSFT", 305.0),
        ]
        _mock_conn(mock_pool, fetchall_return=rows)

        result = client.get_prices(
            ["AAPL", "MSFT"], datetime(2025, 1, 1), datetime(2025, 1, 31)
        )

        assert list(result.columns) == ["AAPL", "MSFT"]
        assert len(result) == 2
        assert result.loc[result.index[0], "AAPL"] == 150.0

    def test_passes_correct_params_to_cursor(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchall_return=[])
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)

        client.get_prices(["AAPL"], start, end)

        _sql, params = mock_cur.execute.call_args[0]
        assert params[0] == ["AAPL"]
        assert params[1] == start
        assert params[2] == end


# ---------------------------------------------------------------------------
# upsert_prices
# ---------------------------------------------------------------------------

class TestUpsertPrices:
    def test_skips_empty_dataframe(self):
        client, mock_pool = _make_client()
        mock_conn, _ = _mock_conn(mock_pool)

        client.upsert_prices(pd.DataFrame())

        mock_pool.getconn.assert_not_called()

    def test_inserts_rows_for_non_empty_dataframe(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool)

        df = pd.DataFrame(
            {"AAPL": [150.0, 152.0]},
            index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
        )

        with patch("DatabaseClient.psycopg2.extras.execute_values") as mock_ev:
            client.upsert_prices(df)
            assert mock_ev.called
            rows = mock_ev.call_args[0][2]
            assert len(rows) == 2


# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------

class TestTickers:
    def test_get_tickers_returns_list(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[("AAPL",), ("MSFT",)])

        result = client.get_tickers()

        assert result == ["AAPL", "MSFT"]

    def test_upsert_tickers_calls_execute_values(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool)

        with patch("DatabaseClient.psycopg2.extras.execute_values") as mock_ev:
            client.upsert_tickers(["AAPL", "MSFT"], "NASDAQ")
            assert mock_ev.called
            rows = mock_ev.call_args[0][2]
            assert all(r[1] == "NASDAQ" for r in rows)


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------

class TestPairs:
    def test_load_active_pairs_returns_dict_keyed_by_lag(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[
            (1, "AAPL", "MSFT", 1, 2, 5, 0.91),
        ])

        result = client.load_active_pairs("run01")

        assert "MSFT" in result
        assert result["MSFT"]["lead_stock"] == "AAPL"
        assert result["MSFT"]["pair_id"] == 1
        assert result["MSFT"]["action"] == "hold"

    def test_load_active_pairs_filters_by_run_id(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchall_return=[])

        client.load_active_pairs("run01")

        sql, params = mock_cur.execute.call_args[0]
        assert "run_id" in sql
        assert params == ("run01",)

    def test_load_active_pairs_returns_empty_dict_when_none(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[])

        assert client.load_active_pairs("run01") == {}

    def test_save_pair_returns_id_from_insert(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchone_return=(42,))

        pair = {
            "lead_stock": "AAPL", "lag_stock": "MSFT",
            "lag": 1, "short_ma": 2, "long_ma": 5, "corr": 0.88,
        }
        result = client.save_pair(pair, "run01")

        assert result == 42

    def test_save_pair_includes_run_id_in_params(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchone_return=(42,))

        pair = {
            "lead_stock": "AAPL", "lag_stock": "MSFT",
            "lag": 1, "short_ma": 2, "long_ma": 5, "corr": 0.88,
        }
        client.save_pair(pair, "run01")

        _sql, params = mock_cur.execute.call_args[0]
        assert params[0] == "run01"

    def test_deactivate_pair_executes_update(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.deactivate_pair("MSFT", "run01")

        sql_called = mock_cur.execute.call_args[0][0]
        assert "active=FALSE" in sql_called
        assert mock_cur.execute.call_args[0][1] == ("MSFT", "run01")

    def test_deactivate_pair_scopes_to_run_id(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.deactivate_pair("MSFT", "run01")

        sql = mock_cur.execute.call_args[0][0]
        assert "run_id" in sql


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

class TestRunMetadata:
    def test_create_run_inserts_row(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.create_run("abc123", "backtest", {"budget": 10000})

        sql = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO backtest_runs" in sql

    def test_close_run_updates_completed_at(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.close_run("abc123")

        sql = mock_cur.execute.call_args[0][0]
        assert "completed_at" in sql
        assert mock_cur.execute.call_args[0][1] == ("abc123",)


# ---------------------------------------------------------------------------
# Snapshots and trades
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_snapshot_inserts_all_metrics(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)
        ts = datetime(2025, 11, 3, 14, 30)

        client.log_snapshot(
            "abc123", ts,
            portfolio_value=10500.0, cash=5000.0,
            spy_value=10200.0, active_pairs=3,
            avg_correlation=0.87, cash_ratio=0.48,
            daily_buys=2, daily_sells=1,
        )

        sql = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO portfolio_snapshots" in sql
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == ts
        assert params[1] == "abc123"
        assert params[2] == 10500.0

    def test_log_trade_inserts_fill_row(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)
        ts = datetime(2025, 11, 3, 14, 30)

        client.log_trade(
            run_id="abc123", symbol="MSFT", side="buy",
            quantity=10, price=300.0, filled_at=ts,
            pair_id=1, slippage=0.05,
        )

        sql = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO trades" in sql
        params = mock_cur.execute.call_args[0][1]
        assert "MSFT" in params
        assert "buy" in params
