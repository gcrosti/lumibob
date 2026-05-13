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

    def test_clear_tickers_executes_delete(self):
        """clear_tickers() should issue a DELETE FROM tickers statement."""
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.clear_tickers()

        sql = mock_cur.execute.call_args[0][0]
        assert "DELETE" in sql.upper()
        assert "tickers" in sql.lower()


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------

class TestPairs:
    def _pair_row(self, pid=1, lead="AAPL", lag="MSFT", lag_days=1,
                  short_ma=2, long_ma=5, corr=0.91, initial_cost=None,
                  sim_ret=None, sim_sharpe=None, signal_type=None,
                  zscore_window=None, entry_threshold=None, exit_threshold=None,
                  coint_pvalue=None, halflife_days=None, lead_short_qty=None):
        """Build a 17-column pairs row matching the SELECT in load_active_pairs."""
        return (pid, lead, lag, lag_days, short_ma, long_ma, corr, initial_cost,
                sim_ret, sim_sharpe, signal_type, zscore_window,
                entry_threshold, exit_threshold,
                coint_pvalue, halflife_days, lead_short_qty)

    def test_load_active_pairs_returns_dict_keyed_by_lag(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[self._pair_row()])

        result = client.load_active_pairs("run01")

        assert "MSFT" in result
        assert result["MSFT"]["lead_stock"] == "AAPL"
        assert result["MSFT"]["pair_id"] == 1
        assert result["MSFT"]["action"] == "hold"
        assert result["MSFT"]["corr_long"] == 0.91
        assert result["MSFT"]["lead_short_qty"] is None

    def test_load_active_pairs_includes_lead_short_qty_when_set(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[self._pair_row(lead_short_qty=12.5)])

        result = client.load_active_pairs("run01")

        assert result["MSFT"]["lead_short_qty"] == 12.5

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
            "lag": 1, "short_ma": 2, "long_ma": 5, "corr_long": 0.88,
        }
        result = client.save_pair(pair, "run01")

        assert result == 42

    def test_save_pair_includes_run_id_in_params(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchone_return=(42,))

        pair = {
            "lead_stock": "AAPL", "lag_stock": "MSFT",
            "lag": 1, "short_ma": 2, "long_ma": 5, "corr_long": 0.88,
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

    def test_save_pair_includes_simulated_return_when_provided(self):
        """simulated_return is passed through to the INSERT params."""
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchone_return=(7,))

        pair = {
            "lead_stock": "AAPL", "lag_stock": "MSFT",
            "lag": 2, "short_ma": 1, "long_ma": 3,
            "corr_long": 0.92, "simulated_return": 0.07,
        }
        client.save_pair(pair, "run01")

        _sql, params = mock_cur.execute.call_args[0]
        assert 0.07 in params

    def test_save_pair_passes_none_for_missing_simulated_return(self):
        """When simulated_return is absent, None is stored."""
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchone_return=(8,))

        pair = {
            "lead_stock": "AAPL", "lag_stock": "MSFT",
            "lag": 1, "short_ma": 2, "long_ma": 5, "corr_long": 0.91,
        }
        client.save_pair(pair, "run01")

        _sql, params = mock_cur.execute.call_args[0]
        assert None in params

    def test_migrate_pairs_simulated_return_executes_alter(self):
        """Migration should issue ALTER TABLE … ADD COLUMN IF NOT EXISTS for all new columns."""
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.migrate_pairs_simulated_return()

        all_calls = [call[0][0] for call in mock_cur.execute.call_args_list]
        assert any("simulated_return" in sql for sql in all_calls)
        assert any("initial_cost" in sql for sql in all_calls)
        assert any("daily_topups" in sql for sql in all_calls)
        assert any("pairs_scanned" in sql for sql in all_calls)
        assert any("ADD COLUMN" in sql for sql in all_calls)

    def test_migrate_pairs_sim_sharpe_executes_alter(self):
        """migrate_pairs_sim_sharpe should add sim_sharpe column to pairs table."""
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.migrate_pairs_sim_sharpe()

        sql = mock_cur.execute.call_args[0][0]
        assert "sim_sharpe" in sql
        assert "ADD COLUMN IF NOT EXISTS" in sql

    def test_save_pair_includes_sim_sharpe_when_provided(self):
        """sim_sharpe is passed through to the INSERT params."""
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchone_return=(9,))

        pair = {
            "lead_stock": "AAPL", "lag_stock": "MSFT",
            "lag": 1, "short_ma": 2, "long_ma": 5,
            "corr_long": 0.92, "sim_sharpe": 1.4,
        }
        client.save_pair(pair, "run01")

        _sql, params = mock_cur.execute.call_args[0]
        assert 1.4 in params

    def test_save_pair_passes_none_for_missing_sim_sharpe(self):
        """When sim_sharpe is absent, None is stored rather than raising."""
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchone_return=(10,))

        pair = {
            "lead_stock": "AAPL", "lag_stock": "MSFT",
            "lag": 1, "short_ma": 2, "long_ma": 5, "corr_long": 0.91,
        }
        client.save_pair(pair, "run01")

        _sql, params = mock_cur.execute.call_args[0]
        assert None in params

    def test_save_pair_passes_none_when_corr_long_absent(self):
        """pairs.correlation is NULL when corr_long is not on the pair dict."""
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool, fetchone_return=(11,))

        pair = {
            "lead_stock": "AAPL", "lag_stock": "MSFT",
            "lag": 1, "short_ma": 2, "long_ma": 5,
        }
        client.save_pair(pair, "run01")

        _sql, params = mock_cur.execute.call_args[0]
        assert params[6] is None

    def test_update_pair_initial_cost_executes_update(self):
        """update_pair_initial_cost should UPDATE pairs SET initial_cost for the given id."""
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.update_pair_initial_cost(42, 500.0)

        sql = mock_cur.execute.call_args[0][0]
        assert "initial_cost" in sql
        assert "UPDATE pairs" in sql
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == 500.0
        assert params[1] == 42

    def test_load_active_pairs_includes_initial_cost(self):
        """load_active_pairs should return initial_cost in each pair dict."""
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[self._pair_row(initial_cost=487.50)])

        result = client.load_active_pairs("run01")

        assert "MSFT" in result
        assert result["MSFT"]["initial_cost"] == 487.50

    def test_load_active_pairs_initial_cost_none_when_not_set(self):
        """initial_cost should be None when the DB value is NULL."""
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[self._pair_row(initial_cost=None)])

        result = client.load_active_pairs("run01")

        assert result["MSFT"]["initial_cost"] is None

    def test_load_active_pairs_includes_sim_sharpe_and_zscore_fields(self):
        """All fields needed by before_market_opens() are present in loaded pairs."""
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[
            self._pair_row(
                sim_ret=0.08, sim_sharpe=1.2, signal_type="zscore",
                zscore_window=20, entry_threshold=2.0, exit_threshold=0.5,
            )
        ])

        result = client.load_active_pairs("run01")
        pair = result["MSFT"]

        assert pair["simulated_return"] == 0.08
        assert pair["sim_sharpe"] == 1.2
        assert pair["signal_type"] == "zscore"
        assert pair["zscore_window"] == 20
        assert pair["entry_threshold"] == 2.0
        assert pair["exit_threshold"] == 0.5

    def test_load_active_pairs_sim_sharpe_none_when_not_set(self):
        """sim_sharpe is None when the DB value is NULL."""
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[self._pair_row(sim_sharpe=None)])

        result = client.load_active_pairs("run01")

        assert result["MSFT"]["sim_sharpe"] is None


# ---------------------------------------------------------------------------
# Failed ticker registry
# ---------------------------------------------------------------------------

class TestFailedTickers:
    def test_migrate_creates_table_if_not_exists(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.migrate_failed_tickers()

        sql = mock_cur.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS" in sql
        assert "failed_tickers" in sql

    def test_get_failed_tickers_returns_symbol_list(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[("T.PRA",), ("ACHR.WS",)])

        result = client.get_failed_tickers()

        assert result == ["T.PRA", "ACHR.WS"]

    def test_get_failed_tickers_returns_empty_list_when_none(self):
        client, mock_pool = _make_client()
        _mock_conn(mock_pool, fetchall_return=[])

        result = client.get_failed_tickers()

        assert result == []

    def test_mark_ticker_failed_inserts_symbol_and_reason(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.mark_ticker_failed("T.PRA", "no data from Alpaca")

        sql = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO failed_tickers" in sql
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == "T.PRA"
        assert params[1] == "no data from Alpaca"

    def test_mark_ticker_failed_uses_on_conflict_do_nothing(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.mark_ticker_failed("T.PRA")

        sql = mock_cur.execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    def test_mark_ticker_failed_default_reason_is_empty_string(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.mark_ticker_failed("T.PRA")

        params = mock_cur.execute.call_args[0][1]
        assert params[1] == ""


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
            daily_topups=3, pairs_scanned=120,
            candidates_found=5, candidates_buy_ready=3,
            gross_long_pct=0.55, gross_short_pct=0.42,
        )

        sql = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO portfolio_snapshots" in sql
        assert "gross_long_pct" in sql
        assert "gross_short_pct" in sql
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == ts
        assert params[1] == "abc123"
        assert params[2] == 10500.0
        assert params[-2] == 0.55   # gross_long_pct
        assert params[-1] == 0.42   # gross_short_pct

    def test_log_snapshot_includes_funnel_and_topup_columns(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)
        ts = datetime(2025, 11, 3, 14, 30)

        client.log_snapshot(
            "abc123", ts,
            portfolio_value=10500.0, cash=5000.0,
            daily_buys=2, daily_sells=1,
            daily_topups=3, pairs_scanned=120,
            candidates_found=5, candidates_buy_ready=3,
        )

        sql = mock_cur.execute.call_args[0][0]
        assert "daily_topups" in sql
        assert "pairs_scanned" in sql
        assert "candidates_found" in sql
        assert "candidates_buy_ready" in sql
        assert "avg_watchlist_ttl" in sql
        params = mock_cur.execute.call_args[0][1]
        # avg_watchlist_ttl is now the final param; preceding columns shift by 1
        assert params[-8] == 3    # daily_topups
        assert params[-7] == 120  # pairs_scanned
        assert params[-6] == 5    # candidates_found
        assert params[-5] == 3    # candidates_buy_ready
        assert params[-4] is None  # avg_zscore (not passed → None)
        assert params[-3] is None  # avg_watchlist_ttl (not passed → None)
        assert params[-2] is None  # gross_long_pct (not passed → None)
        assert params[-1] is None  # gross_short_pct (not passed → None)

    def test_update_pair_lead_short_qty_executes_update(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.update_pair_lead_short_qty(99, 7.5)

        sql = mock_cur.execute.call_args[0][0]
        assert "UPDATE pairs" in sql
        assert "lead_short_qty" in sql
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == 7.5
        assert params[1] == 99

    def test_update_pair_lead_short_qty_accepts_none(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.update_pair_lead_short_qty(5, None)

        params = mock_cur.execute.call_args[0][1]
        assert params[0] is None

    def test_migrate_short_leg_issues_alter_for_leg_column(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.migrate_short_leg()

        all_calls = [call[0][0] for call in mock_cur.execute.call_args_list]
        assert any("leg" in sql for sql in all_calls)
        assert any("lead_short_qty" in sql for sql in all_calls)
        assert any("ADD COLUMN IF NOT EXISTS" in sql for sql in all_calls)

    def test_migrate_snapshot_deployment_issues_alter_for_gross_columns(self):
        client, mock_pool = _make_client()
        _, mock_cur = _mock_conn(mock_pool)

        client.migrate_snapshot_deployment()

        all_calls = [call[0][0] for call in mock_cur.execute.call_args_list]
        assert any("gross_long_pct" in sql for sql in all_calls)
        assert any("gross_short_pct" in sql for sql in all_calls)
        assert any("ADD COLUMN IF NOT EXISTS" in sql for sql in all_calls)

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
        assert params[-1] == "long"  # leg
