"""
DatabaseClient — single source of truth for all database I/O.

Owns every SQL statement in the project; no other class writes raw queries.
Instantiated once at strategy startup and shared with StockDataCache and
BobsBrain. Stateless beyond the connection pool so it can be reused safely
across the strategy lifecycle.
"""

import json
import logging
import math
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import pool


def _float_correlation_value(value: Any) -> float | None:
    """Normalize a correlation scalar; None if missing, NaN, or non-finite."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _pair_corr_long_for_db(pair: dict) -> float | None:
    """Long-horizon correlation persisted in ``pairs.correlation``."""
    return _float_correlation_value(pair.get("corr_long"))


class DatabaseClient:
    def __init__(self, db_url: str, min_conn: int = 1, max_conn: int = 5):
        self._pool = pool.SimpleConnectionPool(min_conn, max_conn, db_url)

    def close(self) -> None:
        self._pool.closeall()

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def get_prices(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Return daily close prices as a DataFrame with a DatetimeIndex and
        symbol columns — matching the shape that yf.download() produces so
        callers need no conversion.

        Returns an empty DataFrame when no rows are found.
        """
        sql = """
            SELECT time, symbol, close
            FROM   stock_prices
            WHERE  symbol = ANY(%s)
              AND  time >= %s
              AND  time <= %s
            ORDER  BY time
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (symbols, start, end))
                rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["time", "symbol", "close"])
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
        return df.pivot(index="time", columns="symbol", values="close")

    def upsert_prices(self, df: pd.DataFrame) -> None:
        """
        Bulk-insert OHLCV rows. df must have a DatetimeIndex and symbol
        columns containing close prices (at minimum). Extra OHLCV columns
        (open, high, low, volume) are used when present.

        Duplicate (symbol, time) rows are silently skipped.
        """
        if df.empty:
            return

        rows: list[tuple] = []
        for ts, row in df.iterrows():
            for symbol in df.columns:
                val = row[symbol]
                if pd.isna(val):
                    continue
                rows.append((pd.Timestamp(ts), symbol, None, None, None, float(val), None))

        sql = """
            INSERT INTO stock_prices (time, symbol, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        with self._conn() as conn:
            psycopg2.extras.execute_values(conn.cursor(), sql, rows)

    def upsert_ohlcv(self, records: list[dict]) -> None:
        """
        Insert OHLCV records from a list of dicts with keys:
        time, symbol, open, high, low, close, volume.
        Duplicate (symbol, time) rows are silently skipped.
        """
        if not records:
            return
        rows = [
            (
                r["time"], r["symbol"],
                r.get("open"), r.get("high"), r.get("low"),
                r["close"],
                r.get("volume"),
            )
            for r in records
        ]
        sql = """
            INSERT INTO stock_prices (time, symbol, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        with self._conn() as conn:
            psycopg2.extras.execute_values(conn.cursor(), sql, rows)

    # ------------------------------------------------------------------
    # Ticker universe
    # ------------------------------------------------------------------

    def get_tickers(self) -> list[str]:
        """Return all symbols currently stored in the tickers table."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol FROM tickers ORDER BY symbol")
                return [row[0] for row in cur.fetchall()]

    def clear_tickers(self) -> None:
        """
        Remove all rows from the tickers table, forcing a full refresh from
        Alpaca on the next run. Call this after changing asset filters in
        AlpacaClient.get_tradeable_assets() so stale symbols are evicted.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tickers")

    def upsert_tickers(self, symbols: list[str], exchange: str) -> None:
        """Insert or update ticker rows for a given exchange."""
        today = date.today()
        rows = [(s, exchange, today) for s in symbols]
        sql = """
            INSERT INTO tickers (symbol, exchange, last_updated)
            VALUES %s
            ON CONFLICT (symbol) DO UPDATE
                SET exchange = EXCLUDED.exchange,
                    last_updated = EXCLUDED.last_updated
        """
        with self._conn() as conn:
            psycopg2.extras.execute_values(conn.cursor(), sql, rows)

    # ------------------------------------------------------------------
    # Pairs persistence  (replaces pairs/pair_history.json)
    # ------------------------------------------------------------------

    def load_active_pairs(self, run_id: str) -> dict:
        """
        Return active pairs for the given run as a dict keyed by lag_symbol —
        the same shape that pair_history.json used — for drop-in compatibility
        with BobsBrain.

        All columns needed by BobsBrain.before_market_opens() are selected so
        that DB-loaded pairs can be evaluated immediately without missing fields.
        The ``correlation`` column is exposed in-memory as ``corr_long``.
        """
        sql = """
            SELECT id, lead_symbol, lag_symbol, lag_days,
                   short_ma, long_ma, correlation, initial_cost,
                   simulated_return, sim_sharpe, signal_type,
                   zscore_window, entry_threshold, exit_threshold,
                   coint_pvalue, halflife_days
            FROM   pairs
            WHERE  active = TRUE
              AND  run_id = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (run_id,))
                rows = cur.fetchall()

        result: dict = {}
        for row in rows:
            (pid, lead, lag, lag_days, short_ma, long_ma, correlation, initial_cost,
             sim_ret, sim_sharpe, signal_type, zscore_window,
             entry_threshold, exit_threshold,
             coint_pvalue, halflife_days) = row
            result[lag] = {
                "pair_id":          pid,
                "lead_stock":       lead,
                "lag_stock":        lag,
                "lag":              lag_days,
                "short_ma":         short_ma,
                "long_ma":          long_ma,
                "corr_long":        _float_correlation_value(correlation),
                "action":           "hold",
                "initial_cost":     float(initial_cost) if initial_cost is not None else None,
                "simulated_return": float(sim_ret) if sim_ret is not None else None,
                "sim_sharpe":       float(sim_sharpe) if sim_sharpe is not None else None,
                "signal_type":      signal_type,
                "zscore_window":    zscore_window,
                "entry_threshold":  float(entry_threshold) if entry_threshold is not None else None,
                "exit_threshold":   float(exit_threshold) if exit_threshold is not None else None,
                "coint_pvalue":     float(coint_pvalue) if coint_pvalue is not None else None,
                "halflife_days":    float(halflife_days) if halflife_days is not None else None,
            }
        return result

    def migrate_pairs_simulated_return(self) -> None:
        """
        Idempotent migration: add the simulated_return and initial_cost columns
        to the pairs table if they do not already exist, and add the discovery
        funnel + top-up indicator columns to portfolio_snapshots.
        Safe to call on every startup.
        """
        statements = [
            "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS simulated_return DOUBLE PRECISION",
            "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS initial_cost NUMERIC",
            "ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS daily_topups INT",
            "ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS pairs_scanned INT",
            "ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS candidates_found INT",
            "ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS candidates_buy_ready INT",
        ]
        with self._conn() as conn:
            with conn.cursor() as cur:
                for sql in statements:
                    cur.execute(sql)

    def migrate_zscore_columns(self) -> None:
        """
        Idempotent migration: add Z-score signal columns to the pairs table
        and avg_zscore to portfolio_snapshots.  Safe to call on every startup.
        """
        statements = [
            "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS signal_type TEXT",
            "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS zscore_window INT",
            "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS entry_threshold DOUBLE PRECISION",
            "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS exit_threshold DOUBLE PRECISION",
            "ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS avg_zscore DOUBLE PRECISION",
            "ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS avg_watchlist_ttl DOUBLE PRECISION",
        ]
        with self._conn() as conn:
            with conn.cursor() as cur:
                for sql in statements:
                    cur.execute(sql)

    def migrate_pairs_sim_sharpe(self) -> None:
        """
        Idempotent migration: add the sim_sharpe column to the pairs table if
        it does not already exist.  Safe to call on every startup.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS sim_sharpe DOUBLE PRECISION"
                )

    def migrate_ticker_metadata(self) -> None:
        """
        Idempotent migration: create the ticker_metadata table if it does not
        already exist, and add SIC-related columns.  Safe to call on every startup.
        """
        statements = [
            """
            CREATE TABLE IF NOT EXISTS ticker_metadata (
                symbol      TEXT PRIMARY KEY,
                sector      TEXT,
                is_etf      BOOLEAN NOT NULL DEFAULT FALSE,
                fetched_at  TIMESTAMPTZ NOT NULL
            )
            """,
            "ALTER TABLE ticker_metadata ADD COLUMN IF NOT EXISTS sic_code INT",
            "ALTER TABLE ticker_metadata ADD COLUMN IF NOT EXISTS sic_sector TEXT",
            "ALTER TABLE ticker_metadata ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'yfinance'",
        ]
        with self._conn() as conn:
            with conn.cursor() as cur:
                for sql in statements:
                    cur.execute(sql)

    def get_ticker_metadata(self, symbols: list[str]) -> pd.DataFrame:
        """
        Return sector/ETF metadata for the given symbols as a DataFrame with
        columns [symbol, sector, is_etf, fetched_at].  Only rows that already
        exist in the DB are returned; missing symbols are simply absent.
        """
        if not symbols:
            return pd.DataFrame(columns=['symbol', 'sector', 'is_etf', 'fetched_at'])
        sql = """
            SELECT symbol, sector, is_etf, fetched_at
            FROM   ticker_metadata
            WHERE  symbol = ANY(%s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (symbols,))
                rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=['symbol', 'sector', 'is_etf', 'fetched_at'])
        return pd.DataFrame(rows, columns=['symbol', 'sector', 'is_etf', 'fetched_at'])

    def upsert_ticker_metadata(self, records: list[dict]) -> None:
        """
        Insert or update ticker metadata rows.  Each record must contain:
            symbol (str), sector (str | None), is_etf (bool), fetched_at (datetime)
        Existing rows are overwritten so stale data can be refreshed by clearing
        the table and re-running the strategy.
        """
        if not records:
            return
        rows = [
            (r['symbol'], r.get('sector'), bool(r.get('is_etf', False)), r['fetched_at'])
            for r in records
        ]
        sql = """
            INSERT INTO ticker_metadata (symbol, sector, is_etf, fetched_at)
            VALUES %s
            ON CONFLICT (symbol) DO UPDATE
                SET sector     = EXCLUDED.sector,
                    is_etf     = EXCLUDED.is_etf,
                    fetched_at = EXCLUDED.fetched_at
        """
        with self._conn() as conn:
            psycopg2.extras.execute_values(conn.cursor(), sql, rows)

    def upsert_sec_metadata(self, records: list[dict]) -> None:
        """
        Insert or update ticker metadata from SEC EDGAR SIC data.
        Each record must have: symbol, sic_code, sic_sector, is_etf, fetched_at.
        Overwrites sector with sic_sector so SEC data takes precedence.
        """
        if not records:
            return
        rows = [
            (
                r['symbol'],
                r.get('sic_sector'),
                bool(r.get('is_etf', False)),
                r['fetched_at'],
                r.get('sic_code'),
                r.get('sic_sector'),
                'sec_edgar',
            )
            for r in records
        ]
        sql = """
            INSERT INTO ticker_metadata (symbol, sector, is_etf, fetched_at, sic_code, sic_sector, source)
            VALUES %s
            ON CONFLICT (symbol) DO UPDATE
                SET sector     = EXCLUDED.sector,
                    is_etf     = EXCLUDED.is_etf,
                    fetched_at = EXCLUDED.fetched_at,
                    sic_code   = EXCLUDED.sic_code,
                    sic_sector = EXCLUDED.sic_sector,
                    source     = EXCLUDED.source
        """
        with self._conn() as conn:
            psycopg2.extras.execute_values(conn.cursor(), sql, rows)

    def save_pair(self, pair: dict, run_id: str) -> int:
        """
        Insert a new pair row scoped to run_id. Returns the new pair id.
        Silently skips if the same (run_id, lead, lag, lag_days) already exists
        and returns the existing id instead.

        Optional pair keys:
            corr_long (float)          -- long-horizon correlation → pairs.correlation
            simulated_return (float)   -- best historical simulated return
            sim_sharpe (float)         -- annualised Sharpe of the simulated strategy
            signal_type (str)          -- 'ma' or 'zscore'
            zscore_window (int)        -- rolling window for Z-score spread
            entry_threshold (float)    -- Z-score entry level
            exit_threshold (float)     -- Z-score exit level
        """
        sql = """
            INSERT INTO pairs
                (run_id, lead_symbol, lag_symbol, lag_days, short_ma, long_ma,
                 correlation, simulated_return, sim_sharpe, signal_type, zscore_window,
                 entry_threshold, exit_threshold, coint_pvalue, halflife_days,
                 discovered_at, last_updated, active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            ON CONFLICT (run_id, lead_symbol, lag_symbol, lag_days)
                WHERE run_id IS NOT NULL
            DO NOTHING
            RETURNING id
        """
        today = date.today()
        with self._conn() as conn:
            with conn.cursor() as cur:
                correlation = _pair_corr_long_for_db(pair)
                sim_ret = pair.get("simulated_return")
                sim_sharpe = pair.get("sim_sharpe")
                signal_type = pair.get("signal_type", "ma")
                zscore_window = pair.get("zscore_window")
                entry_threshold = pair.get("entry_threshold")
                exit_threshold = pair.get("exit_threshold")
                coint_pvalue = pair.get("coint_pvalue")
                halflife_days = pair.get("halflife_days")
                cur.execute(sql, (
                    run_id,
                    pair["lead_stock"],
                    pair["lag_stock"],
                    pair.get("lag", 1),
                    pair.get("short_ma", 2),
                    pair.get("long_ma", 5),
                    correlation,
                    float(sim_ret) if sim_ret is not None else None,
                    float(sim_sharpe) if sim_sharpe is not None else None,
                    signal_type,
                    int(zscore_window) if zscore_window is not None else None,
                    float(entry_threshold) if entry_threshold is not None else None,
                    float(exit_threshold) if exit_threshold is not None else None,
                    float(coint_pvalue) if coint_pvalue is not None else None,
                    float(halflife_days) if halflife_days is not None else None,
                    today,
                    today,
                ))
                row = cur.fetchone()
                if row:
                    return row[0]

                # Row already exists for this run — fetch its id
                cur.execute(
                    """SELECT id FROM pairs
                       WHERE run_id=%s AND lead_symbol=%s
                         AND lag_symbol=%s AND lag_days=%s""",
                    (run_id, pair["lead_stock"], pair["lag_stock"], pair.get("lag", 1)),
                )
                result = cur.fetchone()
                return result[0] if result else -1

    def update_pair_correlation(self, pair_id: int, correlation: float) -> None:
        """Update the correlation and last_updated timestamp for an existing pair."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE pairs SET correlation=%s, last_updated=%s WHERE id=%s",
                    (float(correlation), date.today(), pair_id),
                )

    def update_pair_initial_cost(self, pair_id: int, initial_cost: float) -> None:
        """Record the initial purchase cost for a pair at first buy time."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE pairs SET initial_cost=%s WHERE id=%s",
                    (float(initial_cost), pair_id),
                )

    def deactivate_pair(self, lag_symbol: str, run_id: str) -> None:
        """Mark active pairs for the given lag symbol and run as inactive."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE pairs SET active=FALSE WHERE lag_symbol=%s AND run_id=%s AND active=TRUE",
                    (lag_symbol, run_id),
                )

    # ------------------------------------------------------------------
    # Unfetchable ticker registry
    # ------------------------------------------------------------------

    def migrate_coint_cache(self) -> None:
        """
        Idempotent migration: create the pair_coint_cache table and add
        coint_pvalue / halflife_days columns to the pairs table.
        Safe to call on every startup.
        """
        statements = [
            """
            CREATE TABLE IF NOT EXISTS pair_coint_cache (
                lead_symbol     VARCHAR(20)       NOT NULL,
                lag_symbol      VARCHAR(20)       NOT NULL,
                lookback_window INT               NOT NULL,
                window_end_date DATE              NOT NULL,
                coint_pvalue    DOUBLE PRECISION  NOT NULL,
                halflife_days   DOUBLE PRECISION,
                computed_at     TIMESTAMP         NOT NULL DEFAULT NOW(),
                PRIMARY KEY (lead_symbol, lag_symbol, lookback_window, window_end_date)
            )
            """,
            "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS coint_pvalue  DOUBLE PRECISION",
            "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS halflife_days DOUBLE PRECISION",
        ]
        with self._conn() as conn:
            with conn.cursor() as cur:
                for sql in statements:
                    cur.execute(sql)

    def load_coint_cache(
        self,
        window_end_date: date,
        lookback_window: int,
    ) -> dict[tuple[str, str], tuple[float, float | None]]:
        """
        Load all cache entries for the given (window_end_date, lookback_window)
        into an in-memory dict keyed by (lead_symbol, lag_symbol).

        Returns an empty dict if the table does not exist yet or has no rows.
        """
        sql = """
            SELECT lead_symbol, lag_symbol, coint_pvalue, halflife_days
            FROM   pair_coint_cache
            WHERE  window_end_date = %s
              AND  lookback_window = %s
        """
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (window_end_date, lookback_window))
                    rows = cur.fetchall()
            return {
                (lead, lag): (float(pval), float(hl) if hl is not None else None)
                for lead, lag, pval, hl in rows
            }
        except Exception:
            return {}

    def write_coint_cache(
        self,
        entries: dict[tuple[str, str], tuple[float, float | None]],
        window_end_date: date,
        lookback_window: int,
    ) -> None:
        """
        Upsert cointegration cache entries.  *entries* maps
        (lead_symbol, lag_symbol) → (coint_pvalue, halflife_days).
        No-op when *entries* is empty or the table does not exist.
        """
        if not entries:
            return
        rows = [
            (lead, lag, window_end_date, lookback_window, pval, hl)
            for (lead, lag), (pval, hl) in entries.items()
        ]
        sql = """
            INSERT INTO pair_coint_cache
                (lead_symbol, lag_symbol, window_end_date, lookback_window,
                 coint_pvalue, halflife_days, computed_at)
            VALUES %s
            ON CONFLICT (lead_symbol, lag_symbol, lookback_window, window_end_date)
            DO UPDATE SET
                coint_pvalue = EXCLUDED.coint_pvalue,
                halflife_days = EXCLUDED.halflife_days,
                computed_at  = NOW()
        """
        try:
            with self._conn() as conn:
                psycopg2.extras.execute_values(conn.cursor(), sql, rows)
        except Exception:
            logger.debug('write_coint_cache failed — non-fatal', exc_info=True)

    def migrate_failed_tickers(self) -> None:
        """
        Idempotent migration: create the failed_tickers table if it does not
        already exist. Safe to call on every startup.
        """
        sql = """
            CREATE TABLE IF NOT EXISTS failed_tickers (
                symbol     VARCHAR(20) PRIMARY KEY,
                reason     TEXT,
                failed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    def get_failed_tickers(self) -> list[str]:
        """Return all symbols that have been marked as unfetchable."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol FROM failed_tickers")
                return [row[0] for row in cur.fetchall()]

    def mark_ticker_failed(self, symbol: str, reason: str = "") -> None:
        """
        Record a symbol as unfetchable. Idempotent — if the symbol is already
        present the row is left unchanged so the original failed_at timestamp
        is preserved.
        """
        sql = """
            INSERT INTO failed_tickers (symbol, reason)
            VALUES (%s, %s)
            ON CONFLICT (symbol) DO NOTHING
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (symbol, reason))

    # ------------------------------------------------------------------
    # Run metadata
    # ------------------------------------------------------------------

    def create_run(self, run_id: str, mode: str, settings: dict) -> None:
        """Insert a new run row at strategy startup."""
        sql = """
            INSERT INTO backtest_runs (run_id, mode, started_at, settings)
            VALUES (%s, %s, NOW(), %s)
            ON CONFLICT (run_id) DO NOTHING
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (run_id, mode, json.dumps(settings)))

    def close_run(self, run_id: str) -> None:
        """Set completed_at for a run when the strategy finishes."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE backtest_runs SET completed_at=NOW() WHERE run_id=%s",
                    (run_id,),
                )

    # ------------------------------------------------------------------
    # Portfolio snapshots
    # ------------------------------------------------------------------

    def log_snapshot(self, run_id: str, time: datetime, **metrics: Any) -> None:
        """
        Insert one portfolio snapshot row. Accepted keyword metrics:
            portfolio_value, cash, spy_value, active_pairs,
            avg_correlation, cash_ratio, daily_buys, daily_sells,
            daily_topups, pairs_scanned, candidates_found,
            candidates_buy_ready, avg_zscore, avg_watchlist_ttl
        """
        sql = """
            INSERT INTO portfolio_snapshots
                (time, run_id, portfolio_value, cash, spy_value,
                 active_pairs, avg_correlation, cash_ratio,
                 daily_buys, daily_sells, daily_topups,
                 pairs_scanned, candidates_found, candidates_buy_ready,
                 avg_zscore, avg_watchlist_ttl)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        def _f(v):
            return float(v) if v is not None else None

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    time,
                    run_id,
                    _f(metrics.get("portfolio_value")),
                    _f(metrics.get("cash")),
                    _f(metrics.get("spy_value")),
                    metrics.get("active_pairs"),
                    _f(metrics.get("avg_correlation")),
                    _f(metrics.get("cash_ratio")),
                    metrics.get("daily_buys"),
                    metrics.get("daily_sells"),
                    metrics.get("daily_topups"),
                    metrics.get("pairs_scanned"),
                    metrics.get("candidates_found"),
                    metrics.get("candidates_buy_ready"),
                    _f(metrics.get("avg_zscore")),
                    _f(metrics.get("avg_watchlist_ttl")),
                ))

    # ------------------------------------------------------------------
    # Trade fills
    # ------------------------------------------------------------------

    def log_trade(
        self,
        run_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        filled_at: datetime,
        pair_id: int | None = None,
        slippage: float = 0.0,
        exit_reason: str | None = None,
    ) -> None:
        """Insert one trade fill row.

        exit_reason is only meaningful for sell-side trades:
          'zscore_exit'  — spread reverted below exit_threshold
          'displaced'    — pair crowded out of top-K by a higher-scoring candidate
          'data_missing' — price data unavailable; reason could not be determined
        """
        sql = """
            INSERT INTO trades
                (run_id, pair_id, symbol, side, quantity, price, slippage, filled_at, exit_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    run_id, pair_id, symbol, side,
                    quantity, price, slippage, filled_at, exit_reason,
                ))
