"""
DatabaseClient — single source of truth for all database I/O.

Owns every SQL statement in the project; no other class writes raw queries.
Instantiated once at strategy startup and shared with StockDataCache and
BobsBrain. Stateless beyond the connection pool so it can be reused safely
across the strategy lifecycle.
"""

import json
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import pool


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
        """
        sql = """
            SELECT id, lead_symbol, lag_symbol, lag_days,
                   short_ma, long_ma, correlation
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
            pid, lead, lag, lag_days, short_ma, long_ma, corr = row
            result[lag] = {
                "pair_id":    pid,
                "lead_stock": lead,
                "lag_stock":  lag,
                "lag":        lag_days,
                "short_ma":   short_ma,
                "long_ma":    long_ma,
                "corr":       float(corr) if corr is not None else None,
                "action":     "hold",
            }
        return result

    def save_pair(self, pair: dict, run_id: str) -> int:
        """
        Insert a new pair row scoped to run_id. Returns the new pair id.
        Silently skips if the same (run_id, lead, lag, lag_days) already exists
        and returns the existing id instead.
        """
        sql = """
            INSERT INTO pairs
                (run_id, lead_symbol, lag_symbol, lag_days, short_ma, long_ma,
                 correlation, discovered_at, last_updated, active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            ON CONFLICT (run_id, lead_symbol, lag_symbol, lag_days)
                WHERE run_id IS NOT NULL
            DO NOTHING
            RETURNING id
        """
        today = date.today()
        with self._conn() as conn:
            with conn.cursor() as cur:
                corr = pair.get("corr")
                cur.execute(sql, (
                    run_id,
                    pair["lead_stock"],
                    pair["lag_stock"],
                    pair.get("lag", 1),
                    pair.get("short_ma", 2),
                    pair.get("long_ma", 5),
                    float(corr) if corr is not None else None,
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
            avg_correlation, cash_ratio, daily_buys, daily_sells
        """
        sql = """
            INSERT INTO portfolio_snapshots
                (time, run_id, portfolio_value, cash, spy_value,
                 active_pairs, avg_correlation, cash_ratio,
                 daily_buys, daily_sells)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
    ) -> None:
        """Insert one trade fill row."""
        sql = """
            INSERT INTO trades
                (run_id, pair_id, symbol, side, quantity, price, slippage, filled_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    run_id, pair_id, symbol, side,
                    quantity, price, slippage, filled_at,
                ))
