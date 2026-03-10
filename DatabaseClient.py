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
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
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
                if hasattr(df.columns, "levels"):
                    close = val
                    rows.append((pd.Timestamp(ts), symbol, None, None, None, close, None))
                else:
                    close = val
                    rows.append((pd.Timestamp(ts), symbol, None, None, None, close, None))

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

    def load_active_pairs(self) -> dict:
        """
        Return active pairs as a dict keyed by lag_symbol — the same shape
        that pair_history.json used — for drop-in compatibility with BobsBrain.
        """
        sql = """
            SELECT id, lead_symbol, lag_symbol, lag_days,
                   short_ma, long_ma, correlation
            FROM   pairs
            WHERE  active = TRUE
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
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

    def save_pair(self, pair: dict) -> int:
        """
        Upsert a pair row (matched on lead_symbol + lag_symbol + lag_days).
        Returns the pair id.
        """
        sql = """
            INSERT INTO pairs
                (lead_symbol, lag_symbol, lag_days, short_ma, long_ma,
                 correlation, discovered_at, last_updated, active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            ON CONFLICT DO NOTHING
            RETURNING id
        """
        today = date.today()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    pair["lead_stock"],
                    pair["lag_stock"],
                    pair.get("lag", 1),
                    pair.get("short_ma", 2),
                    pair.get("long_ma", 5),
                    pair.get("corr"),
                    today,
                    today,
                ))
                row = cur.fetchone()
                if row:
                    return row[0]

                # Row already exists — fetch its id
                cur.execute(
                    "SELECT id FROM pairs WHERE lead_symbol=%s AND lag_symbol=%s AND lag_days=%s",
                    (pair["lead_stock"], pair["lag_stock"], pair.get("lag", 1)),
                )
                result = cur.fetchone()
                return result[0] if result else -1

    def update_pair_correlation(self, pair_id: int, correlation: float) -> None:
        """Update the correlation and last_updated timestamp for an existing pair."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE pairs SET correlation=%s, last_updated=%s WHERE id=%s",
                    (correlation, date.today(), pair_id),
                )

    def deactivate_pair(self, lag_symbol: str) -> None:
        """Mark all active pairs for the given lag symbol as inactive."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE pairs SET active=FALSE WHERE lag_symbol=%s AND active=TRUE",
                    (lag_symbol,),
                )

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
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    time,
                    run_id,
                    metrics.get("portfolio_value"),
                    metrics.get("cash"),
                    metrics.get("spy_value"),
                    metrics.get("active_pairs"),
                    metrics.get("avg_correlation"),
                    metrics.get("cash_ratio"),
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
