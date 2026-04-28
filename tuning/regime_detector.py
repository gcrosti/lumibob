"""
regime_detector — classifies a date window into a market regime.

Regimes recognised
------------------
calm_bull   Low volatility, positive SPY trend.  Parameters tuned here should
            reduce entry threshold (z~1.5) and widen exits.
vol_shock   Very high volatility, sharp SPY drawdown.  Short windows, tight
            position sizing.
sideways    Moderate-to-high volatility, flat or negative SPY trend.
            The strategy's "natural habitat" for mean-reversion pairs.
trend_bull  Low-moderate volatility, sustained strong SPY uptrend.  Reduce
            position count / deployed % to limit long-only beta drag.

Feature definitions
-------------------
spy_vol_20d   Mean of rolling 20-day *annualised* realised vol of SPY over
              the window (stddev of log-returns × sqrt(252)).
spy_ret_20d   Mean of rolling 20-day forward return of SPY.
spy_ret_50d   Total SPY return from window start to end.
dispersion    Mean of rolling 20-day cross-sectional return dispersion
              (std dev of 1-day returns across all stocks in that day's
              stock_prices snapshot, then averaged over the window).

Calibration (reference windows)
--------------------------------
Window               spy_vol_20d  spy_ret_20d  regime
2017-01 → 2017-12    ~0.07        ~0.013       calm_bull
2020-02 → 2020-06    ~0.55        ~-0.010      vol_shock
2022-01 → 2022-12    ~0.23        ~-0.007      sideways
2023-04 → 2023-12    ~0.14        ~0.019       trend_bull
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_ALPACA_KEY    = os.getenv('ALPACA_API_KEY', '')
_ALPACA_SECRET = os.getenv('ALPACA_API_SECRET', '')


# ---------------------------------------------------------------------------
# Regime labels (canonical strings used as DB keys / lookup-table keys)
# ---------------------------------------------------------------------------

CALM_BULL   = 'calm_bull'
VOL_SHOCK   = 'vol_shock'
SIDEWAYS    = 'sideways'
TREND_BULL  = 'trend_bull'
UNKNOWN     = 'unknown'   # fallback when data is insufficient

ALL_REGIMES = (CALM_BULL, VOL_SHOCK, SIDEWAYS, TREND_BULL)


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeFeatures:
    spy_vol_20d: float      # annualised 20d realised vol (mean over window)
    spy_ret_20d: float      # rolling 20d return of SPY (mean over window)
    spy_ret_window: float   # total SPY return across the full window
    dispersion: float       # cross-sectional daily return dispersion (mean)
    n_spy_days: int         # number of SPY trading days found in DB

    def __str__(self) -> str:
        return (
            f'vol_20d={self.spy_vol_20d:.3f}  ret_20d={self.spy_ret_20d:+.4f}'
            f'  ret_window={self.spy_ret_window:+.4f}  disp={self.dispersion:.4f}'
            f'  ({self.n_spy_days} SPY days)'
        )


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Detects market regime for a date window using data already in stock_prices.

    Parameters
    ----------
    db_url : str
        PostgreSQL connection string.
    lookback_extra_days : int
        Calendar days before *start* to fetch for rolling-window warm-up.
        Must be at least 70 trading days to compute stable 20-day / 50-day
        rolling statistics.  Defaults to 110 calendar days (~75 trading days).
    """

    def __init__(
        self,
        db_url: str,
        lookback_extra_days: int = 110,
    ) -> None:
        self._db_url = db_url
        self._lookback_extra = lookback_extra_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_window(self, start: date, end: date) -> str:
        """
        Return the regime label for the window [*start*, *end*].

        Falls back to UNKNOWN if the DB has fewer than 15 SPY trading days
        in the requested window (data not yet pre-warmed).
        """
        features = self.get_features(start, end)
        if features is None:
            return UNKNOWN
        regime = self.label(features)
        logger.info(
            'Regime [%s → %s]: %s  |  %s',
            start, end, regime, features,
        )
        return regime

    def get_features(self, start: date, end: date) -> RegimeFeatures | None:
        """
        Compute regime features for [*start*, *end*] from stock_prices.

        Returns None if the DB has insufficient data for the window.
        """
        fetch_start = start - timedelta(days=self._lookback_extra)

        spy_prices = self._fetch_spy(fetch_start, end)
        if spy_prices is None or len(spy_prices) < 5:
            logger.warning(
                'RegimeDetector: insufficient SPY data for %s → %s (%d days fetched)',
                start, end, len(spy_prices) if spy_prices is not None else 0,
            )
            return None

        window_prices = spy_prices[spy_prices.index.date >= start]  # type: ignore[union-attr]
        if len(window_prices) < 5:
            # DB had lookback data but not the window itself — re-fetch window from Alpaca.
            logger.info(
                'RegimeDetector: window portion thin (%d days); retrying from Alpaca',
                len(window_prices),
            )
            spy_prices = self._fetch_spy_alpaca(start - timedelta(days=self._lookback_extra), end)
            if spy_prices is None or len(spy_prices) < 5:
                logger.warning('RegimeDetector: SPY data too thin after Alpaca retry')
                return None
            window_prices = spy_prices[spy_prices.index.date >= start]  # type: ignore[union-attr]
            if len(window_prices) < 5:
                logger.warning('RegimeDetector: window still thin after retry')
                return None

        # Restrict cross-sectional data to the main window (not lookback).
        disp = self._fetch_dispersion(start, end)

        log_ret = np.log(spy_prices / spy_prices.shift(1)).dropna()

        # 20-day rolling annualised vol (mean over window).
        roll_vol = log_ret.rolling(20).std() * np.sqrt(252)
        window_vol = roll_vol[roll_vol.index.date >= start]  # type: ignore[union-attr]
        spy_vol_20d = float(window_vol.dropna().mean()) if len(window_vol.dropna()) > 0 else 0.0

        # 20-day rolling forward return (mean over window).
        roll_ret = spy_prices.pct_change(20)
        window_roll_ret = roll_ret[roll_ret.index.date >= start]  # type: ignore[union-attr]
        spy_ret_20d = float(window_roll_ret.dropna().mean()) if len(window_roll_ret.dropna()) > 0 else 0.0

        # Total SPY return from window start to end.
        spy_ret_window = float(window_prices.iloc[-1] / window_prices.iloc[0] - 1)

        return RegimeFeatures(
            spy_vol_20d=spy_vol_20d,
            spy_ret_20d=spy_ret_20d,
            spy_ret_window=spy_ret_window,
            dispersion=disp,
            n_spy_days=len(window_prices),
        )

    @staticmethod
    def label(f: RegimeFeatures) -> str:
        """
        Map a RegimeFeatures snapshot to a regime label using simple
        threshold rules calibrated against the Phase 3 reference windows.

        Threshold table (annualised spy_vol_20d):
            vol_shock  : spy_vol_20d ≥ 0.30
            calm_bull  : spy_vol_20d < 0.13 AND spy_ret_20d ≥ 0.0
            trend_bull : spy_vol_20d < 0.18 AND spy_ret_20d ≥ 0.005
            sideways   : everything else
        """
        v = f.spy_vol_20d
        r = f.spy_ret_20d

        if v >= 0.30:
            return VOL_SHOCK
        if v < 0.13 and r >= 0.0:
            return CALM_BULL
        if v < 0.18 and r >= 0.005:
            return TREND_BULL
        return SIDEWAYS

    # ------------------------------------------------------------------
    # Internal data fetchers
    # ------------------------------------------------------------------

    def _fetch_spy(self, start: date, end: date) -> pd.Series | None:
        """
        Return daily close prices for SPY.

        Tries stock_prices table first; falls back to Alpaca API if SPY is not
        stored there (SPY is not in the trading universe tickers table).
        """
        # --- Try DB first ---
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt   = datetime.combine(end,   datetime.min.time())
        try:
            with psycopg2.connect(self._db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT time::date AS d, close
                        FROM stock_prices
                        WHERE symbol = 'SPY'
                          AND time >= %s AND time <= %s
                        ORDER BY d
                        """,
                        (start_dt, end_dt),
                    )
                    rows = cur.fetchall()
            if rows:
                idx = pd.to_datetime([r[0] for r in rows])
                return pd.Series([float(r[1]) for r in rows], index=idx, name='SPY')
        except Exception:
            logger.warning('RegimeDetector: DB SPY fetch failed; trying Alpaca')

        return self._fetch_spy_alpaca(start, end)

    def _fetch_spy_alpaca(self, start: date, end: date) -> pd.Series | None:
        """Fetch SPY via Alpaca and persist to stock_prices."""
        if not _ALPACA_KEY:
            logger.warning('RegimeDetector: no Alpaca credentials; cannot fetch SPY')
            return None
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt   = datetime.combine(end,   datetime.min.time())
        try:
            from AlpacaClient import AlpacaClient
            client = AlpacaClient(_ALPACA_KEY, _ALPACA_SECRET)
            bars = client.get_historical_bars(['SPY'], start_dt, end_dt)
            if bars.empty or 'SPY' not in bars.columns:
                logger.warning('RegimeDetector: Alpaca returned no SPY bars')
                return None
            spy_series = bars['SPY'].dropna()
            spy_series.index = pd.to_datetime(spy_series.index)
            self._store_spy(bars)
            return spy_series
        except Exception:
            logger.exception('RegimeDetector: Alpaca SPY fetch failed')
            return None

    def _store_spy(self, bars: pd.DataFrame) -> None:
        """
        Write SPY bars fetched from Alpaca back into stock_prices so that
        subsequent calls are served from the DB cache.
        """
        if bars.empty or 'SPY' not in bars.columns:
            return
        spy_col = bars['SPY'].dropna()
        rows = [
            ('SPY', pd.Timestamp(idx).to_pydatetime(), float(val), float(val),
             float(val), float(val), 0)
            for idx, val in spy_col.items()
        ]
        if not rows:
            return
        try:
            with psycopg2.connect(self._db_url) as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO stock_prices (symbol, time, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, time) DO NOTHING
                        """,
                        rows,
                    )
                conn.commit()
            logger.info('RegimeDetector: stored %d SPY rows in stock_prices', len(rows))
        except Exception:
            logger.warning('RegimeDetector: could not persist SPY bars to DB')

    def _fetch_dispersion(self, start: date, end: date) -> float:
        """
        Compute mean cross-sectional daily return dispersion over [start, end].

        Dispersion = std dev of all stocks' 1-day returns on each trading day,
        then averaged over the window.  Returns 0.0 on failure.
        """
        # Extend start by 1 day to allow LAG to produce returns on start itself.
        fetch_start = start - timedelta(days=3)
        start_dt = datetime.combine(fetch_start, datetime.min.time())
        end_dt   = datetime.combine(end,         datetime.min.time())
        try:
            with psycopg2.connect(self._db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        WITH daily_ret AS (
                            SELECT time::date AS d,
                                   symbol,
                                   close / NULLIF(
                                       LAG(close) OVER (PARTITION BY symbol ORDER BY time),
                                       0
                                   ) - 1.0 AS ret
                            FROM stock_prices
                            WHERE time >= %s AND time <= %s
                        )
                        SELECT d, STDDEV(ret) AS daily_disp
                        FROM daily_ret
                        WHERE ret IS NOT NULL
                          AND d >= %s
                        GROUP BY d
                        HAVING COUNT(ret) >= 50
                        ORDER BY d
                        """,
                        (start_dt, end_dt, start),
                    )
                    rows = cur.fetchall()
        except Exception:
            logger.warning('RegimeDetector: dispersion query failed, returning 0.0')
            return 0.0

        values = [float(r[1]) for r in rows if r[1] is not None]
        return float(np.mean(values)) if values else 0.0
