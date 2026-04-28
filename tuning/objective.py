"""
objective — BacktestObjective for Optuna.

Wraps a BobsBrain backtest and converts the results into a scalar score:

    score = Sharpe(daily) - penalty_dd × max_drawdown - penalty_trades × log(n+1)
            ± spy_penalty

Hard constraint: if portfolio total return ≤ SPY total return over the
backtest window, a fixed penalty of -2.0 is added.  This is cheaper than
pruning the trial, which would exclude it from Optuna's median statistics.

Scoring uses the portfolio_snapshots and trades tables written by BobsBrain
during the run.  The run_id is recovered by querying backtest_runs for the
most recently completed run started after this call's invocation timestamp.
This is safe for single-threaded (n_jobs=1) Optuna studies; parallel workers
must use a DB-level mechanism (e.g. passing a deterministic run_id) instead.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import optuna
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from tuning.parameter_space import defaults

load_dotenv()

logger = logging.getLogger(__name__)

_DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')


class BacktestObjective:
    """
    Callable Optuna objective that runs a BobsBrain backtest and scores it.

    Parameters
    ----------
    train_start, train_end : date
        Inclusive backtest window passed to BobsBrain.backtest().
    budget : float
        Starting cash for each trial (default 10 000).
    base_params : dict | None
        Fixed parameters merged before trial suggestions.  Defaults to the
        full canonical default set from parameter_space.defaults().
    tiers : tuple[int, ...]
        Which parameter tiers the trial should suggest values for.
    penalty_dd : float
        λ — weight on max-drawdown penalty in the composite score.
    penalty_trades : float
        μ — weight on log-trade-count penalty (guards against overtrading).
    min_trades : int
        Runs with fewer than this many fills return a heavy penalty instead
        of being scored normally (catches extreme underdeployment).
    """

    def __init__(
        self,
        train_start: date,
        train_end: date,
        budget: float = 10_000,
        base_params: dict[str, Any] | None = None,
        tiers: tuple[int, ...] = (2,),
        penalty_dd: float = 0.5,
        penalty_trades: float = 0.01,
        min_trades: int = 5,
    ) -> None:
        self.train_start = datetime(train_start.year, train_start.month, train_start.day)
        self.train_end = datetime(train_end.year, train_end.month, train_end.day)
        self.budget = budget
        self.base_params = base_params if base_params is not None else defaults()
        self.tiers = tiers
        self.penalty_dd = penalty_dd
        self.penalty_trades = penalty_trades
        self.min_trades = min_trades

    # ------------------------------------------------------------------
    # Optuna interface
    # ------------------------------------------------------------------

    def __call__(self, trial: optuna.Trial) -> float:
        from tuning.parameter_space import suggest

        trial_params = suggest(trial, self.tiers)
        full_params = {**self.base_params, **trial_params}

        logger.info('Trial %d starting — suggested: %s', trial.number, trial_params)

        run_id = self._run_backtest(full_params)

        if run_id is None:
            logger.error('Trial %d: backtest failed, pruning', trial.number)
            raise optuna.exceptions.TrialPruned()

        score = self.score_run(run_id, trial=trial)
        logger.info('Trial %d  score=%.4f  run_id=%s', trial.number, score, run_id)
        return score

    # ------------------------------------------------------------------
    # Backtest runner
    # ------------------------------------------------------------------

    def _run_backtest(self, params: dict[str, Any]) -> str | None:
        """
        Run BobsBrain.backtest() with *params* and return the run_id written
        to the DB, or None if the backtest raises an exception.
        """
        from lumibot.backtesting import YahooDataBacktesting
        from BobsBrain import BobsBrain

        start_ts = datetime.now(timezone.utc)
        try:
            BobsBrain.backtest(
                YahooDataBacktesting,
                self.train_start,
                self.train_end,
                budget=self.budget,
                parameters=params,
                show_plot=False,
                show_tearsheet=False,
                save_tearsheet=False,
            )
        except Exception:
            logger.exception('BobsBrain.backtest raised an exception')
            return None

        return self._find_run_id(after_ts=start_ts)

    @staticmethod
    def _find_run_id(after_ts: datetime) -> str | None:
        """Query the DB for the most recently completed run started after *after_ts*."""
        with psycopg2.connect(_DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id
                    FROM backtest_runs
                    WHERE started_at >= %s
                      AND mode = 'backtest'
                      AND completed_at IS NOT NULL
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (after_ts,),
                )
                row = cur.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_run(self, run_id: str, trial: optuna.Trial | None = None) -> float:
        """
        Compute the composite score for an already-completed run.

            score = Sharpe - penalty_dd × max_DD - penalty_trades × log(n_trades + 1)
                    + spy_penalty

        *trial* is optional — when provided, the score is reported to Optuna
        for the MedianPruner.  Pass None when scoring outside a study
        (e.g. baseline comparison).

        Returns -999.0 if the run produced no usable portfolio data.
        """
        with psycopg2.connect(_DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT time, portfolio_value, spy_value
                    FROM portfolio_snapshots
                    WHERE run_id = %s
                    ORDER BY time
                    """,
                    (run_id,),
                )
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                cur.execute(
                    "SELECT COUNT(*) FROM trades WHERE run_id = %s",
                    (run_id,),
                )
                n_trades: int = cur.fetchone()[0]

        snapshots = pd.DataFrame(rows, columns=cols)

        if snapshots.empty or len(snapshots) < 3:
            logger.warning('run_id=%s: too few snapshots (%d)', run_id, len(snapshots))
            return -999.0

        pv = snapshots['portfolio_value'].astype(float)
        spy = snapshots['spy_value'].astype(float)

        # Catch extreme underdeployment (strategy barely traded).
        if n_trades < self.min_trades:
            logger.warning('run_id=%s: only %d trades — heavy penalty', run_id, n_trades)
            return -50.0

        # Catch frozen portfolio (pricing / data issue).
        if pv.std() < 1e-6:
            logger.warning('run_id=%s: portfolio value never changed', run_id)
            return -999.0

        # --- Sharpe (annualised daily) ---
        daily_ret = pv.pct_change().dropna()
        if daily_ret.std() < 1e-9:
            return -999.0
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))

        # --- Max drawdown (positive fraction) ---
        roll_max = pv.cummax()
        drawdown = ((pv - roll_max) / roll_max).min()
        max_dd = abs(float(drawdown))

        # --- Trade-count penalty ---
        trade_penalty = float(np.log1p(n_trades)) * self.penalty_trades

        # --- SPY hard constraint (soft implementation via additive penalty) ---
        spy_penalty = 0.0
        if not spy.isna().all() and spy.iloc[0] > 0:
            port_total = pv.iloc[-1] / pv.iloc[0] - 1
            spy_total = spy.iloc[-1] / spy.iloc[0] - 1
            if port_total <= spy_total:
                spy_penalty = -2.0

        score = sharpe - self.penalty_dd * max_dd - trade_penalty + spy_penalty

        if trial is not None:
            trial.report(score, step=0)

        return score
