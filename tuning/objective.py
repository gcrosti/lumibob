"""
objective — BacktestObjective for Optuna.

Two scoring modes, selected via *discriminatory_weight* (0.0–1.0):

    Pure Sharpe (discriminatory_weight=0.0, default — Studies 2 and 3):

        score = Sharpe(daily) - penalty_dd × max_drawdown
                - penalty_trades × log(n+1)
                - spy_penalty_weight × (2.0 if return ≤ SPY else 0.0)

    Blended discriminatory (discriminatory_weight > 0 — Study 1 / Tier 2):

        score = w × spearman_rho(composite_score, round_trip_pnl)
                + (1−w) × sharpe_component

        where sharpe_component is the Sharpe score normalised to [−1, 1]
        so the two terms are on a comparable scale.

        A hard floor is applied first: if mean round-trip P&L falls below
        *pnl_floor*, a heavy penalty is returned regardless of rho.  This
        prevents the optimizer finding a perfectly discriminating set of
        pairs that all lose money.

*spy_penalty_weight* controls the SPY hard-constraint strength:
  - 1.0 (default, Phases 1–3): full -2.0 penalty if strategy ≤ SPY.
  - 0.0 (Phase 4+): SPY constraint lifted; goal is positive Sharpe.

*trial_timeout_secs* (default None): if the backtest subprocess exceeds
this wall-clock budget, the trial is pruned rather than blocking the study.
Set to 1200 (20 min) for Phase 4 coarse to guard against pathological runs.

Scoring uses the portfolio_snapshots, pairs, and trades tables written by
BobsBrain during the run.  The run_id is recovered via a unique
tuning_trial_token injected into the backtest's parameters and written by
BobsBrain into backtest_runs.settings — exact attribution that is safe under
parallel workers (the previous most-recent-run timestamp heuristic
cross-attributed runs when concurrent workers finished out of order; caught
by the Study 00 cloud pipe check, 2026-07-14).
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
    param_names : frozenset[str] | None
        Optional allowlist restricting which params within *tiers* are
        suggested.  None (default) frees every param in the tiers.
    penalty_dd : float
        λ — weight on max-drawdown penalty in the composite score.
    penalty_trades : float
        μ — weight on log-trade-count penalty (guards against overtrading).
    min_trades : int
        Runs with fewer than this many fills return a heavy penalty instead
        of being scored normally (catches extreme underdeployment).
    spy_penalty_weight : float
        Multiplier on the -2.0 SPY hard-constraint penalty.  Set to 0.0 to
        remove the SPY constraint entirely (Phase 4 goal: positive Sharpe).
    trial_timeout_secs : int | None
        If set, a trial whose backtest exceeds this many seconds is pruned.
        Prevents pathological cold-cache runs from blocking the study.
    discriminatory_weight : float
        w in [0.0, 1.0].  0.0 = pure Sharpe (default, backward compatible).
        > 0 blends Spearman rank correlation between composite_score and
        round-trip P&L into the objective.  Recommended value for Study 1
        (Tier 2 signal construction): 0.7.
    pnl_floor : float
        Mean round-trip P&L floor for discriminatory scoring.  If the run's
        mean P&L falls below this value the trial is penalised regardless of
        how well the score discriminates.  Prevents optimizing a beautifully
        discriminating set of pairs that all lose money.  Default -100.
    min_round_trips : int
        Minimum matched buy/sell round-trips needed for a valid discriminatory
        score.  Runs below this threshold return a heavy penalty.  Default 10.
    price_cache_only : bool
        Default True: trial backtests read prices from the DB cache only and
        never call Alpaca (the cache is fully backfilled; live fetches during
        a study add only rate-limit exposure and failure-marking risk).  Set
        False only for a deliberate cache-warming run.
    """

    def __init__(
        self,
        train_start: date,
        train_end: date,
        budget: float = 10_000,
        base_params: dict[str, Any] | None = None,
        tiers: tuple[int, ...] = (2,),
        param_names: frozenset[str] | None = None,
        penalty_dd: float = 0.5,
        penalty_trades: float = 0.01,
        min_trades: int = 5,
        spy_penalty_weight: float = 1.0,
        trial_timeout_secs: int | None = None,
        discriminatory_weight: float = 0.0,
        pnl_floor: float = -100.0,
        min_round_trips: int = 10,
        price_cache_only: bool = True,
    ) -> None:
        self.train_start = datetime(train_start.year, train_start.month, train_start.day)
        self.train_end = datetime(train_end.year, train_end.month, train_end.day)
        self.budget = budget
        self.base_params = base_params if base_params is not None else defaults()
        self.tiers = tiers
        self.param_names = param_names
        self.penalty_dd = penalty_dd
        self.penalty_trades = penalty_trades
        self.min_trades = min_trades
        self.spy_penalty_weight = spy_penalty_weight
        self.trial_timeout_secs = trial_timeout_secs
        self.discriminatory_weight = discriminatory_weight
        self.pnl_floor = pnl_floor
        self.min_round_trips = min_round_trips
        self.price_cache_only = price_cache_only

    # ------------------------------------------------------------------
    # Optuna interface
    # ------------------------------------------------------------------

    def __call__(self, trial: optuna.Trial) -> float:
        from tuning.parameter_space import suggest

        trial_params = suggest(trial, self.tiers, param_names=self.param_names)
        full_params = {**self.base_params, **trial_params}

        logger.info('Trial %d starting — suggested: %s', trial.number, trial_params)

        run_id = self._run_backtest(full_params)

        if run_id is None:
            logger.error('Trial %d: backtest failed, pruning', trial.number)
            raise optuna.exceptions.TrialPruned()

        trial.set_user_attr('run_id', run_id)
        score = self.score_run(run_id, trial=trial)
        logger.info('Trial %d  score=%.4f  run_id=%s', trial.number, score, run_id)
        return score

    # ------------------------------------------------------------------
    # Backtest runner
    # ------------------------------------------------------------------

    def _run_backtest(self, params: dict[str, Any]) -> str | None:
        """
        Run BobsBrain.backtest() with *params* and return the run_id written
        to the DB, or None if the backtest raises an exception or times out.

        When *trial_timeout_secs* is set, the backtest is run in a subprocess
        via ``subprocess.run(..., timeout=...)``.  This guarantees an OS-level
        SIGKILL after the deadline even if the backtest is blocked inside a
        C-extension or network call (threading.Timer cannot do this).
        """
        start_ts = datetime.now(timezone.utc)

        # Unique token written into backtest_runs.settings by BobsBrain so
        # this worker recovers exactly its own run.  Timestamp-based recovery
        # cross-attributes runs when multiple workers finish out of order.
        import uuid
        token = uuid.uuid4().hex
        # price_cache_only: trials never touch Alpaca — the cache is fully
        # backfilled, and live fetches during a study only add rate-limit
        # exposure and failure-marking risk. Set False only for a deliberate
        # cache-warming run.
        params = {
            **params,
            'tuning_trial_token': token,
            'price_cache_only': self.price_cache_only,
        }

        if self.trial_timeout_secs is not None:
            return self._run_backtest_subprocess(params, start_ts, token)

        from lumibot.backtesting import YahooDataBacktesting
        from BobsBrain import BobsBrain

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

        return self._find_run_id(after_ts=start_ts, token=token)

    def _run_backtest_subprocess(
        self,
        params: dict[str, Any],
        start_ts: datetime,
        token: str,
    ) -> str | None:
        """
        Run BobsBrain.backtest in a child process so the OS-level timeout
        (SIGKILL) fires reliably regardless of what the backtest is blocked on.
        """
        import json
        import subprocess
        import sys
        import tempfile

        # Serialise params to a temp file; the child reads and runs the backtest.
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as fh:
            json.dump(
                {
                    'train_start': self.train_start.isoformat(),
                    'train_end':   self.train_end.isoformat(),
                    'budget':      self.budget,
                    'params':      params,
                },
                fh,
            )
            param_file = fh.name

        script = (
            'import json, sys\n'
            'from lumibot.backtesting import YahooDataBacktesting\n'
            'from BobsBrain import BobsBrain\n'
            'from datetime import datetime\n'
            'from dotenv import load_dotenv; load_dotenv()\n'
            f'd = json.load(open({param_file!r}))\n'
            'BobsBrain.backtest(\n'
            '    YahooDataBacktesting,\n'
            '    datetime.fromisoformat(d["train_start"]),\n'
            '    datetime.fromisoformat(d["train_end"]),\n'
            '    budget=d["budget"],\n'
            '    parameters=d["params"],\n'
            '    show_plot=False, show_tearsheet=False, save_tearsheet=False,\n'
            ')\n'
        )

        try:
            result = subprocess.run(
                [sys.executable, '-c', script],
                timeout=self.trial_timeout_secs,
                capture_output=False,
            )
            if result.returncode != 0:
                logger.warning('Backtest subprocess exited with code %d', result.returncode)
                return None
        except subprocess.TimeoutExpired:
            logger.warning(
                'Trial timed out after %d s (subprocess killed) — pruning',
                self.trial_timeout_secs,
            )
            return None
        except Exception:
            logger.exception('Backtest subprocess raised')
            return None
        finally:
            import os as _os
            try:
                _os.unlink(param_file)
            except OSError:
                pass

        return self._find_run_id(after_ts=start_ts, token=token)

    @staticmethod
    def _find_run_id(after_ts: datetime, token: str | None = None) -> str | None:
        """
        Recover the run_id this trial's backtest wrote to backtest_runs.

        With *token* (always set on the tuning path): match the token BobsBrain
        wrote into settings — exact attribution, safe under parallel workers.
        No fallback to the timestamp heuristic: a missing token means the
        backtest did not complete a run, and silently scoring some other
        worker's run would corrupt the study.

        Without *token*: legacy most-recent-run heuristic (single-worker only).
        """
        with psycopg2.connect(_DB_URL) as conn:
            with conn.cursor() as cur:
                if token is not None:
                    cur.execute(
                        """
                        SELECT run_id
                        FROM backtest_runs
                        WHERE settings->>'tuning_trial_token' = %s
                          AND completed_at IS NOT NULL
                        LIMIT 1
                        """,
                        (token,),
                    )
                else:
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
        if self.spy_penalty_weight > 0.0 and not spy.isna().all() and spy.iloc[0] > 0:
            port_total = pv.iloc[-1] / pv.iloc[0] - 1
            spy_total = spy.iloc[-1] / spy.iloc[0] - 1
            if port_total <= spy_total:
                spy_penalty = -2.0 * self.spy_penalty_weight

        sharpe_score = sharpe - self.penalty_dd * max_dd - trade_penalty + spy_penalty

        if self.discriminatory_weight <= 0.0:
            score = sharpe_score
        else:
            rho = self._discriminatory_score(run_id)
            if rho is None:
                # Insufficient data for discriminatory scoring — fall back to Sharpe.
                score = sharpe_score
            else:
                # Normalise Sharpe to [-1, 1] (divide by 3 — a generous ceiling
                # for this strategy) so the two terms are on a comparable scale.
                sharpe_norm = float(np.clip(sharpe_score / 3.0, -1.0, 1.0))
                w = self.discriminatory_weight
                score = w * rho + (1.0 - w) * sharpe_norm

        if trial is not None:
            trial.report(score, step=0)

        return score

    def _discriminatory_score(self, run_id: str) -> float | None:
        """
        Compute Spearman rank correlation between composite_score at entry and
        round-trip P&L for long-leg matched trades in *run_id*.

        Returns the correlation coefficient in [-1, 1], or None if there are
        fewer than *min_round_trips* matched pairs or no composite_score data.

        A hard floor on mean P&L is applied: if the mean falls below
        *pnl_floor*, returns -1.0 (worst possible discriminatory score) to
        penalise a trial that finds a discriminating but losing configuration.
        """
        sql = """
            SELECT p.composite_score,
                   SUM(CASE WHEN t.side = 'sell'
                            THEN t.quantity * t.price - t.slippage
                            ELSE 0 END)
                 - SUM(CASE WHEN t.side = 'buy'
                            THEN t.quantity * t.price + t.slippage
                            ELSE 0 END) AS round_trip_pnl
            FROM pairs p
            JOIN trades t
              ON t.pair_id = p.id AND t.leg = 'long'
            WHERE p.run_id = %s
              AND p.composite_score IS NOT NULL
            GROUP BY p.id, p.composite_score
            HAVING COUNT(DISTINCT CASE WHEN t.side = 'sell' THEN t.id END) > 0
               AND COUNT(DISTINCT CASE WHEN t.side = 'buy'  THEN t.id END) > 0
        """
        with psycopg2.connect(_DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (run_id,))
                rows = cur.fetchall()

        if len(rows) < self.min_round_trips:
            logger.warning(
                'run_id=%s: only %d round-trips for discriminatory score (need %d)',
                run_id, len(rows), self.min_round_trips,
            )
            return None

        scores = np.array([float(r[0]) for r in rows])
        pnls   = np.array([float(r[1]) for r in rows])

        mean_pnl = float(pnls.mean())
        if mean_pnl < self.pnl_floor:
            logger.warning(
                'run_id=%s: mean P&L %.2f below floor %.2f — penalising',
                run_id, mean_pnl, self.pnl_floor,
            )
            return -1.0

        # Spearman rho via rank correlation.
        score_ranks = pd.Series(scores).rank()
        pnl_ranks   = pd.Series(pnls).rank()
        rho = float(score_ranks.corr(pnl_ranks))
        logger.info('run_id=%s: discriminatory rho=%.4f  mean_pnl=%.2f', run_id, rho, mean_pnl)
        return rho
