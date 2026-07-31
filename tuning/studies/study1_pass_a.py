"""
study1_pass_a — Tier 2 signal construction (timescales + weights).

Purpose
-------
Find timescales and composite score weights that make the composite score
a reliable ranking signal — i.e. high-scoring pairs should outperform
low-scoring pairs.  This is a prerequisite for regime-conditioned Tier 3
tuning: there is no point conditioning a regime on a score that does not
discriminate.

Objective
---------
Blended discriminatory (discriminatory_weight=0.7):

    score = 0.7 × Spearman_rho(composite_score_at_entry, round_trip_pnl)
          + 0.3 × clip(sharpe / 3, -1, 1)

    subject to: mean(round_trip_pnl) > pnl_floor (default -100)
                n_round_trips ≥ 10

Pure Sharpe is not used here because it conflates score quality with
regime luck.  A well-discriminating score in a losing regime produces bad
Sharpe; the optimizer would discard those params even if the score is
doing exactly the right thing.

Free parameters (10 signal-construction params — PASS_A_PARAMS)
---------------------------------------
    lookback_window    [60, 252]
    zscore_window      [10, min(40, lookback_window // 3)]   — constrained
    cooldown_days      [max(3, zscore_window // 2), 21]      — constrained
    w_corr_long        [0, 1]  (normalised to sum-to-1 with other weights)
    w_corr_short       [0, 1]
    w_z_depth          [0, 1]
    w_coint            [0.1, 1]
    w_halflife         [0, 1]
    corr_long_window   [45, 252]
    corr_short_window  [10, min(40, corr_long_window)]       — constrained

Training window
---------------
Three 3-month folds rotated round-robin across trials, covering sideways,
bull, and mixed regimes.  Each trial runs on one fold; TPE builds a shared
distribution that generalises across regime types.  This replaces the
original single 30-month window (which would cost ~300 min/trial).

Gate criterion
--------------
Best-trial Spearman rho > 0.15 in at least 2 of 3 OOS folds.
Gate is evaluated AFTER the study completes.  If fewer than 2 folds pass,
investigate root cause before running Study 1 Pass B.

Usage
-----
    # Single worker:
    RUN_MODE=backtest python -m tuning.studies.study1_pass_a

    # Parallel workers (share the same Optuna study via PostgreSQL):
    for i in 1 2 3 4; do
      RUN_MODE=backtest python -m tuning.studies.study1_pass_a \\
        >> /tmp/study1a_w${i}.log 2>&1 &
    done
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

import optuna
from dotenv import load_dotenv

from tuning.objective import BacktestObjective
from tuning.parameter_space import defaults, normalize_weights

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s — %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUDGET = float(os.getenv('TUNE_BUDGET', '10000'))
N_TRIALS = int(os.getenv('TUNE_N_TRIALS', '90'))   # ~30 per fold across 3 folds
TIMEOUT_HOURS = float(os.getenv('TUNE_TIMEOUT_HOURS', '24'))
STUDY_NAME = os.getenv('TUNE_STUDY_NAME', 'study1_pass_a_v2')
DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')

# Discriminatory objective weight (plan: 0.7).
DISCRIMINATORY_WEIGHT = float(os.getenv('TUNE_DISC_WEIGHT', '0.7'))

# Three 5-month training folds — rotated round-robin across trials so TPE
# sees all regimes.  Extended from 3→5 months (~65→108 trading days) so each
# fold generates enough round-trips for reliable Spearman rho estimation.
# Parameter space (zscore_window, cooldown_days, max_k) is unchanged from v1;
# only the window length changes to improve statistical power.
TRAIN_FOLDS = [
    (date(2021, 12, 1), date(2022, 4, 30)),   # sideways / bear (~108 days)
    (date(2023, 2, 1),  date(2023, 6, 30)),   # bull (~107 days)
    (date(2023, 7, 1),  date(2023, 11, 30)),  # mixed Q3/Q4 (~109 days)
]

# OOS folds for post-study gate check (same windows as training folds).
GATE_FOLDS: list[tuple[str, date, date]] = [
    ('sideways_2022',  date(2021, 12, 1), date(2022, 4, 30)),
    ('bull_2023',      date(2023, 2, 1),  date(2023, 6, 30)),
    ('mixed_2023_q4',  date(2023, 7, 1),  date(2023, 11, 30)),
]
_GATE_RHO_THRESHOLD = 0.15
_GATE_MIN_PASSING_FOLDS = 2

# The signal-construction params this study frees (see module docstring).
# Tier 2 also contains discovery/sizing params (max_k, target_deployed_pct,
# max_daily_candidates, HDBSCAN params, ...) — those stay at defaults here
# and are optimized in Pass B on top of this study's output.
# w_coint / w_halflife were removed from the parameter space with the composite
# score cleanup (PR #50) — any re-run of this study frees 8 params, not the
# original 10, and per the optuna-study skill would need a version bump.
PASS_A_PARAMS = frozenset({
    'lookback_window', 'zscore_window', 'cooldown_days',
    'w_corr_long', 'w_corr_short', 'w_z_depth',
    'corr_long_window', 'corr_short_window',
})


# ---------------------------------------------------------------------------
# Fold-rotating objective
# ---------------------------------------------------------------------------

class FoldRotatingObjective:
    """
    Wraps BacktestObjective to rotate through TRAIN_FOLDS round-robin.

    Each trial runs on the fold corresponding to trial.number % len(folds).
    This ensures TPE sees a balanced mix of regimes and learns parameters
    that generalise rather than overfitting to a single window.

    Parallel workers share the same Optuna study (via PostgreSQL RDB storage)
    and will naturally distribute across folds since trial numbers are
    assigned sequentially by Optuna.
    """

    def __init__(self, folds: list[tuple[date, date]], **objective_kwargs: Any) -> None:
        self._folds = folds
        self._kwargs = objective_kwargs

    def __call__(self, trial: optuna.Trial) -> float:
        train_start, train_end = self._folds[trial.number % len(self._folds)]
        trial.set_user_attr('fold_start', train_start.isoformat())
        trial.set_user_attr('fold_end', train_end.isoformat())
        obj = BacktestObjective(
            train_start=train_start,
            train_end=train_end,
            **self._kwargs,
        )
        return obj(trial)


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run() -> optuna.Study:
    """
    Run (or resume) Study 1 Pass A.

    Returns the Optuna Study object so callers can inspect best_params and
    best_value programmatically.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = optuna.storages.RDBStorage(
        url=DB_URL,
        engine_kwargs={'pool_pre_ping': True, 'pool_size': 1},
        skip_compatibility_check=False,
    )

    # Base: all defaults.  Tier 2 params overridden by suggest().
    # short_leg_fraction=0.0 → long-only (keeps trial times uniform).
    base = defaults()
    base['short_leg_fraction'] = 0.0

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    objective = FoldRotatingObjective(
        folds=TRAIN_FOLDS,
        budget=BUDGET,
        base_params=base,
        tiers=(2,),
        param_names=PASS_A_PARAMS,
        spy_penalty_weight=0.0,
        discriminatory_weight=DISCRIMINATORY_WEIGHT,
        pnl_floor=-100.0,
        min_round_trips=25,
        trial_timeout_secs=10800,        # 3 hr hard kill per trial (5-month window ~120 min; 50% headroom)
    )

    n_existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(
        'Study "%s" — %d completed trials so far  target=%d  '
        'timeout=%.1fh  disc_weight=%.2f  folds=%d',
        STUDY_NAME, n_existing, N_TRIALS,
        TIMEOUT_HOURS, DISCRIMINATORY_WEIGHT, len(TRAIN_FOLDS),
    )
    logger.info('Folds: %s', [(str(s), str(e)) for s, e in TRAIN_FOLDS])

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_HOURS * 3600,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    _log_results(study)
    _run_gate(study, base)

    return study


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_results(study: optuna.Study) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    logger.info(
        'Study complete — %d completed, %d pruned, %d failed',
        len(completed), len(pruned), len(failed),
    )

    if not completed:
        logger.warning('No completed trials — cannot report best params')
        return

    best_params = normalize_weights(study.best_params)
    logger.info('Best value  : %.4f', study.best_value)
    best_fold = study.best_trial.user_attrs.get('fold_start', 'unknown')
    logger.info('Best trial fold: %s', best_fold)
    logger.info('Best params (normalised weights):')
    for k, v in sorted(best_params.items()):
        logger.info('  %-30s %s', k, v)


def _run_gate(study: optuna.Study, base_params: dict) -> None:
    """
    Gate: run best-trial params on each OOS fold, check Spearman rho > 0.15
    in >= 2 of 3 folds.  Prints PASS/FAIL and the recommended next step.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning('Gate check skipped — no completed trials')
        return

    best_params = normalize_weights({**base_params, **study.best_params})

    logger.info('--- Study 1 Pass A Gate: OOS Spearman rho ---')
    fold_rhos: dict[str, float | None] = {}

    for fold_name, fold_start, fold_end in GATE_FOLDS:
        logger.info('  Fold "%s"  %s → %s', fold_name, fold_start, fold_end)
        gate_obj = BacktestObjective(
            train_start=fold_start,
            train_end=fold_end,
            budget=BUDGET,
            base_params=best_params,
            tiers=(),           # all params fixed
            spy_penalty_weight=0.0,
            discriminatory_weight=0.0,
            min_round_trips=25,
            trial_timeout_secs=10800,
        )
        run_id = gate_obj._run_backtest(best_params)
        if run_id is None:
            logger.warning('  [%s] backtest failed', fold_name)
            fold_rhos[fold_name] = None
            continue
        rho = gate_obj._discriminatory_score(run_id)
        fold_rhos[fold_name] = rho
        logger.info('  [%s] rho=%s', fold_name,
                    f'{rho:.4f}' if rho is not None else 'N/A (insufficient round-trips)')

    passing = [fn for fn, rho in fold_rhos.items()
               if rho is not None and rho > _GATE_RHO_THRESHOLD]
    verdict = 'PASS' if len(passing) >= _GATE_MIN_PASSING_FOLDS else 'FAIL'

    logger.info('Gate: %d/%d folds pass rho>%.2f — %s',
                len(passing), len(GATE_FOLDS), _GATE_RHO_THRESHOLD, verdict)

    if verdict == 'PASS':
        logger.info('NEXT STEP: Study 1 Pass B (tuning.studies.study1_pass_b)')
    else:
        close = sum(1 for r in fold_rhos.values()
                    if r is not None and r > _GATE_RHO_THRESHOLD * 0.5)
        if close >= _GATE_MIN_PASSING_FOLDS:
            logger.warning(
                'GATE NARROWLY FAILED — rho borderline in multiple folds. '
                'Consider 150 total trials before concluding failure.'
            )
        else:
            logger.error(
                'GATE FAILED — score lacks discriminatory power. '
                'Investigate: timescale mismatch, weight collapse, '
                'or insufficient round-trips.'
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run()
