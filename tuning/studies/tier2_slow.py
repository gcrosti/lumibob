"""
tier2_slow — Phase 1 proof study for Tier 2 (slow-adaptive) parameters.

Runs a single-fold Optuna study on the warm-cache training window to
confirm the tuning harness works end-to-end.

Gate criterion (Phase 1 advance condition):
    best_trial_score > baseline_score  (even marginally)

Usage
-----
    # Standard run (uses defaults from this file):
    RUN_MODE=backtest python -m tuning.studies.tier2_slow

    # Override number of trials or timeout via env vars:
    TUNE_N_TRIALS=20 TUNE_TIMEOUT_HOURS=2 RUN_MODE=backtest \\
        python -m tuning.studies.tier2_slow

Design
------
Training window : 2024-01-02 → 2024-03-25
    58 trading days — already warm-cached from Phase 2 verification run.
    Use this window for Phase 1 to avoid cold-cache overhead.
    Longer windows (12 months) will be used in Phase 3+ after warming.

N_TRIALS        : 50 (with TIMEOUT_HOURS=4 wall-clock cap).
    At ~72 min / trial, expect 3-4 completed trials in the time budget.
    The goal is end-to-end validation, not exhaustive search.

Storage         : Optuna PostgreSQL RDB (same DB as strategy data).
    Allows resuming interrupted studies and future parallel workers.
    Study name 'tier2_proof_v2' is preserved across restarts.
"""

from __future__ import annotations

import logging
import os
from datetime import date

import optuna
from dotenv import load_dotenv

from tuning.objective import BacktestObjective
from tuning.parameter_space import defaults

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s — %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — override via environment variables for flexibility
# ---------------------------------------------------------------------------

TRAIN_START = date(2024, 1, 2)
TRAIN_END = date(2024, 3, 25)
BUDGET = float(os.getenv('TUNE_BUDGET', '10000'))
N_TRIALS = int(os.getenv('TUNE_N_TRIALS', '50'))
TIMEOUT_HOURS = float(os.getenv('TUNE_TIMEOUT_HOURS', '4'))
STUDY_NAME = os.getenv('TUNE_STUDY_NAME', 'tier2_proof_v2')
DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run() -> optuna.Study:
    """
    Run (or resume) the Phase 1 Tier 2 proof study.

    Returns the Optuna Study object so callers can inspect best_params and
    best_value programmatically.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = optuna.storages.RDBStorage(
        url=DB_URL,
        engine_kwargs={'pool_pre_ping': True, 'pool_size': 1},
        skip_compatibility_check=False,
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        load_if_exists=True,
    )

    objective = BacktestObjective(
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        budget=BUDGET,
        base_params=defaults(),
        tiers=(2,),
    )

    n_existing = len(study.trials)
    logger.info(
        'Study "%s" — %d existing trials  train=%s→%s  budget=%.0f  '
        'max_trials=%d  timeout=%.1fh',
        STUDY_NAME, n_existing, TRAIN_START, TRAIN_END,
        BUDGET, N_TRIALS, TIMEOUT_HOURS,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_HOURS * 3600,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    _log_results(study)
    _run_gate_check(study, objective)

    return study


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_results(study: optuna.Study) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    logger.info(
        'Study complete — %d completed, %d pruned, %d failed',
        len(completed), len(pruned), len(failed),
    )

    if completed:
        logger.info('Best value  : %.4f', study.best_value)
        logger.info('Best params : %s', study.best_params)
    else:
        logger.warning('No completed trials — cannot report best params')


def _run_gate_check(study: optuna.Study, objective: BacktestObjective) -> None:
    """
    Run one backtest with the canonical default parameters on the same
    training window and compare to the study's best trial.

    Prints PASS / FAIL for the Phase 1 gate criterion.
    """
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        logger.warning('Gate check skipped — no completed trials')
        return

    logger.info('Running baseline (default params) for gate comparison...')

    run_id = objective._run_backtest(objective.base_params)
    if run_id is None:
        logger.warning('Baseline backtest failed — gate check skipped')
        return

    baseline_score = objective.score_run(run_id, trial=None)
    best_score = study.best_value
    delta = best_score - baseline_score

    verdict = 'PASS' if delta > 0 else 'FAIL'
    logger.info(
        'Gate check: best_trial=%.4f  baseline=%.4f  delta=%+.4f  [%s]',
        best_score, baseline_score, delta, verdict,
    )
    if verdict == 'FAIL':
        logger.warning(
            'Phase 1 gate NOT met. Consider: wider search space, longer '
            'training window, more trials, or revisiting the objective function.',
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run()
