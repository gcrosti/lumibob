"""
study0_short_leg_fraction — Validation gate for short_leg_fraction.

Purpose
-------
Confirm that a fractional short leg adds value before building the full
regime machinery around it.  short_leg_fraction is the only free parameter;
all other params are fixed at their current defaults.

Gate criterion
--------------
Best short_leg_fraction > 0.05 in at least one of the two folds:
  - sideways_2022 : 2022-02-01 → 2022-04-30
  - bull_2023     : 2023-04-01 → 2023-06-30

If it snaps to 0 in both folds, partial hedging adds no value and the
short leg hypothesis is dead.  Do not build regime machinery around it.

Design
------
- Objective : pure Sharpe (discriminatory_weight=0.0).  We do not need
  the discriminatory objective here — we are asking whether any fraction
  beats long-only, not whether a score discriminates.
- Trials    : 30 per fold (60 total).  short_leg_fraction [0.0, 1.0] is a
  1-D search; 30 trials is sufficient for TPE to converge.
- Wall-clock : ~2 hrs (warm cache).

Usage
-----
    RUN_MODE=backtest python -m tuning.studies.study0_short_leg_fraction
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
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

BUDGET = float(os.getenv('TUNE_BUDGET', '10000'))
N_TRIALS = int(os.getenv('TUNE_N_TRIALS', '30'))
TIMEOUT_HOURS = float(os.getenv('TUNE_TIMEOUT_HOURS', '3'))
DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')

FOLDS: list[tuple[str, date, date]] = [
    ('sideways_2022', date(2022, 2, 1), date(2022, 4, 30)),
    ('bull_2023',     date(2023, 4, 1), date(2023, 6, 30)),
]

_GATE_THRESHOLD = 0.05  # best short_leg_fraction must exceed this in >= 1 fold


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run() -> dict[str, optuna.Study]:
    """
    Run Study 0 across both folds.  Returns a dict mapping fold name to study.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = optuna.storages.RDBStorage(
        url=DB_URL,
        engine_kwargs={'pool_pre_ping': True, 'pool_size': 1},
        skip_compatibility_check=False,
    )

    # Build base params: all defaults, then override short_leg_fraction to 0.0
    # so the Tier 3 suggest() call is the only source of variation.
    base = defaults()
    base['short_leg_fraction'] = 0.0

    studies: dict[str, optuna.Study] = {}

    for fold_name, train_start, train_end in FOLDS:
        study_name = f'study0_{fold_name}'
        logger.info(
            'Starting fold "%s"  train=%s→%s  n_trials=%d',
            fold_name, train_start, train_end, N_TRIALS,
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            load_if_exists=True,
        )

        objective = BacktestObjective(
            train_start=train_start,
            train_end=train_end,
            budget=BUDGET,
            base_params=base,
            tiers=(3,),           # only short_leg_fraction (Tier 3) is free
            spy_penalty_weight=0.0,  # Phase 4+: goal is positive Sharpe
            discriminatory_weight=0.0,  # pure Sharpe — 1-D search
        )

        study.optimize(
            objective,
            n_trials=N_TRIALS,
            timeout=TIMEOUT_HOURS * 3600,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        _log_fold_results(study, fold_name)
        studies[fold_name] = study

    _run_gate(studies)
    return studies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_fold_results(study: optuna.Study, fold_name: str) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    logger.info(
        '[%s] %d completed, %d pruned, %d failed',
        fold_name, len(completed), len(pruned), len(failed),
    )
    if completed:
        slf = study.best_params.get('short_leg_fraction', float('nan'))
        logger.info(
            '[%s] best_value=%.4f  best short_leg_fraction=%.4f',
            fold_name, study.best_value, slf,
        )
    else:
        logger.warning('[%s] No completed trials', fold_name)


def _run_gate(studies: dict[str, optuna.Study]) -> None:
    """
    Gate: best short_leg_fraction > _GATE_THRESHOLD in at least one fold.
    Prints PASS/FAIL and the recommended next step.
    """
    best_slfs = {}
    for fold_name, study in studies.items():
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            best_slfs[fold_name] = study.best_params.get('short_leg_fraction', 0.0)
        else:
            best_slfs[fold_name] = 0.0

    passes = [fn for fn, v in best_slfs.items() if v > _GATE_THRESHOLD]
    verdict = 'PASS' if passes else 'FAIL'

    logger.info('--- Study 0 Gate ---')
    for fold_name, v in best_slfs.items():
        logger.info('  %s: best short_leg_fraction = %.4f', fold_name, v)
    logger.info('Gate verdict: %s', verdict)

    if verdict == 'PASS':
        logger.info(
            'NEXT STEP: run Study 1 Pass A '
            '(tuning.studies.study1_pass_a).'
        )
    else:
        logger.warning(
            'GATE FAILED: short_leg_fraction snaps to ~0 in all folds. '
            'Partial hedging adds no value. '
            'Do not build regime machinery around it. '
            'Investigate whether the short leg hypothesis is structurally '
            'unachievable (e.g. spread convergence = both legs rising).'
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run()
