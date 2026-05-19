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

Free parameters (all Tier 2, 10 total)
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
2022-01-01 → 2024-06-30 (covers sideways, bull, and mixed regimes).
Wide enough for TPE to distinguish timescale-dependent signal quality.

Gate criterion
--------------
Best-trial Spearman rho > 0.15 in at least 2 of 3 OOS folds
(sideways_2022, bull_2023, mixed_2023_24).
Gate is evaluated AFTER the study completes.  If fewer than 2 folds pass,
investigate root cause before running Study 1 Pass B.

Usage
-----
    RUN_MODE=backtest python -m tuning.studies.study1_pass_a
"""

from __future__ import annotations

import logging
import os
from datetime import date

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
N_TRIALS = int(os.getenv('TUNE_N_TRIALS', '200'))
TIMEOUT_HOURS = float(os.getenv('TUNE_TIMEOUT_HOURS', '8'))
STUDY_NAME = os.getenv('TUNE_STUDY_NAME', 'study1_pass_a_v1')
DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')

# Full training window — covers sideways + bull + mixed.
TRAIN_START = date(2022, 1, 3)   # first trading day of 2022
TRAIN_END   = date(2024, 6, 28)  # last trading day of June 2024

# Discriminatory objective weight (plan: 0.7).
DISCRIMINATORY_WEIGHT = float(os.getenv('TUNE_DISC_WEIGHT', '0.7'))

# OOS folds for post-study gate check.
# These windows must NOT overlap the training window.
GATE_FOLDS: list[tuple[str, date, date]] = [
    ('sideways_2022',  date(2022, 1, 3),  date(2022, 4, 29)),
    ('bull_2023',      date(2023, 4, 3),  date(2023, 6, 30)),
    ('mixed_2023_24',  date(2023, 9, 1),  date(2024, 3, 29)),
]
_GATE_RHO_THRESHOLD = 0.15
_GATE_MIN_PASSING_FOLDS = 2


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

    # Base: all defaults.  Tier 2 params will be overridden by suggest().
    # Tier 3 stays at defaults (short_leg_fraction=0.0 → long-only).
    base = defaults()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    objective = BacktestObjective(
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        budget=BUDGET,
        base_params=base,
        tiers=(2,),
        spy_penalty_weight=0.0,          # Phase 4+: goal is positive Sharpe
        discriminatory_weight=DISCRIMINATORY_WEIGHT,
        pnl_floor=-100.0,
        min_round_trips=10,
        trial_timeout_secs=1200,         # 20 min hard kill per trial
    )

    n_existing = len(study.trials)
    logger.info(
        'Study "%s" — %d existing trials  train=%s→%s  budget=%.0f  '
        'max_trials=%d  timeout=%.1fh  disc_weight=%.2f',
        STUDY_NAME, n_existing, TRAIN_START, TRAIN_END,
        BUDGET, N_TRIALS, TIMEOUT_HOURS, DISCRIMINATORY_WEIGHT,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_HOURS * 3600,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    _log_results(study)
    _run_gate(study, objective)

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
    logger.info('Best params (normalised weights):')
    for k, v in sorted(best_params.items()):
        logger.info('  %-30s %s', k, v)


def _run_gate(study: optuna.Study, objective: BacktestObjective) -> None:
    """
    Gate: evaluate best-trial params on each OOS fold using a pure-Sharpe
    discriminatory score (discriminatory_weight=1.0 but using the stored
    score from the run).  The gate checks Spearman rho > 0.15 in >= 2 folds.

    This re-runs the best-trial params on each fold as a fresh backtest and
    calls objective._discriminatory_score() on the resulting run.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning('Gate check skipped — no completed trials')
        return

    best_params = normalize_weights({**objective.base_params, **study.best_params})

    logger.info('--- Study 1 Pass A Gate: OOS Spearman rho ---')

    fold_rhos: dict[str, float | None] = {}
    for fold_name, fold_start, fold_end in GATE_FOLDS:
        logger.info('  Running fold "%s"  %s → %s', fold_name, fold_start, fold_end)
        gate_obj = BacktestObjective(
            train_start=fold_start,
            train_end=fold_end,
            budget=objective.budget,
            base_params=best_params,
            tiers=(),                  # no free params — everything fixed
            spy_penalty_weight=0.0,
            discriminatory_weight=0.0,  # only need _discriminatory_score
            min_round_trips=10,
            trial_timeout_secs=1200,
        )
        run_id = gate_obj._run_backtest(best_params)
        if run_id is None:
            logger.warning('  [%s] backtest failed — skipping fold', fold_name)
            fold_rhos[fold_name] = None
            continue

        rho = gate_obj._discriminatory_score(run_id)
        fold_rhos[fold_name] = rho
        logger.info('  [%s] rho = %s', fold_name, f'{rho:.4f}' if rho is not None else 'N/A')

    passing = [
        fn for fn, rho in fold_rhos.items()
        if rho is not None and rho > _GATE_RHO_THRESHOLD
    ]
    verdict = 'PASS' if len(passing) >= _GATE_MIN_PASSING_FOLDS else 'FAIL'

    logger.info(
        'Gate: %d/%d folds pass rho>%.2f — %s',
        len(passing), len(GATE_FOLDS), _GATE_RHO_THRESHOLD, verdict,
    )

    if verdict == 'PASS':
        logger.info(
            'NEXT STEP: run Study 1 Pass B '
            '(tuning.studies.study1_pass_b) with Tier 2 signal params fixed '
            'at best_params above.'
        )
    else:
        n_close = sum(
            1 for rho in fold_rhos.values()
            if rho is not None and rho > _GATE_RHO_THRESHOLD * 0.5
        )
        if n_close >= _GATE_MIN_PASSING_FOLDS:
            logger.warning(
                'GATE NARROWLY FAILED — rho is close to threshold in multiple '
                'folds. Consider widening to 300 trials before concluding failure.'
            )
        else:
            logger.error(
                'GATE FAILED — composite score lacks discriminatory power at any '
                'tested timescale/weight combination. Investigate: '
                '(1) lookback/zscore mismatch, '
                '(2) all weights collapsed to one component, '
                '(3) insufficient trades for reliable rank correlation.'
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run()
