"""
study1_pass_b — Tier 2 discovery + portfolio construction.

Purpose
-------
Pass A calibrated *whether the composite score discriminates* (timescales +
weights).  Pass B takes that calibrated score as fixed and asks a different
question: given a working signal, which discovery and position-sizing
parameters maximise risk-adjusted returns?  These params (cluster shape,
candidate volume, deployment level, K) affect how the strategy *uses* the
score to build a portfolio, not whether the score is good — so the objective
is pure Sharpe, not the discriminatory blend.

Base parameters
---------------
Loaded at startup from the completed Pass A study (TUNE_PASS_A_STUDY,
default 'study1_pass_a_v2'): best-trial signal params, weight-normalised,
merged over canonical defaults.  short_leg_fraction stays 0.0 (long-only —
the short leg is regime-conditioned and belongs to Study 2).

Free parameters (8, all Tier 2)
-------------------------------
    hdbscan_min_samples                 [1, 5]
    hdbscan_cluster_selection_epsilon   [0.0, 0.5]
    min_intra_cluster_corr              [0.1, 0.6]
    cluster_recompute_days              [14, 90]
    max_daily_candidates                [50, 300]
    target_deployed_pct                 [0.40, 0.90]
    max_k                               [5, 50]
    max_halflife_days                   [20, 120]

hdbscan_min_cluster_size is NOT free: it is Tier 1 and computed dynamically
at runtime since the Phase 2 clustering fixes.

Training window
---------------
The same three 5-month folds as Pass A, rotated round-robin — keeping the
windows identical means the fixed signal params are used in exactly the
conditions they were calibrated for.

Gate
----
None.  Best-trial output feeds Study 2 as its fixed Tier 2 foundation.

Usage
-----
    # Single worker:
    RUN_MODE=backtest python -m tuning.studies.study1_pass_b

    # Parallel workers (share the same Optuna study via PostgreSQL):
    for i in 1 2 3 4; do
      RUN_MODE=backtest python -m tuning.studies.study1_pass_b \\
        >> /tmp/study1b_w${i}.log 2>&1 &
    done
"""

from __future__ import annotations

import logging
import os

import optuna
from dotenv import load_dotenv

from tuning.parameter_space import defaults, normalize_weights
from tuning.studies.study1_pass_a import (
    TRAIN_FOLDS,
    FoldRotatingObjective,
)

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
N_TRIALS = int(os.getenv('TUNE_N_TRIALS', '75'))   # ~25 per fold across 3 folds
TIMEOUT_HOURS = float(os.getenv('TUNE_TIMEOUT_HOURS', '24'))
STUDY_NAME = os.getenv('TUNE_STUDY_NAME', 'study1_pass_b_v1')
PASS_A_STUDY = os.getenv('TUNE_PASS_A_STUDY', 'study1_pass_a_v2')
DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')

# Discovery / portfolio-construction params this study frees.  All Tier 2;
# disjoint from PASS_A_PARAMS by construction (asserted in run()).
PASS_B_PARAMS = frozenset({
    'hdbscan_min_samples',
    'hdbscan_cluster_selection_epsilon',
    'min_intra_cluster_corr',
    'cluster_recompute_days',
    'max_daily_candidates',
    'target_deployed_pct',
    'max_k',
    'max_halflife_days',
})


# ---------------------------------------------------------------------------
# Pass A output loader
# ---------------------------------------------------------------------------

def load_pass_a_base(storage: optuna.storages.RDBStorage) -> dict:
    """
    Build the fixed base parameter set from the completed Pass A study.

    Fails loudly if Pass A has no completed trials — running Pass B without
    a calibrated score would answer its question against an arbitrary signal.
    """
    try:
        pass_a = optuna.load_study(study_name=PASS_A_STUDY, storage=storage)
    except KeyError as exc:
        raise SystemExit(
            f'Pass A study "{PASS_A_STUDY}" not found in storage — '
            f'run tuning.studies.study1_pass_a first (or set TUNE_PASS_A_STUDY).'
        ) from exc

    completed = [t for t in pass_a.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise SystemExit(
            f'Pass A study "{PASS_A_STUDY}" has no completed trials — '
            f'Pass B needs its best-trial signal params as the fixed base.'
        )

    # best_params holds raw suggested values; weights must be re-normalised
    # exactly as Pass A's own results reporting does.
    base = normalize_weights({**defaults(), **pass_a.best_params})
    base['short_leg_fraction'] = 0.0   # long-only until Study 2

    logger.info(
        'Pass A base loaded from "%s" (%d completed trials, best=%.4f)',
        PASS_A_STUDY, len(completed), pass_a.best_value,
    )
    for k in sorted(pass_a.best_params):
        logger.info('  fixed %-30s %s', k, base[k])
    return base


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run() -> optuna.Study:
    """
    Run (or resume) Study 1 Pass B.

    Returns the Optuna Study object so callers can inspect best_params and
    best_value programmatically.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from tuning.studies.study1_pass_a import PASS_A_PARAMS
    assert not (PASS_B_PARAMS & PASS_A_PARAMS), (
        'PASS_B_PARAMS overlaps PASS_A_PARAMS — a param cannot be both '
        'fixed from Pass A and free in Pass B'
    )

    storage = optuna.storages.RDBStorage(
        url=DB_URL,
        engine_kwargs={'pool_pre_ping': True, 'pool_size': 1},
        skip_compatibility_check=False,
    )

    base = load_pass_a_base(storage)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=43, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    objective = FoldRotatingObjective(
        folds=TRAIN_FOLDS,
        budget=BUDGET,
        base_params=base,
        tiers=(2,),
        param_names=PASS_B_PARAMS,
        spy_penalty_weight=0.0,
        discriminatory_weight=0.0,       # pure Sharpe — see module docstring
        trial_timeout_secs=10800,        # 3 hr hard kill per trial
    )

    n_existing = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(
        'Study "%s" — %d completed trials so far  target=%d  '
        'timeout=%.1fh  objective=pure Sharpe  folds=%d',
        STUDY_NAME, n_existing, N_TRIALS, TIMEOUT_HOURS, len(TRAIN_FOLDS),
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

    logger.info('Best value  : %.4f', study.best_value)
    best_fold = study.best_trial.user_attrs.get('fold_start', 'unknown')
    logger.info('Best trial fold: %s', best_fold)
    logger.info('Best params:')
    for k, v in sorted(study.best_params.items()):
        logger.info('  %-35s %s', k, v)
    logger.info('NEXT STEP: Study 2 — per-regime Tier 3 (fixed Tier 2 = '
                'Pass A signal params + these discovery params)')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run()
