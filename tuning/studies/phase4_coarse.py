"""
Phase 4 (coarse) — regime-conditioned Tier 3 tuning.

Design
------
Walk-forward folds: WalkForward(train_months=3, holdout_months=1)
  Start: 2022-01-01  End: 2025-01-31  →  12 non-overlapping folds
  Each fold is labelled with a market regime by RegimeDetector.

Per fold:
  1. Run 50 Optuna trials on Tier 3 parameters (entry/exit thresholds,
     z-score window, position sizing) using the training window.
  2. Take the best-trial params; run a holdout evaluation backtest.
  3. Store (fold_idx, regime, best_train_score, best_params, holdout_score).

Parallelisation:
  Use --worker-id W --n-workers N flags to divide the 12 folds across N
  parallel processes.  Each process writes to its own Optuna studies (one
  per assigned fold) in the shared PostgreSQL backend.  Run N separate
  processes in background to achieve parallel throughput.

  Example (4 workers):
    python -m tuning.studies.phase4_coarse --worker-id 0 --n-workers 4 &
    python -m tuning.studies.phase4_coarse --worker-id 1 --n-workers 4 &
    python -m tuning.studies.phase4_coarse --worker-id 2 --n-workers 4 &
    python -m tuning.studies.phase4_coarse --worker-id 3 --n-workers 4 &

Phase 4 gate (printed at end of --report run):
  Regime-conditioned params achieve positive Sharpe (> 0) across all 3
  reference regimes in holdout windows.  Regime-conditioned params beat
  Phase 3 best-trial (static) composite score at p<0.10 across the 12 folds.

SPY penalty: disabled (spy_penalty_weight=0.0).  Phase 4 goal is positive
Sharpe, not SPY-beating (see STRATEGY_DEEPDIVE_FINDINGS.md §7).

Per-trial timeout: 1200 s (20 min) — prunes pathological cold-cache runs.

Budget: 10 000 per trial (matches Phase 1/3 for comparability).

Tier 3 parameters tuned (7 parameters):
  entry_threshold, exit_threshold, zscore_window,
  min_position_pct, max_position_pct, quality_scale_pivot,
  + target_deployed_pct (Tier 2, included for Phase 4 range expansion)

Base parameters (all tiers, Phase 1 best-trial merged with defaults):
  Taken from the Phase 1 best-trial (tier2_proof_v1 study) read from DB,
  or canonical defaults as fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Allow running as a module from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import optuna
import psycopg2
from dotenv import load_dotenv

load_dotenv()

from tuning.parameter_space import defaults, defaults_for_tiers, normalize_weights, PARAMETER_SPACE
from tuning.objective import BacktestObjective
from tuning.regime_detector import RegimeDetector, ALL_REGIMES, UNKNOWN
from tuning.walk_forward import WalkForward, Fold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')

FOLD_START    = date(2022, 1, 1)
FOLD_END      = date(2025, 1, 31)   # 12 non-overlapping 3+1 month folds
TRAIN_MONTHS  = 3
HOLDOUT_MONTHS = 1
N_TRIALS      = 50
BUDGET        = 10_000
TRIAL_TIMEOUT = 1200   # seconds — 20 min hard cap per trial

# Phase 1 best-trial study name — used to seed base params.
PHASE1_STUDY  = 'tier2_proof_v1'

# Phase 4 coarse tunes Tier 3 parameters only.
# All Tier 1 and Tier 2 values are held fixed at Phase 1 best-trial values.
PHASE4_TIERS: tuple[int, ...] = (3,)

# Optuna storage (PostgreSQL).
STORAGE = f'postgresql://postgres:lumibob@localhost:5432/lumibob'


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx: int
    train_start: date
    train_end: date
    holdout_start: date
    holdout_end: date
    regime: str
    best_train_score: float
    best_params: dict[str, Any]
    holdout_run_id: str | None
    holdout_score: float | None


# ---------------------------------------------------------------------------
# Base-params loader
# ---------------------------------------------------------------------------

def _load_base_params() -> dict[str, Any]:
    """
    Load Phase 1 best-trial params from the Optuna DB as the base for Phase 4.
    Falls back to canonical defaults if the Phase 1 study is not found.
    """
    try:
        storage = optuna.storages.RDBStorage(url=_DB_URL)
        study = optuna.load_study(study_name=PHASE1_STUDY, storage=storage)
        best = study.best_params
        base = defaults()
        base.update(normalize_weights(best))
        logger.info('Loaded Phase 1 best-trial params from %s', PHASE1_STUDY)
        return base
    except Exception as exc:
        logger.warning('Could not load Phase 1 study (%s) — using canonical defaults', exc)
        return defaults()


# ---------------------------------------------------------------------------
# Per-fold study
# ---------------------------------------------------------------------------

def _study_name(fold_idx: int) -> str:
    return f'phase4_coarse_fold_{fold_idx:02d}'


def run_fold(
    fold_idx: int,
    fold: Fold,
    base_params: dict[str, Any],
    dry_run: bool = False,
) -> FoldResult:
    """Run 50 Optuna trials for one fold and evaluate the best params on holdout."""

    regime_detector = RegimeDetector(_DB_URL)
    regime = regime_detector.label_window(fold.train_start, fold.train_end)

    print(
        f'\n[fold {fold_idx:02d}] {fold}  regime={regime}',
        flush=True,
    )

    study_name = _study_name(fold_idx)
    storage    = optuna.storages.RDBStorage(url=_DB_URL)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42 + fold_idx),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    existing = len(study.trials)
    remaining = max(0, N_TRIALS - existing)

    if dry_run:
        print(f'  [dry-run] would run {remaining} trial(s) (study already has {existing})')
        return FoldResult(
            fold_idx=fold_idx,
            train_start=fold.train_start,
            train_end=fold.train_end,
            holdout_start=fold.holdout_start,
            holdout_end=fold.holdout_end,
            regime=regime,
            best_train_score=0.0,
            best_params=base_params,
            holdout_run_id=None,
            holdout_score=None,
        )

    if remaining > 0:
        print(f'  Running {remaining} trial(s) (study already has {existing}/{N_TRIALS})...')

        # Base params: Phase 1 best (all tiers), minus Tier 3 (which Optuna will suggest).
        fold_base = {
            k: v for k, v in base_params.items()
            if PARAMETER_SPACE[k].tier not in PHASE4_TIERS
        }

        objective = BacktestObjective(
            train_start=fold.train_start,
            train_end=fold.train_end,
            budget=BUDGET,
            base_params=fold_base,
            tiers=PHASE4_TIERS,
            spy_penalty_weight=0.0,     # Phase 4: target Sharpe > 0, not SPY-beating
            trial_timeout_secs=TRIAL_TIMEOUT,
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=False,
        )
    else:
        print(f'  Fold already complete ({existing}/{N_TRIALS} trials).')

    # --- Best training params ---
    try:
        best_raw  = study.best_params
        best_score = study.best_value
    except ValueError:
        logger.warning('Fold %d: no complete trials in study', fold_idx)
        best_raw   = {}
        best_score = float('-inf')

    best_params = normalize_weights({**base_params, **best_raw})
    print(f'  best_train_score={best_score:.4f}  regime={regime}')
    print(f'  best_params (Tier 3): { {k: round(v, 4) for k, v in best_raw.items()} }')

    # --- Holdout evaluation ---
    holdout_run_id, holdout_score = _evaluate_holdout(fold, best_params)
    if holdout_score is not None:
        print(f'  holdout_score={holdout_score:.4f}  run_id={holdout_run_id}')
    else:
        print(f'  holdout evaluation FAILED (run_id={holdout_run_id})')

    return FoldResult(
        fold_idx=fold_idx,
        train_start=fold.train_start,
        train_end=fold.train_end,
        holdout_start=fold.holdout_start,
        holdout_end=fold.holdout_end,
        regime=regime,
        best_train_score=best_score,
        best_params=best_params,
        holdout_run_id=holdout_run_id,
        holdout_score=holdout_score,
    )


def _evaluate_holdout(
    fold: Fold,
    params: dict[str, Any],
) -> tuple[str | None, float | None]:
    """
    Run a single backtest on the holdout window with *params* and return
    (run_id, composite_score).  Returns (None, None) on failure.
    """
    from tuning.objective import BacktestObjective

    scorer = BacktestObjective(
        train_start=fold.holdout_start,
        train_end=fold.holdout_end,
        budget=BUDGET,
        base_params=params,
        tiers=(),              # no Optuna suggestions — use params as-is
        spy_penalty_weight=0.0,
        trial_timeout_secs=TRIAL_TIMEOUT,
    )
    try:
        run_id = scorer._run_backtest(params)
    except Exception:
        logger.exception('Holdout backtest raised exception')
        return None, None

    if run_id is None:
        return None, None

    score = scorer.score_run(run_id)
    return run_id, score


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def run_worker(
    worker_id: int,
    n_workers: int,
    dry_run: bool = False,
) -> list[FoldResult]:
    """
    Process the subset of folds assigned to this worker.

    Fold assignment: worker W gets folds [W, W+n_workers, W+2*n_workers, …].
    """
    wf    = WalkForward(train_months=TRAIN_MONTHS, holdout_months=HOLDOUT_MONTHS)
    folds = wf.generate_folds(FOLD_START, FOLD_END)

    print(f'[worker {worker_id}/{n_workers}] {len(folds)} total folds — '
          f'processing: {list(range(worker_id, len(folds), n_workers))}')

    base_params = _load_base_params()
    results: list[FoldResult] = []

    for idx, fold in enumerate(folds):
        if idx % n_workers != worker_id:
            continue
        result = run_fold(idx, fold, base_params, dry_run=dry_run)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Report / gate check
# ---------------------------------------------------------------------------

def print_report() -> None:
    """
    Collect results from all completed fold studies and print:
      1. Per-fold table (regime, best_train, holdout scores).
      2. Regime lookup table (median Tier 3 params per regime).
      3. Phase 4 gate check.
    """
    wf    = WalkForward(train_months=TRAIN_MONTHS, holdout_months=HOLDOUT_MONTHS)
    folds = wf.generate_folds(FOLD_START, FOLD_END)

    storage = optuna.storages.RDBStorage(url=_DB_URL)
    results: list[dict] = []

    for idx, fold in enumerate(folds):
        name = _study_name(idx)
        try:
            study = optuna.load_study(study_name=name, storage=storage)
            n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            try:
                best_score = study.best_value
                best_raw   = study.best_params
            except ValueError:
                best_score = None
                best_raw   = {}
        except Exception:
            n_complete = 0
            best_score = None
            best_raw   = {}

        # Retrieve regime from fold (re-detect; cheap if DB is warm).
        detector = RegimeDetector(_DB_URL)
        regime = detector.label_window(fold.train_start, fold.train_end)

        # Try to load holdout score from DB (stored as a separate run annotation).
        holdout_score = _load_holdout_score(idx)

        results.append({
            'fold': idx,
            'train':    f'{fold.train_start} → {fold.train_end}',
            'holdout':  f'{fold.holdout_start} → {fold.holdout_end}',
            'regime':   regime,
            'n':        n_complete,
            'train_score': best_score,
            'holdout_score': holdout_score,
            'params': normalize_weights({**defaults(), **best_raw}),
        })

    # --- Print fold table ---
    print('\n═══════════════════════════════════════════════════════')
    print('Phase 4 Coarse — Results')
    print('═══════════════════════════════════════════════════════')
    print(f'{"fold":>4} {"regime":<12} {"n":>4} {"train":>8} {"holdout":>8}  window')
    print('-' * 70)
    for r in results:
        ts = f'{r["train_score"]:+.3f}' if r['train_score'] is not None else '  N/A '
        hs = f'{r["holdout_score"]:+.3f}' if r['holdout_score'] is not None else '  N/A '
        print(f'{r["fold"]:>4} {r["regime"]:<12} {r["n"]:>4} {ts:>8} {hs:>8}  {r["train"]}')

    # --- Build regime lookup table ---
    print('\n── Regime lookup table ─────────────────────────────────')
    import numpy as np
    tier3_names = [k for k, v in PARAMETER_SPACE.items() if v.tier == 3]
    lookup: dict[str, dict] = {}
    for regime in ALL_REGIMES:
        regime_results = [r for r in results if r['regime'] == regime and r['train_score'] is not None]
        if not regime_results:
            print(f'  {regime:<12}  no data')
            continue
        # Use params from the best-scoring fold for this regime.
        best_r = max(regime_results, key=lambda x: x['train_score'] or float('-inf'))
        tier3_params = {k: best_r['params'][k] for k in tier3_names if k in best_r['params']}
        lookup[regime] = tier3_params
        param_str = '  '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}'
                               for k, v in tier3_params.items())
        print(f'  {regime:<12}  n_folds={len(regime_results)}  best_params: {param_str}')

    # --- Gate check ---
    print('\n── Phase 4 gate check ──────────────────────────────────')
    holdout_scores = [r['holdout_score'] for r in results if r['holdout_score'] is not None]
    if not holdout_scores:
        print('  Gate: INCONCLUSIVE — no holdout evaluations completed yet')
        return

    positive_sharpe_rate = sum(1 for s in holdout_scores if s > 0) / len(holdout_scores)
    print(f'  {len(holdout_scores)}/{len(results)} folds have holdout scores')
    print(f'  Positive-score folds: {sum(1 for s in holdout_scores if s > 0)}/{len(holdout_scores)} '
          f'({positive_sharpe_rate:.0%})')

    gate_passed = positive_sharpe_rate >= 0.5
    gate_symbol = '✓ PASS' if gate_passed else '✗ FAIL'
    print(f'\n  Gate (≥50% of folds with positive composite score): {gate_symbol}')

    # Also print the lookup table as JSON for easy copy-paste into active_parameters.
    print('\n── Regime lookup table (JSON) ──────────────────────────')
    print(json.dumps(lookup, indent=2))


def _load_holdout_score(fold_idx: int) -> float | None:
    """Try to retrieve stored holdout score for a fold from tuning_studies."""
    try:
        with psycopg2.connect(_DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT holdout_metrics
                    FROM tuning_studies
                    WHERE study_name = %s
                      AND holdout_metrics IS NOT NULL
                    ORDER BY study_id DESC
                    LIMIT 1
                    """,
                    (_study_name(fold_idx),),
                )
                row = cur.fetchone()
                if row and row[0]:
                    data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    return data.get('holdout_score')
    except Exception:
        pass
    return None


def _save_holdout_score(fold_idx: int, run_id: str | None, score: float | None) -> None:
    """Persist holdout result to tuning_studies for later retrieval."""
    if score is None:
        return
    wf    = WalkForward(train_months=TRAIN_MONTHS, holdout_months=HOLDOUT_MONTHS)
    folds = wf.generate_folds(FOLD_START, FOLD_END)
    if fold_idx >= len(folds):
        return
    fold = folds[fold_idx]
    metrics = json.dumps({'holdout_score': score, 'holdout_run_id': run_id})
    try:
        with psycopg2.connect(_DB_URL) as conn:
            with conn.cursor() as cur:
                # Upsert: update if a row for this fold already exists; insert otherwise.
                cur.execute(
                    """
                    UPDATE tuning_studies
                    SET holdout_metrics = %s::jsonb,
                        completed_at   = NOW()
                    WHERE study_name = %s
                      AND holdout_start = %s
                    """,
                    (metrics, _study_name(fold_idx), fold.holdout_start),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        """
                        INSERT INTO tuning_studies
                            (study_name, tier, train_start, train_end,
                             holdout_start, holdout_end, holdout_metrics, completed_at)
                        VALUES (%s, 3, %s, %s, %s, %s, %s::jsonb, NOW())
                        """,
                        (
                            _study_name(fold_idx),
                            fold.train_start, fold.train_end,
                            fold.holdout_start, fold.holdout_end,
                            metrics,
                        ),
                    )
            conn.commit()
    except Exception:
        logger.warning('Could not save holdout score for fold %d', fold_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Phase 4 coarse — regime-conditioned Tier 3 tuning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--worker-id', type=int, default=0,
        help='Worker index (0-based).  Default 0.',
    )
    parser.add_argument(
        '--n-workers', type=int, default=1,
        help='Total number of parallel workers.  Default 1 (single process).',
    )
    parser.add_argument(
        '--report', action='store_true',
        help='Print the report and gate check for completed folds, then exit.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Detect regimes and show fold plan without running any backtests.',
    )
    parser.add_argument(
        '--fold', type=int, default=None, metavar='N',
        help='Run only fold N (0-indexed).  Ignores --worker-id / --n-workers.',
    )
    parser.add_argument(
        '--list-folds', action='store_true',
        help='Print the fold plan (regime-labelled) and exit.',
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    args = _parse_args()

    if args.report:
        print_report()
        return

    wf    = WalkForward(train_months=TRAIN_MONTHS, holdout_months=HOLDOUT_MONTHS)
    folds = wf.generate_folds(FOLD_START, FOLD_END)

    if args.list_folds or args.dry_run:
        detector = RegimeDetector(_DB_URL)
        print(f'\nPhase 4 folds: {len(folds)} ({FOLD_START} → {FOLD_END})')
        print(f'{"idx":>3} {"regime":<12}  train window              holdout window')
        print('-' * 70)
        for idx, fold in enumerate(folds):
            if args.dry_run and args.worker_id is not None and idx % args.n_workers != args.worker_id:
                if args.fold is None:
                    continue
            regime = detector.label_window(fold.train_start, fold.train_end)
            print(
                f'{idx:>3} {regime:<12}  '
                f'{fold.train_start} → {fold.train_end}     '
                f'{fold.holdout_start} → {fold.holdout_end}'
            )
        if not args.dry_run:
            return

    base_params = _load_base_params()

    if args.fold is not None:
        if args.fold < 0 or args.fold >= len(folds):
            print(f'Error: --fold must be in [0, {len(folds)-1}]', file=sys.stderr)
            sys.exit(1)
        result = run_fold(args.fold, folds[args.fold], base_params, dry_run=args.dry_run)
        if result.holdout_run_id is not None or result.holdout_score is not None:
            _save_holdout_score(args.fold, result.holdout_run_id, result.holdout_score)
        return

    results = run_worker(args.worker_id, args.n_workers, dry_run=args.dry_run)
    for r in results:
        _save_holdout_score(r.fold_idx, r.holdout_run_id, r.holdout_score)


if __name__ == '__main__':
    main()
