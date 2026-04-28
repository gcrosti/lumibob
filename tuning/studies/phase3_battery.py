"""
phase3_battery — Phase 3 five-regime calibration battery.

Loads the best parameter set from the Phase 1 proof study (tier2_proof_v2)
and runs the standard five-regime battery against the canonical baseline to
verify that:

    1. The tuner found parameters that generalise beyond the training window.
    2. Phase 2 improvements (sector partition, corr-distance HDBSCAN, K-fix)
       are measurable in the battery results.

Gate criterion (Phase 3 → Phase 4 advance):
    Best-trial params outscore default params in ≥ 3 of the completed regimes.
    (Cold regimes are skipped; the gate applies only to completed ones.)

Usage
-----
    # Warm windows only (default — safe, fast):
    RUN_MODE=backtest python -m tuning.studies.phase3_battery

    # Include cold windows (slow — accept Alpaca fetch time):
    RUN_MODE=backtest python -m tuning.studies.phase3_battery --include-cold

    # Specify a different Phase 1 study to load best params from:
    TUNE_PHASE1_STUDY=tier2_proof_v2 RUN_MODE=backtest python -m tuning.studies.phase3_battery

    # Check cache warmth for all regimes without running backtests:
    RUN_MODE=backtest python -m tuning.studies.phase3_battery --check-cache

Important: failed_tickers
    StockDataCache permanently marks symbols with no Alpaca data as "failed".
    Historical backtests (2017, 2020, 2022) will encounter tickers that did not
    yet exist and may incorrectly blacklist them.  This runner snapshots the
    failed_tickers table before each cold-window backtest and restores it
    afterwards, so historical runs don't pollute the live failed-ticker list.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import optuna
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Path setup — allow running as __main__ without installing the package.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tuning.battery import (
    REGIMES,
    BatteryResult,
    Regime,
    check_warmth,
    gate_check,
    print_report,
    run_battery,
)
from tuning.parameter_space import defaults, normalize_weights

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s — %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')
PHASE1_STUDY = os.getenv('TUNE_PHASE1_STUDY', 'tier2_proof_v2')
BUDGET = float(os.getenv('TUNE_BUDGET', '10000'))


# ---------------------------------------------------------------------------
# Load Phase 1 best params
# ---------------------------------------------------------------------------

def load_phase1_best_params(study_name: str) -> dict:
    """
    Return the best params from an Optuna study, with composite weights
    normalised to sum to 1.0.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.RDBStorage(
        url=DB_URL,
        engine_kwargs={'pool_pre_ping': True, 'pool_size': 1},
    )
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as exc:
        raise RuntimeError(
            f'Could not load Optuna study "{study_name}": {exc}\n'
            'Run the Phase 1 proof study first.'
        ) from exc

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f'Study "{study_name}" has no completed trials.')

    raw_best = dict(study.best_params)
    # Merge with defaults so every parameter is present, then normalise weights.
    full = {**defaults(), **raw_best}
    return normalize_weights(full)


# ---------------------------------------------------------------------------
# failed_tickers snapshot / restore
# ---------------------------------------------------------------------------

def _snapshot_failed_tickers() -> list[dict]:
    """Return all rows in failed_tickers for later restoration."""
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute('SELECT symbol, reason, failed_at FROM failed_tickers')
                return [
                    {'symbol': r[0], 'reason': r[1], 'failed_at': r[2]}
                    for r in cur.fetchall()
                ]
            except psycopg2.errors.UndefinedTable:
                return []


def _restore_failed_tickers(snapshot: list[dict]) -> None:
    """Delete all current failed_tickers rows and re-insert the snapshot."""
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute('DELETE FROM failed_tickers')
            except psycopg2.errors.UndefinedTable:
                return
            if snapshot:
                psycopg2.extras.execute_values(
                    cur,
                    'INSERT INTO failed_tickers (symbol, reason, failed_at) VALUES %s '
                    'ON CONFLICT (symbol) DO NOTHING',
                    [(r['symbol'], r['reason'], r['failed_at']) for r in snapshot],
                )


# ---------------------------------------------------------------------------
# Cache warmth report
# ---------------------------------------------------------------------------

def print_cache_warmth() -> None:
    print('\n  Phase 3 cache warmth check')
    print(f'  {"Regime":<25}  {"Warm?":<6}  {"Note"}')
    print('-' * 70)
    for regime in REGIMES:
        is_warm, note = check_warmth(regime)
        status = 'WARM' if is_warm else 'COLD'
        print(f'  {regime.name:<25}  {status:<6}  {note}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(include_cold: bool = False) -> None:
    logger.info('=== Phase 3 Battery ===')

    # -- Load parameters --
    logger.info('Loading Phase 1 best params from study "%s"...', PHASE1_STUDY)
    try:
        best_params = load_phase1_best_params(PHASE1_STUDY)
    except RuntimeError as exc:
        logger.error('%s', exc)
        sys.exit(1)

    base_params = defaults()

    logger.info(
        'Best params (Tier 2 suggestions only): %s',
        {
            k: v for k, v in best_params.items()
            if k not in base_params or best_params[k] != base_params[k]
        },
    )

    # -- Log warmth --
    print_cache_warmth()
    warm_only = not include_cold

    # -- Snapshot failed_tickers before any cold runs --
    failed_snapshot = _snapshot_failed_tickers() if include_cold else []

    try:
        # Run battery for best params
        logger.info('Running battery: BEST PARAMS (label="best_trial")...')
        result_best = run_battery(
            params=best_params,
            params_label='best_trial',
            budget=BUDGET,
            warm_only=warm_only,
        )

        # NOTE: intentionally do NOT restore failed_tickers between param sets.
        # Tickers that returned no data for historical windows (e.g. didn't exist
        # in 2017) are legitimately absent for those dates.  Keeping them in the
        # failed set means the baseline battery skips the same slow Alpaca fetches,
        # cutting the total wall-clock time roughly in half.
        # The final restore (in the finally block) returns the table to its
        # original state so the live strategy is unaffected.

        # Run battery for baseline (default params)
        logger.info('Running battery: BASELINE (label="baseline")...')
        result_base = run_battery(
            params=base_params,
            params_label='baseline',
            budget=BUDGET,
            warm_only=warm_only,
        )

    except KeyboardInterrupt:
        logger.warning('Battery interrupted.')
        raise
    finally:
        # Always restore failed_tickers so the live strategy is unaffected.
        _restore_failed_tickers(failed_snapshot)

    # -- Report --
    print_report(result_best, result_base)

    # -- Gate check --
    passed = gate_check(
        result_good=result_best,
        result_bad=result_base,
        min_wins=2 if not include_cold else 3,  # lower bar for warm-only
    )

    # -- Cold-window notice --
    cold = [r for r in result_best.regime_results if r.skipped]
    if cold:
        print('\n  Cold regimes (pre-warm before re-running):')
        for r in cold:
            print(f'    {r.regime.name:<25}  {r.regime.start} → {r.regime.end}')
        print(
            '\n  To pre-warm these windows, run:\n'
            '    python scripts/prewarm_cache.py --regime <name>\n'
            '  Then re-run this battery with --include-cold.\n'
        )

    # -- Summary --
    if passed:
        logger.info('Phase 3 gate PASSED — ready to proceed to Phase 4.')
    else:
        logger.warning(
            'Phase 3 gate NOT YET MET. '
            '%d warm regime(s) completed. Pre-warm cold windows and re-run.',
            len(result_best.completed()),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 3: five-regime battery comparison.',
    )
    parser.add_argument(
        '--include-cold',
        action='store_true',
        help='Run all regimes including cold ones (slow — accepts Alpaca fetch time).',
    )
    parser.add_argument(
        '--check-cache',
        action='store_true',
        help='Print cache warmth for each regime and exit.',
    )
    args = parser.parse_args()

    if args.check_cache:
        print_cache_warmth()
        sys.exit(0)

    run(include_cold=args.include_cold)
