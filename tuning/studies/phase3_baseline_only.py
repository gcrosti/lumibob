"""
phase3_baseline_only — targeted 3-regime baseline battery.

Used after the full best-trial battery was interrupted early.  Skips the
best-trial backtests entirely; instead it re-loads metrics for those runs
directly from the DB using known (or auto-discovered) run IDs.  Then it runs
only the baseline parameter set for the same 3 regimes and produces the full
side-by-side Phase 3 report and gate check.

Target regimes
--------------
  1. calm_bull_2017   (2017-01 → 2017-12)
  2. vol_shock_2020   (2020-02 → 2020-06)
  3. sideways_2022    (2022-01 → 2022-12)

Usage
-----
    # Auto-discover best-trial run IDs from the DB by time-window:
    RUN_MODE=backtest python -m tuning.studies.phase3_baseline_only

    # Provide known run IDs (faster, no ambiguity):
    RUN_MODE=backtest python -m tuning.studies.phase3_baseline_only \\
        --calm-bull-id   3f7def.. \\
        --vol-shock-id   feac3e.. \\
        --sideways-id    <run_id after sideways_2022 completes>

    # List DB runs in each time window and exit (to find run IDs manually):
    RUN_MODE=backtest python -m tuning.studies.phase3_baseline_only --list-runs
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tuning.battery import (
    REGIMES,
    BatteryResult,
    Regime,
    RegimeResult,
    _fill_metrics,
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

DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')
PHASE1_STUDY = os.getenv('TUNE_PHASE1_STUDY', 'tier2_proof_v2')
BUDGET = float(os.getenv('TUNE_BUDGET', '10000'))

# The three regimes we have (or will have) best-trial results for.
TARGET_REGIMES: list[Regime] = [r for r in REGIMES if r.name in {
    'calm_bull_2017', 'vol_shock_2020', 'sideways_2022'
}]

# Window tolerance: look for backtest_runs whose portfolio_snapshots span
# within this many days of the regime boundary.
_WINDOW_TOLERANCE_DAYS = 30


# ---------------------------------------------------------------------------
# Helpers to load best-trial results from the DB
# ---------------------------------------------------------------------------

def _list_runs_in_window(regime: Regime) -> list[dict]:
    """
    Return all backtest runs whose portfolio_snapshots time range overlaps the
    regime window.  Used to help the user find the right run IDs.
    """
    lookback_start = regime.start - timedelta(days=150)
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    r.run_id,
                    r.started_at,
                    r.completed_at,
                    MIN(p.time)::date  AS ps_start,
                    MAX(p.time)::date  AS ps_end,
                    COUNT(p.*)         AS n_snapshots
                FROM backtest_runs r
                JOIN portfolio_snapshots p ON p.run_id = r.run_id
                WHERE r.mode = 'backtest'
                  AND p.time >= %s
                  AND p.time <= %s
                GROUP BY r.run_id, r.started_at, r.completed_at
                ORDER BY r.started_at DESC
                """,
                (
                    datetime.combine(lookback_start, datetime.min.time()),
                    datetime.combine(regime.end + timedelta(days=10), datetime.min.time()),
                ),
            )
            rows = cur.fetchall()
    return [
        {
            'run_id': r[0],
            'started_at': r[1],
            'completed_at': r[2],
            'ps_start': r[3],
            'ps_end': r[4],
            'n_snapshots': r[5],
        }
        for r in rows
    ]


def _best_run_for_regime(regime: Regime) -> str | None:
    """
    Auto-discover the best-trial run ID for a regime by finding the run
    whose portfolio_snapshots span best matches the regime window.

    Prefers runs with completed_at set; falls back to partial runs.
    Returns the run_id of the most recent matching run, or None.
    """
    candidates = _list_runs_in_window(regime)
    if not candidates:
        return None

    # Filter: ps_start must be within tolerance of regime.start
    # and ps_end must reach at least 50% through the regime window.
    regime_days = (regime.end - regime.start).days
    min_ps_end = regime.start + timedelta(days=int(regime_days * 0.5))

    valid = [
        c for c in candidates
        if c['ps_start'] is not None
        and c['ps_start'] <= regime.start + timedelta(days=_WINDOW_TOLERANCE_DAYS)
        and c['ps_end'] is not None
        and c['ps_end'] >= min_ps_end
    ]

    if not valid:
        return None

    # Prefer completed runs
    completed = [c for c in valid if c['completed_at'] is not None]
    pool = completed if completed else valid
    return pool[0]['run_id']


def _load_best_trial_result(regime: Regime, run_id: str | None) -> RegimeResult:
    """
    Reconstruct a RegimeResult for the best-trial run from DB metrics.
    If run_id is None, attempt auto-discovery.
    """
    result = RegimeResult(regime=regime, params_label='best_trial')

    if run_id is None:
        run_id = _best_run_for_regime(regime)

    if run_id is None:
        result.skipped = True
        result.skip_reason = 'best-trial run not found in DB for this regime'
        logger.warning('[baseline_only] No best-trial run found for %s — skipping.', regime.name)
        return result

    result.run_id = run_id
    logger.info('[baseline_only] Using best-trial run %s for %s', run_id, regime.name)
    _fill_metrics(result)
    return result


# ---------------------------------------------------------------------------
# failed_tickers snapshot / restore
# ---------------------------------------------------------------------------

def _snapshot_failed_tickers() -> list[dict]:
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute('SELECT symbol, reason, failed_at FROM failed_tickers')
                return [{'symbol': r[0], 'reason': r[1], 'failed_at': r[2]} for r in cur.fetchall()]
            except psycopg2.errors.UndefinedTable:
                return []


def _restore_failed_tickers(snapshot: list[dict]) -> None:
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
# List-runs helper (--list-runs flag)
# ---------------------------------------------------------------------------

def cmd_list_runs() -> None:
    print()
    for regime in TARGET_REGIMES:
        print(f'  === {regime.name} ({regime.start} → {regime.end}) ===')
        rows = _list_runs_in_window(regime)
        if not rows:
            print('    (no matching runs)\n')
            continue
        print(f'  {"run_id":<36}  {"started_at":<26}  {"completed_at":<26}  '
              f'{"ps_start":<12}  {"ps_end":<12}  {"snapshots":>9}')
        print('  ' + '-' * 120)
        for r in rows[:10]:
            cpl = str(r['completed_at'])[:25] if r['completed_at'] else 'NULL'
            print(f'  {r["run_id"]:<36}  {str(r["started_at"])[:25]:<26}  {cpl:<26}  '
                  f'{str(r["ps_start"]):<12}  {str(r["ps_end"]):<12}  {r["n_snapshots"]:>9}')
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    calm_bull_id: str | None = None,
    vol_shock_id: str | None = None,
    sideways_id: str | None = None,
    include_cold: bool = False,
) -> None:
    logger.info('=== Phase 3 — Targeted 3-regime baseline comparison ===')
    warm_only = not include_cold

    id_map = {
        'calm_bull_2017': calm_bull_id,
        'vol_shock_2020': vol_shock_id,
        'sideways_2022':  sideways_id,
    }

    # --- Reconstruct best-trial results from DB ---
    logger.info('Loading best-trial results from DB...')
    result_best = BatteryResult(
        params_label='best_trial',
        params={},  # not re-running, so params are informational only
    )
    for regime in TARGET_REGIMES:
        rr = _load_best_trial_result(regime, id_map.get(regime.name))
        result_best.regime_results.append(rr)
        if not rr.skipped and rr.run_id:
            logger.info(
                '  %s: run=%s  ret=%.1f%%  score=%s',
                regime.name, rr.run_id,
                rr.return_pct or 0,
                f'{rr.score:.4f}' if rr.score is not None else 'N/A',
            )

    completed_best = result_best.completed()
    if not completed_best:
        logger.error('No best-trial results found. Run --list-runs to find run IDs, '
                     'then pass them via --calm-bull-id / --vol-shock-id / --sideways-id.')
        sys.exit(1)

    logger.info('%d best-trial regime(s) loaded.', len(completed_best))

    # --- Run baseline for the same regimes ---
    base_params = defaults()
    failed_snapshot = _snapshot_failed_tickers()

    try:
        logger.info('Running battery: BASELINE (label="baseline") for %d regime(s)...',
                    len(TARGET_REGIMES))
        result_base = run_battery(
            params=base_params,
            params_label='baseline',
            budget=BUDGET,
            warm_only=warm_only,
            regimes=TARGET_REGIMES,
        )
    except KeyboardInterrupt:
        logger.warning('Baseline battery interrupted.')
        raise
    finally:
        _restore_failed_tickers(failed_snapshot)

    # --- Report (re-uses battery.py's print_report which iterates REGIMES) ---
    # Pad result_best with SKIP entries for the 2 non-target regimes so
    # print_report's full-regime table renders correctly.
    target_names = {r.name for r in TARGET_REGIMES}
    for regime in REGIMES:
        if regime.name not in target_names:
            result_best.regime_results.append(
                RegimeResult(regime=regime, params_label='best_trial',
                             skipped=True, skip_reason='not in targeted 3-regime run')
            )
            result_base.regime_results.append(
                RegimeResult(regime=regime, params_label='baseline',
                             skipped=True, skip_reason='not in targeted 3-regime run')
            )

    print_report(result_best, result_base)

    # Gate: need ≥ 2 wins out of 3 completed (lower bar for partial run).
    passed = gate_check(
        result_good=result_best,
        result_bad=result_base,
        min_wins=2,
    )

    if passed:
        logger.info('Phase 3 gate PASSED — ready to proceed to Phase 4.')
    else:
        logger.warning('Phase 3 gate NOT YET MET.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 3: 3-regime baseline comparison (skips re-running best-trial).',
    )
    parser.add_argument('--calm-bull-id',  default=None, metavar='RUN_ID',
                        help='Run ID of the calm_bull_2017 best-trial backtest.')
    parser.add_argument('--vol-shock-id',  default=None, metavar='RUN_ID',
                        help='Run ID of the vol_shock_2020 best-trial backtest.')
    parser.add_argument('--sideways-id',   default=None, metavar='RUN_ID',
                        help='Run ID of the sideways_2022 best-trial backtest.')
    parser.add_argument('--include-cold', action='store_true',
                        help='Run baseline for all regimes regardless of DB warmth check '
                             '(needed when Lumibot\'s local cache has the data but stock_prices does not).')
    parser.add_argument('--list-runs', action='store_true',
                        help='List DB runs in each regime window and exit.')
    args = parser.parse_args()

    if args.list_runs:
        cmd_list_runs()
        sys.exit(0)

    run(
        calm_bull_id=args.calm_bull_id,
        vol_shock_id=args.vol_shock_id,
        sideways_id=args.sideways_id,
        include_cold=args.include_cold,
    )
