#!/usr/bin/env python3
"""
H1 short-leg validation: same date windows and params as H5 baselines 0ec7cc / 691011,
with enable_short_leg=True. Compare new run_id rows in backtest_runs / portfolio_snapshots / trades.

Usage (from repo root, with .env loaded):

    python scripts/run_h1_validation_backtests.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv

load_dotenv(_REPO / '.env')

from lumibot.backtesting import YahooDataBacktesting

from BobsBrain import BobsBrain

# Mirrors main.py STRATEGY_PARAMETERS + H5 weights; matches DB settings on 0ec7cc / 691011.
PARAMS: dict = {
    'lookback_window': 130,
    'cluster_recompute_days': None,
    'max_k': 20,
    'min_position_pct': 0.03,
    'max_position_pct': 0.20,
    'target_deployed_pct': 0.60,
    'entry_threshold': 2.0,
    'exit_threshold': 0.5,
    'zscore_window': 20,
    'corr_long_window': 90,
    'corr_short_window': 20,
    'w_corr_long': 0.2143,
    'w_corr_short': 0.3571,
    'w_z_depth': 0.1429,
    'w_coint': 0.1786,
    'w_halflife': 0.1071,
    'max_halflife_days': 60,
    'max_daily_candidates': 200,
    'cooldown_days': 7,
    'penny_threshold': 5.0,
    'quality_scale_pivot': 0.7,
    'quality_scale_min': 0.5,
    'quality_scale_max': 1.0,
    'cluster_lookback_days': 126,
    'hdbscan_min_cluster_size': 5,
    'hdbscan_min_samples': 2,
    'pca_variance': 0.95,
    'min_coverage': 0.5,
    'hdbscan_metric': 'precomputed',
    'hdbscan_selection_method': 'eom',
    'hdbscan_cluster_selection_epsilon': 0.0,
    'min_intra_cluster_corr': 0.3,
    'short_leg_fraction': 1.0,  # full dollar-neutral hedge (was enable_short_leg=True)
}

RUNS: list[tuple[str, datetime, datetime, str]] = [
    (
        'A_sideways_2022_q1',
        datetime(2022, 2, 1),
        datetime(2022, 4, 30),  # exclusive end — baselines ran through 2022-04-29
        'Baseline 0ec7cc: +1.59% vs SPY (long-only, H5)',
    ),
    (
        'B_calm_bull_2023_q2',
        datetime(2023, 4, 1),   # matches main.py used for 691011 (Apr 1 = Sat, first trading day = Apr 3)
        datetime(2023, 6, 30),  # exclusive end — baselines ran through 2023-06-29
        'Baseline 691011: -7.77% vs SPY (long-only, H5)',
    ),
]


def main() -> None:
    for label, start, end, baseline_note in RUNS:
        print('=' * 72)
        print(label, start.date(), '->', end.date())
        print('Baseline note:', baseline_note)
        print('short_leg_fraction=1.0, budget=10000')
        print('=' * 72, flush=True)
        result = BobsBrain.backtest(
            YahooDataBacktesting,
            start,
            end,
            budget=10000,
            parameters=PARAMS,
            show_plot=False,
            show_tearsheet=False,
            save_tearsheet=False,
        )
        print(f'RESULT {label}:', result, flush=True)
    print('H1 validation backtests finished.', flush=True)


if __name__ == '__main__':
    main()
