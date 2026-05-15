#!/usr/bin/env python3
"""
Combined validation: runs all four H5-param backtests in the correct order to
ensure cache-sharing within each date window.

Run order (intentional):
  1. baseline sideways  — warms stock_prices for the 2022 window
  2. H1 fixed sideways  — served from cache; same price data as run 1
  3. baseline bull      — warms stock_prices for the 2023 window
  4. H1 fixed bull      — served from cache; same price data as run 3

This ordering maximises same-window cache reuse, making the baseline vs H1
cointegration comparison as clean as possible.

Usage (from repo root, with .env loaded):

    python scripts/run_combined_validation_backtests.py
    # or backgrounded:
    caffeinate -i python scripts/run_combined_validation_backtests.py &
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

_BASE_PARAMS: dict = {
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
}

# (label, start, end, enable_short_leg, description)
RUNS: list[tuple[str, datetime, datetime, bool, str]] = [
    (
        '1_sideways_baseline',
        datetime(2022, 2, 1),
        datetime(2022, 4, 30),
        False,
        'Sideways 2022 — long-only baseline (warms 2022 cache)',
    ),
    (
        '2_sideways_h1_fixed',
        datetime(2022, 2, 1),
        datetime(2022, 4, 30),
        True,
        'Sideways 2022 — H1 fixed (served from 2022 cache)',
    ),
    (
        '3_bull_baseline',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        False,
        'Bull 2023 — long-only baseline (warms 2023 cache)',
    ),
    (
        '4_bull_h1_fixed',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        True,
        'Bull 2023 — H1 fixed (served from 2023 cache)',
    ),
]


def main() -> None:
    for label, start, end, short_leg, description in RUNS:
        params = {**_BASE_PARAMS, 'enable_short_leg': short_leg}
        print('=' * 72)
        print(f'[{label}] {start.date()} -> {end.date()}')
        print(description)
        print(f'enable_short_leg={short_leg}, budget=10000')
        print('=' * 72, flush=True)
        result = BobsBrain.backtest(
            YahooDataBacktesting,
            start,
            end,
            budget=10000,
            parameters=params,
            show_plot=False,
            show_tearsheet=False,
            save_tearsheet=False,
        )
        print(f'RESULT [{label}]:', result, flush=True)
    print('Combined validation backtests finished.', flush=True)


if __name__ == '__main__':
    main()
