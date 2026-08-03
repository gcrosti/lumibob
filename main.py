import os
from datetime import datetime

from dotenv import load_dotenv

from BobsBrain import BobsBrain

load_dotenv()

RUN_MODE = os.getenv('RUN_MODE', 'backtest')

# Strategy parameters (same keys as BobsBrain.initialize() defaults; omitted keys use those defaults).
STRATEGY_PARAMETERS = {
    # --- Position sizing ---
    'max_k': 20,                    # hard ceiling on target portfolio size (Tier 2 tunable)
    'min_position_pct': 0.03,
    'max_position_pct': 0.20,
    'target_deployed_pct': 0.60,

    # --- Signal ---
    'entry_threshold': 2.0,
    'exit_threshold': 0.5,
    'zscore_window': 20,

    # --- Entry criteria ---
    # Candidates are gated on direction + dislocation (z <= -entry_threshold)
    # and on trade magnitude, then ranked by expected_gross_bps. The composite
    # score was removed 2026-08-01 (it did not select positively; see
    # docs/plans/2026-08-01_entry-criteria-overhaul.md).
    'min_expected_gross_bps': 25.0,  # emergency floor, bps of gross notional

    # --- Observability windows (not used for selection) ---
    'corr_long_window': 90,
    'corr_short_window': 20,
    'max_halflife_days': 60,

    # --- Execution (H1 dollar-neutral) ---
    # short_leg_fraction in [0.0, 1.0]: fraction of long notional to short the lead.
    # 0.0 = long-only; 1.0 = full dollar-neutral hedge.  Replaces enable_short_leg.
    'short_leg_fraction': 0.0,

    # --- Discovery ---
    'max_daily_candidates': 200,    # qualified (gates passed) candidates per day
    'max_daily_examined': 2000,     # pairs examined per day (cheap pre-gate scan)
    'cooldown_days': 7,

    # --- Filters ---
    'penny_threshold': 5.0,         # minimum last-close price to pass penny gate

    # --- Clustering / HDBSCAN ---
    'cluster_lookback_days': 126,               # calendar days of price history for clustering
    'hdbscan_min_cluster_size': 5,              # minimum tickers to form a cluster
    'hdbscan_min_samples': 2,                   # HDBSCAN density parameter; lower = less noise
    'pca_variance': 0.95,                       # fraction of variance retained after PCA (euclidean path + Ward fallback)
    'min_coverage': 0.5,                        # min fraction of non-NaN bars to keep a ticker
    'hdbscan_metric': 'precomputed',            # 'precomputed' (1-corr distance, recommended) or 'euclidean'
    'hdbscan_selection_method': 'eom',          # 'eom' (larger) or 'leaf' (finer) clusters
    'hdbscan_cluster_selection_epsilon': 0.0,   # merge clusters closer than this distance
    'min_intra_cluster_corr': 0.3,              # sanity gate: dissolve clusters below this median intra-corr
}

if __name__ == '__main__':
    if RUN_MODE == 'paper':
        from lumibot.brokers import Alpaca
        from lumibot.traders import Trader

        ALPACA_CONFIG = {
            'API_KEY':    os.getenv('ALPACA_API_KEY'),
            'API_SECRET': os.getenv('ALPACA_API_SECRET'),
            'PAPER':      os.getenv('ALPACA_IS_PAPER', 'true').lower() == 'true',
        }
        broker = Alpaca(ALPACA_CONFIG)
        strategy = BobsBrain(
            broker=broker,
            parameters=STRATEGY_PARAMETERS,
        )
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()

    else:
        from lumibot.backtesting import YahooDataBacktesting

        # WS1 smoke test: 2-week window inside price-cache coverage.
        backtesting_start = datetime(2024, 3, 4)
        backtesting_end = datetime(2024, 3, 18)
        result = BobsBrain.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=10000,
            parameters=STRATEGY_PARAMETERS,
            show_plot=False,
            show_tearsheet=False,
            save_tearsheet=False,
        )
        print(result)
