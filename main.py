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

    # --- Scoring ---
    'corr_long_window': 90,
    'corr_short_window': 20,
    # 5-component composite score — normalised defaults (raw: 0.3, 0.5, 0.2, 0.25, 0.15 → ÷1.4).
    # BobsBrain uses these as-is; call tuning.parameter_space.normalize_weights() if adjusting.
    'w_corr_long': 0.2143,
    'w_corr_short': 0.3571,
    'w_z_depth': 0.1429,
    'w_coint': 0.1786,
    'w_halflife': 0.1071,
    'max_halflife_days': 60,

    # --- Execution (H1 dollar-neutral) ---
    'enable_short_leg': False,      # True: short lead in equal notional on each pair entry

    # --- Discovery ---
    'max_daily_candidates': 200,
    'cooldown_days': 7,

    # --- Filters ---
    'penny_threshold': 5.0,         # minimum last-close price to pass penny gate

    # --- Dynamic-K quality scale ---
    'quality_scale_pivot': 0.7,     # pool_corr is divided by this to get the raw scale
    'quality_scale_min': 0.5,       # floor on quality_scale multiplier (K >= max_k * this)
    'quality_scale_max': 1.0,       # ceiling on quality_scale multiplier (must be <= 1.0 to honour max_k hard ceiling)

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

        backtesting_start = datetime(2024, 1, 2)
        backtesting_end = datetime(2024, 3, 26)
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
