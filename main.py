import os
from datetime import datetime

from dotenv import load_dotenv

from BobsBrain import BobsBrain

load_dotenv()

RUN_MODE = os.getenv('RUN_MODE', 'backtest')

# Strategy parameters (same keys as BobsBrain.initialize() defaults; omitted keys use those defaults).
STRATEGY_PARAMETERS = {
    # Position size bounds as a fraction of portfolio (tied to composite score).
    'min_position_pct': 0.03,
    'max_position_pct': 0.20,
    # Target fraction of capital deployed; gap boosts each buy size.
    'target_deployed_pct': 0.60,
    # Long- vs short-horizon correlation windows in bars (log returns).
    'corr_long_window': 90,
    'corr_short_window': 20,
    # Weights for composite score: long corr, short corr, z-depth (typically sum to 1).
    'w_corr_long': 0.3,
    'w_corr_short': 0.5,
    'w_z_depth': 0.2,
    # Cap on new candidate pairs scored per trading day.
    'max_daily_candidates': 200,
    # Minimum days before re-scoring the same unordered pair.
    'cooldown_days': 7,
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
