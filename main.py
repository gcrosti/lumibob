import os
from datetime import datetime

from dotenv import load_dotenv

from BobsBrain import BobsBrain

load_dotenv()

RUN_MODE = os.getenv('RUN_MODE', 'backtest')

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
            parameters={
                'ticker_limit': 100,
                'min_daily_pairs': 10,
                'min_position_pct': 0.03,
                'max_position_pct': 0.20,
                'target_deployed_pct': 0.60,
            },
        )
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()

    else:
        from lumibot.backtesting import YahooDataBacktesting

        backtesting_start = datetime(2024, 1, 2)
        backtesting_end = datetime(2024, 3, 27)
        result = BobsBrain.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=10000,
            parameters={
                'ticker_limit': 100,
                'min_daily_pairs': 10,
                'min_position_pct': 0.03,
                'max_position_pct': 0.20,
                'target_deployed_pct': 0.60,
            },
            show_plot=False,
            show_tearsheet=False,
            save_tearsheet=False,
        )
        print(result)
