from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting

from BobsBrain import BobsBrain


if __name__ == '__main__':
    backtesting_start = datetime(2025, 11, 1)
    backtesting_end = datetime(2025, 11, 14)
    result = BobsBrain.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        budget=10000,
        parameters={'ticker_limit': 30},
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
    )

    print(result)


