import BobsBrain
from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting


if __name__ == '__main__':
    backtesting_start = datetime(2025, 11, 1)
    backtesting_end = datetime(2025, 12, 31)
    result = BobsBrain.Strategy.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        budget=10000
    )

    print(result)


