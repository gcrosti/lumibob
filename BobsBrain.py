import itertools
from datetime import datetime

from lumibot.strategies import Strategy
import json
import os

from pycodestyle import continued_indentation

from StockEvaluator import StockEvaluator
from YahooDBReader import YahooDBReader


class BobsBrain(Strategy):
    """
    executes the day's trades
    """

    # Next steps
    # 1. finish Bob's brain w standard lag
    # 2. run backtest and debug
    # 3. create 'optimization mode' so you can call bobs brain in main to optimize a stock pair
    # 3. research appropriate error handling in Lumibot

    def initialize(self):
        # presets
        self.sleeptime = '1D' # trades happen 1/day
        self.min_correlation = 0.8 # min correlation threshold
        self.lookback_window = 60 # num trailing days during which correlation is evaluated
        self.max_daily_candidates = 10 # max number of new positions to enter daily
        self.max_lag = 5 # the max lag between lead and lag stock to be evaluated

        # load data
        positions = self.get_positions()
        self.file_path = "PyCharmProjects/LumiBob/pairs/pair_history.json"
        self.pairs = {} # dict of pairs (dict) keyed on lag stock
        if os.path.exists(self.file_path):
            with open(self.data_file, "r") as f:
                self.pairs = json.load(f)

        # update corrs for existing positions
        stock_evaluator = StockEvaluator()
        for position in positions:
            if position.symbol not in self.pairs.keys():
                print("oops! Could not find existing position in pairs")
                print(position.symbol)
                continue
            pair = self.pairs[position.symbol]

            lead_stock = self.get_historical_prices(pair['lead_stock'],self.lookback_window,"1d").df['close']
            lag_stock = self.get_historical_prices(pair['lag_stock'], self.lookback_window, "1d").df['close']
            self.pairs[position.symbol]['corr'] = stock_evaluator.get_correlation(lead_stock, lag_stock, pair['lag'])


            # if correlation has dropped below threshold, using moving averages to determine whether to sell
            if self.pairs[position.symbol]['corr'] < self.min_correlation:
                short_ma = lag_stock.rolling(window=pair['short_ma'], min_periods=1).mean()
                long_ma = lag_stock.rolling(window=pair['long_ma'], min_periods=1).mean()
                if short_ma[-1] < long_ma[-1]:
                    self.pairs[position.symbol]['action'] = 'sell'

            # else, use lead stock to determine action
            else:
                self.pairs[position.symbol]['action'] = stock_evaluator.get_action(lead_stock, lag_stock, pair['lag'])


        # generate new pairs to invest in
        yahoo_reader = YahooDBReader()
        start_date = datetime.datetime.now() - datetime.timedelta(days = self.lookback_window)
        end_date = datetime.datetime.now()
        stock_data = yahoo_reader.get_all_stocks(start_date=start_date, end_date=end_date)
        for stock1, stock2 in itertools.combinations(stock_data, 2):
            correlation = stock_evaluator.get_correlation(stock_data[stock1], stock_data[stock2], lag = 1)

            # continue to next pair if correlation is too low
            if correlation < self.min_correlation:
                continue

            # continue to next pair if already invested in lag stock
            if stock2 in self.pairs.keys():
                continue




    def on_trading_iteration(self):
        # distribute budget to stock candidates
        budget = self.get_cash() / 2

        # perform action for each pair
        for pair in self.pairs.values():
            if pair['action'] == 'sell':
                self.sell_all(pair['lag_stock'])
                self.pairs.pop(pair['lag_stock'])

            positions = self.get_positions()
            position_symbols = [p.symbol for p in positions]

            if pair['action'] == 'buy' and pair['symbol'] not in position_symbols:
                # buy

                continue


