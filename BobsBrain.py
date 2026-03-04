import itertools
import os
import json
from datetime import datetime, timedelta

from lumibot.strategies import Strategy

from StockEvaluator import StockEvaluator
from YahooDBReader import YahooDBReader


class BobsBrain(Strategy):
    """
    Executes the day's trades using a lead/lag correlation strategy.

    For each day:
    - Existing positions are re-evaluated: if correlation with their lead stock
      has dropped, a moving average crossover on the lag stock determines whether
      to sell. If correlation is still healthy, the lead stock's MA crossover
      drives the action.
    - New pairs are discovered across all exchange-listed stocks and added up to
      max_daily_candidates per day.
    - Sells are submitted first, then buys with an equal budget split.
    """

    def initialize(self):
        self.sleeptime = '1D'
        self.min_correlation = 0.8
        self.lookback_window = 60
        self.max_daily_candidates = 10
        self.max_lag = 5

        self.file_path = os.path.join(os.path.dirname(__file__), "pairs", "pair_history.json")
        self.pairs = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                self.pairs = json.load(f)

    def on_trading_iteration(self):
        stock_evaluator = StockEvaluator()

        # --- Update actions for existing positions ---
        positions = self.get_positions()
        for position in positions:
            symbol = position.symbol
            if symbol not in self.pairs:
                print(f"Warning: open position {symbol} not found in pairs dict, skipping.")
                continue

            pair = self.pairs[symbol]
            lead_data = self.get_historical_prices(pair['lead_stock'], self.lookback_window, "1d").df['close']
            lag_data = self.get_historical_prices(pair['lag_stock'], self.lookback_window, "1d").df['close']

            pair['corr'] = stock_evaluator.get_correlation(lead_data, lag_data, pair['lag'])

            if pair['corr'] < self.min_correlation:
                short_ma = lag_data.rolling(window=pair['short_ma'], min_periods=1).mean()
                long_ma = lag_data.rolling(window=pair['long_ma'], min_periods=1).mean()
                pair['action'] = 'sell' if short_ma.iloc[-1] < long_ma.iloc[-1] else 'hold'
            else:
                pair['action'] = stock_evaluator.get_action(
                    lead_data, lag_data, pair['lag'],
                    short_ma=pair['short_ma'], long_ma=pair['long_ma']
                )

        # --- Discover new pairs ---
        start_date = datetime.now() - timedelta(days=self.lookback_window)
        end_date = datetime.now()
        yahoo_reader = YahooDBReader()
        stock_data = yahoo_reader.get_all_stocks(start_date=start_date, end_date=end_date)

        new_candidates = 0
        position_symbols = [p.symbol for p in self.get_positions()]

        for stock1, stock2 in itertools.combinations(stock_data.columns, 2):
            if new_candidates >= self.max_daily_candidates:
                break

            # skip if we're already tracking the lag stock
            if stock2 in self.pairs or stock2 in position_symbols:
                continue

            correlation = stock_evaluator.get_correlation(stock_data[stock1], stock_data[stock2], lag=1)
            if correlation < self.min_correlation:
                continue

            self.pairs[stock2] = {
                'lead_stock': stock1,
                'lag_stock': stock2,
                'lag': 1,
                'short_ma': 2,
                'long_ma': 5,
                'corr': correlation,
                'action': 'hold',
            }
            new_candidates += 1

        # --- Execute sells first ---
        to_remove = []
        for symbol, pair in self.pairs.items():
            if pair['action'] == 'sell':
                position = self.get_position(symbol)
                if position and position.quantity > 0:
                    order = self.create_order(symbol, position.quantity, 'sell')
                    self.submit_order(order)
                to_remove.append(symbol)

        for symbol in to_remove:
            self.pairs.pop(symbol)

        # --- Execute buys ---
        buy_pairs = [
            pair for symbol, pair in self.pairs.items()
            if pair['action'] == 'buy' and symbol not in [p.symbol for p in self.get_positions()]
        ]

        if buy_pairs:
            budget = self.get_cash() / 2
            per_stock_budget = budget / len(buy_pairs)
            for pair in buy_pairs:
                price = self.get_last_price(pair['lag_stock'])
                if price and price > 0:
                    quantity = int(per_stock_budget / price)
                    if quantity > 0:
                        order = self.create_order(pair['lag_stock'], quantity, 'buy')
                        self.submit_order(order)
