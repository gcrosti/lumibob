import itertools
import math
import os
import secrets
from datetime import datetime, timedelta

from dotenv import load_dotenv
from lumibot.strategies import Strategy

from AlpacaClient import AlpacaClient
from DatabaseClient import DatabaseClient
from StockDataCache import StockDataCache
from StockEvaluator import StockEvaluator

load_dotenv()


class BobsBrain(Strategy):
    """
    Executes the day's trades using a lead/lag correlation strategy.

    Lifecycle per trading day:
    - before_market_opens(): evaluate all pairs once — update correlations,
      determine actions, discover new candidate pairs. Results are stored in
      self.pairs so that on_trading_iteration() can act on them.
    - on_trading_iteration(): execute the queued sells and buys. Keeping
      execution here (separate from evaluation) allows sleeptime to be reduced
      below '1D' in future so orders can be spread across multiple intraday
      iterations without re-running the expensive evaluation step.

    Requires DB_URL, ALPACA_API_KEY, and ALPACA_API_SECRET to be set (via .env
    or environment). Raises EnvironmentError on startup if any are missing.
    """

    def initialize(self):
        self.sleeptime = '1D'
        self.min_correlation = 0.8
        self.lookback_window = 60
        self.max_daily_candidates = 10
        self.max_lag = 5
        self.ticker_limit = self.parameters.get('ticker_limit', None)
        self._run_mode = os.getenv('RUN_MODE', 'backtest')

        self._spy_start_price = None
        self._starting_portfolio_value = None

        db_url = os.getenv('DB_URL')
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_API_SECRET')

        missing = [name for name, val in [
            ('DB_URL', db_url),
            ('ALPACA_API_KEY', api_key),
            ('ALPACA_API_SECRET', secret_key),
        ] if not val]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the values."
            )

        self._db = DatabaseClient(db_url)
        self._alpaca = AlpacaClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=os.getenv('ALPACA_IS_PAPER', 'true').lower() == 'true',
            mode=self._run_mode,
        )
        self._cache = StockDataCache(self._db, self._alpaca)

        self._run_id = secrets.token_hex(3)
        self.pairs = self._db.load_active_pairs()
        self._db.create_run(
            run_id=self._run_id,
            mode=self._run_mode,
            settings={
                'ticker_limit': self.ticker_limit,
                'lookback_window': self.lookback_window,
                'min_correlation': self.min_correlation,
                'max_daily_candidates': self.max_daily_candidates,
            },
        )

    def before_market_opens(self):
        """
        Runs once per trading day before any iterations.
        Updates correlations and actions for existing positions, then
        discovers new candidate pairs up to max_daily_candidates.
        """
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

            corr = stock_evaluator.get_correlation(lead_data, lag_data, pair['lag'])
            if math.isnan(corr):
                print(f"Warning: NaN correlation for existing pair {symbol}, defaulting to 0.")
                corr = 0.0

            pair['corr'] = corr
            if pair.get('pair_id'):
                self._db.update_pair_correlation(pair['pair_id'], corr)

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

        tickers = self._db.get_tickers()
        if not tickers:
            # First run or after a nightly refresh — populate tickers from Alpaca
            print("Tickers table empty, fetching tradeable assets from Alpaca...")
            tickers = self._alpaca.get_tradeable_assets()
            self._db.upsert_tickers(tickers, 'ALPACA')

        if self.ticker_limit:
            tickers = tickers[:self.ticker_limit]

        stock_data = self._cache.get_prices(tickers, start_date, end_date)

        if stock_data.empty:
            print("Warning: no price data available for pair discovery.")
            return

        new_candidates = 0
        position_symbols = [p.symbol for p in self.get_positions()]

        for stock1, stock2 in itertools.combinations(stock_data.columns, 2):
            if new_candidates >= self.max_daily_candidates:
                break

            if stock2 in self.pairs or stock2 in position_symbols:
                continue

            correlation = stock_evaluator.get_correlation(stock_data[stock1], stock_data[stock2], lag=1)
            if math.isnan(correlation) or correlation < self.min_correlation:
                continue

            action = stock_evaluator.get_action(
                stock_data[stock1], stock_data[stock2], lag=1, short_ma=2, long_ma=5
            )
            if action != 'buy':
                continue

            print(f"Adding new pair: {stock1} -> {stock2} with correlation {correlation:.4f}, action={action}")

            new_pair = {
                'lead_stock': stock1,
                'lag_stock':  stock2,
                'lag':        1,
                'short_ma':   2,
                'long_ma':    5,
                'corr':       correlation,
                'action':     action,
            }
            new_pair['pair_id'] = self._db.save_pair(new_pair)
            self.pairs[stock2] = new_pair
            new_candidates += 1

    def on_trading_iteration(self):
        """
        Executes queued orders based on actions set by before_market_opens().
        Separated from evaluation so that sleeptime can later be reduced to
        allow spreading orders across multiple intraday iterations.
        """
        now = self.get_datetime()

        # --- Execute sells first ---
        to_remove = []
        for symbol, pair in self.pairs.items():
            if pair['action'] == 'sell':
                position = self.get_position(symbol)
                if position and position.quantity > 0:
                    order = self.create_order(symbol, position.quantity, 'sell')
                    self.submit_order(order)
                    price = self.get_last_price(symbol)
                    if price and price > 0:
                        self._db.log_trade(
                            run_id=self._run_id,
                            symbol=symbol,
                            side='sell',
                            quantity=float(position.quantity),
                            price=float(price),
                            filled_at=now,
                            pair_id=pair.get('pair_id'),
                        )
                    else:
                        print(f"Warning: could not log sell trade for {symbol} — price unavailable.")
                self._db.deactivate_pair(symbol)
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
                        self._db.log_trade(
                            run_id=self._run_id,
                            symbol=pair['lag_stock'],
                            side='buy',
                            quantity=float(quantity),
                            price=float(price),
                            filled_at=now,
                            pair_id=pair.get('pair_id'),
                        )

        # --- Log indicators ---
        portfolio_value = self.portfolio_value
        if self._starting_portfolio_value is None:
            self._starting_portfolio_value = portfolio_value

        spy_price = self.get_last_price("SPY")
        spy_value = None
        if spy_price:
            if self._spy_start_price is None:
                self._spy_start_price = spy_price
            spy_value = round((spy_price / self._spy_start_price) * self._starting_portfolio_value, 2)
            self.add_line("spy_value", spy_value)

        active_pairs = list(self.pairs.values())
        avg_corr = (
            sum(p['corr'] for p in active_pairs) / len(active_pairs)
            if active_pairs else 0.0
        )

        self.add_line("active_pairs", float(len(active_pairs)))
        self.add_line("avg_corr",     round(avg_corr, 4))
        self.add_line("cash_ratio",   round(self.cash / portfolio_value, 4))
        self.add_line("daily_buys",   float(len(buy_pairs)))
        self.add_line("daily_sells",  float(len(to_remove)))

        self._db.log_snapshot(
            run_id=self._run_id,
            time=now,
            portfolio_value=float(portfolio_value),
            cash=float(self.cash),
            spy_value=spy_value,
            active_pairs=len(active_pairs),
            avg_correlation=round(avg_corr, 4),
            cash_ratio=round(self.cash / portfolio_value, 4),
            daily_buys=len(buy_pairs),
            daily_sells=len(to_remove),
        )

    def on_strategy_end(self):
        self._db.close_run(self._run_id)
