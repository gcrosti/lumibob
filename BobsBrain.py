import itertools
import math
import os
import random
import secrets
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
from lumibot.strategies import Strategy

from AlpacaClient import AlpacaClient
from DatabaseClient import DatabaseClient
from PairSimulator import PairSimulator
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
        self.min_correlation = 0.9
        self.lookback_window = 60
        self.min_daily_pairs = self.parameters.get('min_daily_pairs', 10)
        self.max_lag = 5
        self.max_position_multiplier = self.parameters.get('max_position_multiplier', 3.0)
        self.top_up_rate = self.parameters.get('top_up_rate', 0.5)
        self.ticker_limit = self.parameters.get('ticker_limit', None)
        self._run_mode = os.getenv('RUN_MODE', 'backtest')

        self._spy_start_price = None
        self._starting_portfolio_value = None
        self._pairs_scanned = 0
        self._candidates_found = 0
        self._candidates_buy_ready = 0

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
        self._db.migrate_pairs_simulated_return()
        self._alpaca = AlpacaClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=os.getenv('ALPACA_IS_PAPER', 'true').lower() == 'true',
            mode=self._run_mode,
        )
        self._cache = StockDataCache(self._db, self._alpaca)

        self._run_id = secrets.token_hex(3)
        self.pairs = self._db.load_active_pairs(self._run_id)
        self._db.create_run(
            run_id=self._run_id,
            mode=self._run_mode,
            settings={
                'ticker_limit': self.ticker_limit,
                'lookback_window': self.lookback_window,
                'min_correlation': self.min_correlation,
                'min_daily_pairs': self.min_daily_pairs,
                'max_position_multiplier': self.max_position_multiplier,
                'top_up_rate': self.top_up_rate,
            },
        )

    def before_market_opens(self):
        """
        Runs once per trading day before any iterations.
        Updates correlations and actions for existing positions, then
        discovers new candidate pairs up to max_daily_candidates.

        Price data is fetched lazily: each symbol is pulled from StockDataCache
        the first time it is needed and stored in a per-call cache so that the
        same symbol is never fetched twice within a single execution. This
        avoids the previous approach of bulk-fetching the entire ticker universe
        upfront regardless of how many pairs are ultimately needed.

        Both the existing-positions update and the discovery loop share the
        same cache, so symbols used in open pairs are free for any subsequent
        discovery combination that happens to include them.
        """
        stock_evaluator = StockEvaluator()
        simulator = PairSimulator()

        end_date = self.get_datetime()
        start_date = end_date - timedelta(days=self.lookback_window)

        # Per-call price series cache shared across the whole method.
        _series_cache: dict[str, pd.Series | None] = {}

        def _get_series(symbol: str) -> pd.Series | None:
            """Return the close-price series for symbol, fetching once if needed."""
            if symbol not in _series_cache:
                df = self._cache.get_prices([symbol], start_date, end_date)
                if df.empty or symbol not in df.columns:
                    _series_cache[symbol] = None
                else:
                    series = df[symbol].dropna()
                    _series_cache[symbol] = series if not series.empty else None
            return _series_cache[symbol]

        # --- Update actions for existing positions ---
        for position in self.get_positions():
            symbol = position.symbol
            if symbol not in self.pairs:
                print(f"Warning: open position {symbol} not found in pairs dict, skipping.")
                continue

            pair = self.pairs[symbol]

            # Always fetch the lag stock — needed for the MA fallback regardless.
            lag_data = _get_series(pair['lag_stock'])
            if lag_data is None:
                pair['action'] = 'sell'
                continue

            # If yesterday's stored correlation is already below the threshold,
            # use the lag stock's own MA crossover to decide without fetching
            # the lead stock at all — no point pulling data for a pair we are
            # likely to exit anyway.
            # Default 1.0 when 'corr' is absent: treat as above-threshold so
            # that the lead stock is always fetched for pairs with unknown corr.
            if pair.get('corr', 1.0) < self.min_correlation:
                short_ma = lag_data.rolling(window=pair['short_ma'], min_periods=1).mean()
                long_ma = lag_data.rolling(window=pair['long_ma'], min_periods=1).mean()
                pair['action'] = 'sell' if short_ma.iloc[-1] < long_ma.iloc[-1] else 'hold'
                continue

            lead_data = _get_series(pair['lead_stock'])
            if lead_data is None:
                pair['action'] = 'sell'
                continue

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
        tickers = self._db.get_tickers()
        if not tickers:
            # First run or after a nightly refresh — populate tickers from Alpaca
            print("Tickers table empty, fetching tradeable assets from Alpaca...")
            tickers = self._alpaca.get_tradeable_assets()
            self._db.upsert_tickers(tickers, 'ALPACA')

        # Shuffle before applying ticker_limit so the limit picks a different
        # random subset of the universe each day rather than always the same
        # alphabetical head.
        random.shuffle(tickers)

        if self.ticker_limit:
            tickers = tickers[:self.ticker_limit]

        new_candidates = 0
        position_symbols = {p.symbol for p in self.get_positions()}
        gate_counts = {'penny': 0, 'correlation': 0, 'cointegration': 0, 'simulation': 0, 'action': 0}
        pairs_scanned = 0
        candidates_found = 0
        candidates_buy_ready = 0

        for stock1, stock2 in itertools.combinations(tickers, 2):
            if new_candidates >= self.min_daily_pairs:
                break

            if stock2 in self.pairs or stock2 in position_symbols:
                continue

            # Fetch each stock lazily; skip immediately if no data or penny stock.
            s1 = _get_series(stock1)
            if s1 is None or s1.iloc[-1] < 5:
                gate_counts['penny'] += 1
                continue

            s2 = _get_series(stock2)
            if s2 is None or s2.iloc[-1] < 5:
                gate_counts['penny'] += 1
                continue

            pairs_scanned += 1

            # Check correlation across all candidate lags, take the best.
            # Screening at lag=1 only would silently reject pairs that are
            # correlated at a different lag, discarding candidates before the
            # optimizer can find the best offset.
            corr_by_lag = {
                lag: stock_evaluator.get_correlation(s1, s2, lag)
                for lag in range(1, self.max_lag + 1)
            }
            best_corr = max(
                (c for c in corr_by_lag.values() if not math.isnan(c)),
                default=float('nan'),
            )
            if math.isnan(best_corr) or best_corr < self.min_correlation:
                gate_counts['correlation'] += 1
                continue

            if not stock_evaluator.is_cointegrated(s1, s2):
                gate_counts['cointegration'] += 1
                continue

            # Optimize lag and MA parameters via mini-backtest simulation.
            # Rejects pairs where the strategy wouldn't have been historically
            # profitable, or where the signal never fired more than once.
            sim_result = simulator.optimize(s1, s2, max_lag=self.max_lag)
            if sim_result.total_return <= 0 or sim_result.num_trades < 2:
                gate_counts['simulation'] += 1
                continue

            candidates_found += 1

            action = stock_evaluator.get_action(
                s1, s2,
                lag=sim_result.lag,
                short_ma=sim_result.short_ma,
                long_ma=sim_result.long_ma,
            )
            if action != 'buy':
                gate_counts['action'] += 1
                continue

            candidates_buy_ready += 1

            # Use correlation at the optimized lag for the stored pair record,
            # falling back to the best screened lag if the optimized lag is missing.
            corr_at_opt_lag = corr_by_lag.get(sim_result.lag, best_corr)
            if math.isnan(corr_at_opt_lag):
                corr_at_opt_lag = best_corr

            print(
                f"Adding new pair: {stock1} -> {stock2} | corr={corr_at_opt_lag:.4f} "
                f"lag={sim_result.lag} ma=({sim_result.short_ma},{sim_result.long_ma}) "
                f"sim_return={sim_result.total_return:.2%}"
            )

            new_pair = {
                'lead_stock':       stock1,
                'lag_stock':        stock2,
                'lag':              sim_result.lag,
                'short_ma':         sim_result.short_ma,
                'long_ma':          sim_result.long_ma,
                'corr':             corr_at_opt_lag,
                'action':           action,
                'simulated_return': sim_result.total_return,
            }
            new_pair['pair_id'] = self._db.save_pair(new_pair, self._run_id)
            self.pairs[stock2] = new_pair
            new_candidates += 1

        if new_candidates < self.min_daily_pairs:
            print(
                f"Warning: only {new_candidates} new pairs found "
                f"(target: {self.min_daily_pairs}). "
                f"Scanned {pairs_scanned} pairs. Gates: {gate_counts}"
            )

        self._pairs_scanned = pairs_scanned
        self._candidates_found = candidates_found
        self._candidates_buy_ready = candidates_buy_ready

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
                self._db.deactivate_pair(symbol, self._run_id)
                to_remove.append(symbol)

        for symbol in to_remove:
            self.pairs.pop(symbol)

        # --- Execute buys ---
        existing_position_symbols = {p.symbol for p in self.get_positions()}
        buy_pairs = [
            pair for symbol, pair in self.pairs.items()
            if pair['action'] == 'buy' and symbol not in existing_position_symbols
        ]

        new_buy_symbols: set[str] = set()
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
                        initial_cost = float(quantity * price)
                        pair['initial_cost'] = initial_cost
                        if pair.get('pair_id'):
                            self._db.update_pair_initial_cost(pair['pair_id'], initial_cost)
                        new_buy_symbols.add(pair['lag_stock'])
                        self._db.log_trade(
                            run_id=self._run_id,
                            symbol=pair['lag_stock'],
                            side='buy',
                            quantity=float(quantity),
                            price=float(price),
                            filled_at=now,
                            pair_id=pair.get('pair_id'),
                        )

        # --- Top up existing positions with a buy signal ---
        daily_topups = 0
        available_cash = self.get_cash()
        for symbol, pair in self.pairs.items():
            if pair['action'] != 'buy':
                continue
            if symbol in new_buy_symbols:
                continue  # just bought today, skip same-day top-up
            if available_cash <= 0:
                break

            position = self.get_position(symbol)
            if not position or position.quantity <= 0:
                continue

            initial_cost = pair.get('initial_cost')
            if not initial_cost:
                continue

            price = self.get_last_price(symbol)
            if not price or price <= 0:
                continue

            max_value = initial_cost * self.max_position_multiplier
            current_value = float(position.quantity) * price
            gap = max_value - current_value
            if gap <= 0:
                continue

            top_up_dollars = min(gap * self.top_up_rate, available_cash)
            quantity = int(top_up_dollars / price)
            if quantity > 0:
                order = self.create_order(symbol, quantity, 'buy')
                self.submit_order(order)
                available_cash -= quantity * price
                daily_topups += 1
                self._db.log_trade(
                    run_id=self._run_id,
                    symbol=symbol,
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

        self.add_line("active_pairs",         float(len(active_pairs)))
        self.add_line("avg_corr",             round(avg_corr, 4))
        self.add_line("cash_ratio",           round(self.cash / portfolio_value, 4))
        self.add_line("daily_buys",           float(len(buy_pairs)))
        self.add_line("daily_sells",          float(len(to_remove)))
        self.add_line("daily_topups",         float(daily_topups))
        self.add_line("pairs_scanned",        float(self._pairs_scanned))
        self.add_line("candidates_found",     float(self._candidates_found))
        self.add_line("candidates_buy_ready", float(self._candidates_buy_ready))

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
            daily_topups=daily_topups,
            pairs_scanned=self._pairs_scanned,
            candidates_found=self._candidates_found,
            candidates_buy_ready=self._candidates_buy_ready,
        )

    def on_strategy_end(self):
        self._db.close_run(self._run_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_penny_stock(series) -> bool:
        """Return True if the most recent close price in *series* is below $5."""
        return not series.empty and float(series.iloc[-1]) < 5
