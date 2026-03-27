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
        self.ticker_limit = self.parameters.get('ticker_limit', None)
        # max_daily_spend_pct: fraction of portfolio value that can be deployed per day.
        # per_pair_allocation: fraction of max_daily_spend allocated to each new pair.
        # With defaults (0.5 × 0.10), each new pair receives ~5% of portfolio value.
        self.max_daily_spend_pct = self.parameters.get('max_daily_spend_pct', 0.5)
        self.per_pair_allocation = self.parameters.get('per_pair_allocation', 0.10)
        self.entry_threshold = self.parameters.get('entry_threshold', 2.0)
        self.exit_threshold = self.parameters.get('exit_threshold', 0.5)
        self._run_mode = os.getenv('RUN_MODE', 'backtest')

        self._spy_start_price = None
        self._starting_portfolio_value = None
        self._pairs_scanned = 0
        self._candidates_found = 0
        self._candidates_buy_ready = 0

        # Watchlist: pairs that passed all discovery gates but lacked a buy signal.
        # Keyed by lag_symbol. Re-evaluated cheaply each day (action check only —
        # no cointegration or simulation re-runs). Entries expire after _watchlist_ttl_days.
        self._watchlist: dict[str, dict] = {}
        self._watchlist_ttl_days: int = 5

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
        self._db.migrate_zscore_columns()
        self._alpaca = AlpacaClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=os.getenv('ALPACA_IS_PAPER', 'true').lower() == 'true',
            mode=self._run_mode,
        )
        self._cache = StockDataCache(self._db, self._alpaca)
        self._failed_tickers: set[str] = set(self._db.get_failed_tickers())

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
                'max_daily_spend_pct': self.max_daily_spend_pct,
                'per_pair_allocation': self.per_pair_allocation,
                'entry_threshold': self.entry_threshold,
                'exit_threshold': self.exit_threshold,
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

            # Always fetch the lag stock — needed for the signal regardless.
            lag_data = _get_series(pair['lag_stock'])
            if lag_data is None:
                pair['action'] = 'sell'
                continue

            lead_data = _get_series(pair['lead_stock'])
            if lead_data is None:
                pair['action'] = 'sell'
                continue

            signal_type = pair.get('signal_type', 'ma')

            if signal_type == 'zscore':
                action, current_z = stock_evaluator.get_zscore_action(
                    lead_data, lag_data,
                    window=pair['zscore_window'],
                    entry_threshold=pair['entry_threshold'],
                    exit_threshold=pair['exit_threshold'],
                )
                pair['current_zscore'] = current_z
                pair['action'] = action
            else:
                # Legacy MA crossover path — also update correlation for existing MA pairs.
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

        # --- Evaluate watchlist candidates ---
        # Re-check the action signal for pairs that previously passed all discovery
        # gates but didn't have a buy signal. This is cheap (no cointegration or
        # simulation re-runs) and converts idle-cash days into buy opportunities.
        today = end_date.date()
        position_symbols = {p.symbol for p in self.get_positions()}
        stale_watchlist = []
        for symbol, candidate in self._watchlist.items():
            if (today - candidate['watchlist_date']).days > self._watchlist_ttl_days:
                stale_watchlist.append(symbol)
                continue
            if symbol in self.pairs or symbol in position_symbols:
                stale_watchlist.append(symbol)
                continue

            lead_data = _get_series(candidate['lead_stock'])
            lag_data = _get_series(candidate['lag_stock'])
            if lead_data is None or lag_data is None:
                stale_watchlist.append(symbol)
                continue

            if candidate.get('signal_type', 'ma') == 'zscore':
                action, _ = stock_evaluator.get_zscore_action(
                    lead_data, lag_data,
                    window=candidate['zscore_window'],
                    entry_threshold=candidate['entry_threshold'],
                    exit_threshold=candidate['exit_threshold'],
                )
            else:
                action = stock_evaluator.get_action(
                    lead_data, lag_data,
                    lag=candidate['lag'],
                    short_ma=candidate['short_ma'],
                    long_ma=candidate['long_ma'],
                )
            if action == 'buy':
                candidate['action'] = 'buy'
                candidate['pair_id'] = self._db.save_pair(candidate, self._run_id)
                self.pairs[symbol] = candidate
                stale_watchlist.append(symbol)
                print(f"Watchlist promotion: {candidate['lead_stock']} -> {symbol}")

        for symbol in stale_watchlist:
            self._watchlist.pop(symbol, None)

        # --- Re-evaluate pending buy pairs (queued but not yet positions) ---
        # Pairs promoted from the watchlist or discovered on a prior day may not
        # have been executed yet (cash ran out). Their signal can expire while
        # they sit in the queue. Re-evaluate here so only pairs with a live buy
        # signal remain as buy candidates; others revert to hold/sell and are
        # cleaned up by on_trading_iteration's normal sell path.
        for symbol in list(self.pairs.keys()):
            if symbol in position_symbols:
                continue  # already handled in the existing positions loop above
            pair = self.pairs[symbol]
            lag_data = _get_series(pair['lag_stock'])
            if lag_data is None:
                pair['action'] = 'sell'
                continue
            lead_data = _get_series(pair['lead_stock'])
            if lead_data is None:
                pair['action'] = 'sell'
                continue
            if pair.get('signal_type', 'ma') == 'zscore':
                pair['action'], _ = stock_evaluator.get_zscore_action(
                    lead_data, lag_data,
                    window=pair['zscore_window'],
                    entry_threshold=pair['entry_threshold'],
                    exit_threshold=pair['exit_threshold'],
                )
            else:
                pair['action'] = stock_evaluator.get_action(
                    lead_data, lag_data,
                    lag=pair['lag'],
                    short_ma=pair['short_ma'],
                    long_ma=pair['long_ma'],
                )

        # --- Discover new pairs ---
        tickers = self._db.get_tickers()
        if not tickers:
            # First run or after a nightly refresh — populate tickers from Alpaca
            print("Tickers table empty, fetching tradeable assets from Alpaca...")
            tickers = self._alpaca.get_tradeable_assets()
            self._db.upsert_tickers(tickers, 'ALPACA')

        # Remove tickers that have previously failed price lookups or were identified
        # as penny stocks so they cannot form new pairs or consume the ticker_limit.
        tickers = [t for t in tickers if t not in self._failed_tickers]

        # Shuffle before applying ticker_limit so the limit picks a different
        # random subset of the universe each day rather than always the same
        # alphabetical head.
        random.shuffle(tickers)

        if self.ticker_limit:
            tickers = tickers[:self.ticker_limit]

        new_candidates = 0
        gate_counts = {'penny': 0, 'correlation': 0, 'cointegration': 0, 'simulation': 0, 'action': 0}
        pairs_scanned = 0
        candidates_found = 0
        candidates_buy_ready = 0
        new_penny_stocks: set[str] = set()

        for stock1, stock2 in itertools.combinations(tickers, 2):
            if new_candidates >= self.min_daily_pairs:
                break

            if stock2 in self.pairs or stock2 in position_symbols:
                continue

            # Fetch each stock lazily; skip immediately if no data or penny stock.
            # Track penny stocks so they are excluded from future runs.
            s1 = _get_series(stock1)
            if s1 is None or s1.iloc[-1] < 5:
                gate_counts['penny'] += 1
                if s1 is not None:
                    new_penny_stocks.add(stock1)
                continue

            s2 = _get_series(stock2)
            if s2 is None or s2.iloc[-1] < 5:
                gate_counts['penny'] += 1
                if s2 is not None:
                    new_penny_stocks.add(stock2)
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

            # Optimise Z-score parameters via mini-backtest simulation.
            # Rejects pairs where the strategy wouldn't have been historically
            # profitable, or where the signal never fired more than once.
            sim_result = simulator.optimize_zscore(s1, s2)
            if sim_result.total_return <= 0 or sim_result.num_trades < 2:
                gate_counts['simulation'] += 1
                continue

            candidates_found += 1

            # Use the best-corr lag for reporting; z-score doesn't use lag directly.
            corr_at_opt_lag = best_corr

            action, _ = stock_evaluator.get_zscore_action(
                s1, s2,
                window=sim_result.zscore_window,
                entry_threshold=sim_result.entry_threshold,
                exit_threshold=sim_result.exit_threshold,
            )
            if action != 'buy':
                gate_counts['action'] += 1
                # Park in watchlist rather than discard — re-check action signal
                # cheaply each day until buy-ready or TTL expires.
                if stock2 not in self._watchlist and stock2 not in self.pairs:
                    self._watchlist[stock2] = {
                        'lead_stock':       stock1,
                        'lag_stock':        stock2,
                        'lag':              1,
                        'short_ma':         2,
                        'long_ma':          5,
                        'corr':             corr_at_opt_lag,
                        'simulated_return': sim_result.total_return,
                        'watchlist_date':   today,
                        'action':           action,
                        'signal_type':      'zscore',
                        'zscore_window':    sim_result.zscore_window,
                        'entry_threshold':  sim_result.entry_threshold,
                        'exit_threshold':   sim_result.exit_threshold,
                    }
                continue

            candidates_buy_ready += 1

            print(
                f"Adding new pair: {stock1} -> {stock2} | corr={corr_at_opt_lag:.4f} "
                f"z_window={sim_result.zscore_window} "
                f"entry={sim_result.entry_threshold} exit={sim_result.exit_threshold} "
                f"sim_return={sim_result.total_return:.2%}"
            )

            new_pair = {
                'lead_stock':       stock1,
                'lag_stock':        stock2,
                'lag':              1,
                'short_ma':         2,
                'long_ma':          5,
                'corr':             corr_at_opt_lag,
                'action':           action,
                'simulated_return': sim_result.total_return,
                'signal_type':      'zscore',
                'zscore_window':    sim_result.zscore_window,
                'entry_threshold':  sim_result.entry_threshold,
                'exit_threshold':   sim_result.exit_threshold,
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

        # Persist newly discovered penny stocks so future runs exclude them before
        # applying ticker_limit, making the daily sample cleaner over time.
        for sym in new_penny_stocks:
            self._failed_tickers.add(sym)
            self._db.mark_ticker_failed(sym, 'penny stock')

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

        # Fixed per-pair allocation: each new position receives a uniform slice of
        # portfolio value regardless of how many pairs fire on a given day.
        # This replaces the previous "half of cash / N pairs" approach which caused
        # variable and sometimes outsized single-pair allocations on low-activity days.
        new_buy_symbols: set[str] = set()
        no_price_symbols: list[str] = []
        daily_new_buys = 0
        if buy_pairs:
            available_cash = self.get_cash()
            per_stock_budget = available_cash * self.max_daily_spend_pct * self.per_pair_allocation
            remaining_cash = available_cash
            for pair in buy_pairs:
                if remaining_cash < per_stock_budget:
                    break
                price = self.get_last_price(pair['lag_stock'])
                if price and price > 0:
                    quantity = round(per_stock_budget / price, 6)
                    if quantity > 0:
                        order = self.create_order(pair['lag_stock'], quantity, 'buy')
                        self.submit_order(order)
                        remaining_cash -= per_stock_budget
                        new_buy_symbols.add(pair['lag_stock'])
                        daily_new_buys += 1
                        self._db.log_trade(
                            run_id=self._run_id,
                            symbol=pair['lag_stock'],
                            side='buy',
                            quantity=float(quantity),
                            price=float(price),
                            filled_at=now,
                            pair_id=pair.get('pair_id'),
                        )
                else:
                    # Price permanently unavailable — record and evict from queue
                    # so this pair stops consuming cash budget each day.
                    symbol = pair['lag_stock']
                    self._failed_tickers.add(symbol)
                    self._db.mark_ticker_failed(symbol, 'get_last_price returned None')
                    no_price_symbols.append(symbol)

        for symbol in no_price_symbols:
            self.pairs.pop(symbol, None)
            self._db.deactivate_pair(symbol, self._run_id)

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

        zscore_pairs = [
            p['current_zscore'] for p in active_pairs
            if p.get('signal_type') == 'zscore' and p.get('current_zscore') is not None
        ]
        avg_zscore = round(sum(zscore_pairs) / len(zscore_pairs), 4) if zscore_pairs else None

        self.add_line("active_pairs",         float(len(active_pairs)))
        self.add_line("avg_corr",             round(avg_corr, 4))
        self.add_line("cash_ratio",           round(self.cash / portfolio_value, 4))
        self.add_line("daily_buys",           float(daily_new_buys))
        self.add_line("daily_sells",          float(len(to_remove)))
        self.add_line("pairs_scanned",        float(self._pairs_scanned))
        self.add_line("candidates_found",     float(self._candidates_found))
        self.add_line("candidates_buy_ready", float(self._candidates_buy_ready))
        self.add_line("watchlist_size",       float(len(self._watchlist)))
        if avg_zscore is not None:
            self.add_line("avg_zscore", avg_zscore)

        self._db.log_snapshot(
            run_id=self._run_id,
            time=now,
            portfolio_value=float(portfolio_value),
            cash=float(self.cash),
            spy_value=spy_value,
            active_pairs=len(active_pairs),
            avg_correlation=round(avg_corr, 4),
            cash_ratio=round(self.cash / portfolio_value, 4),
            daily_buys=daily_new_buys,
            daily_sells=len(to_remove),
            daily_topups=0,
            pairs_scanned=self._pairs_scanned,
            candidates_found=self._candidates_found,
            candidates_buy_ready=self._candidates_buy_ready,
            avg_zscore=avg_zscore,
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
