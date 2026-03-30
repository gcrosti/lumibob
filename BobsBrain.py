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
from TickerClusterer import TickerClusterer

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
        # cluster_recompute_days: how often to rebuild movement clusters.
        # None = compute once on the first trading day and hold for the full run.
        # This is the recommended default for backtests; use an integer (e.g. 30)
        # for live trading so clusters adapt as market regimes shift.
        self.cluster_recompute_days = self.parameters.get('cluster_recompute_days', None)
        # use_clusters: set to False to bypass TickerClusterer and fall back to the
        # original shuffle+combinations path. Useful for A/B comparison backtests.
        # When True, ticker_limit is ignored — clusters are built on the full universe.
        self.use_clusters = self.parameters.get('use_clusters', True)
        # Position sizing parameters.
        # min_position_pct / max_position_pct define the range of portfolio-value
        # fraction allocated to a single new position; actual size scales with the
        # pair's confidence score (Z-score depth, correlation, simulated Sharpe).
        # target_deployed_pct is the fraction of portfolio value the strategy aims
        # to have deployed at any time; a deployment gap boosts individual allocations
        # toward this target when the portfolio is under-invested.
        self.min_position_pct = self.parameters.get('min_position_pct', 0.03)
        self.max_position_pct = self.parameters.get('max_position_pct', 0.20)
        self.target_deployed_pct = self.parameters.get('target_deployed_pct', 0.60)
        self.entry_threshold = self.parameters.get('entry_threshold', 2.0)
        self.exit_threshold = self.parameters.get('exit_threshold', 0.5)
        self.min_sharpe = self.parameters.get('min_sharpe', 0.5)
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
        self._db.migrate_pairs_sim_sharpe()
        self._alpaca = AlpacaClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=os.getenv('ALPACA_IS_PAPER', 'true').lower() == 'true',
            mode=self._run_mode,
        )
        self._cache = StockDataCache(self._db, self._alpaca)
        self._clusterer = TickerClusterer(db=self._db)
        self._failed_tickers: set[str] = set(self._db.get_failed_tickers())

        self._run_id = secrets.token_hex(3)
        self.pairs = self._db.load_active_pairs(self._run_id)
        self._db.create_run(
            run_id=self._run_id,
            mode=self._run_mode,
            settings={
                'ticker_limit': self.ticker_limit,
                'cluster_recompute_days': self.cluster_recompute_days,
                'use_clusters': self.use_clusters,
                'lookback_window': self.lookback_window,
                'min_correlation': self.min_correlation,
                'min_daily_pairs': self.min_daily_pairs,
                'min_position_pct': self.min_position_pct,
                'max_position_pct': self.max_position_pct,
                'target_deployed_pct': self.target_deployed_pct,
                'entry_threshold': self.entry_threshold,
                'exit_threshold': self.exit_threshold,
                'min_sharpe': self.min_sharpe,
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

            # Fetch price history for both legs. If either is unavailable (data
            # gap, delisting, API error) we cannot compute a Z-score, so force a
            # sell rather than hold a position we can no longer evaluate.
            lag_data = _get_series(pair['lag_stock'])
            if lag_data is None:
                pair['action'] = 'sell'
                continue

            lead_data = _get_series(pair['lead_stock'])
            if lead_data is None:
                pair['action'] = 'sell'
                continue

            action, current_z = stock_evaluator.get_zscore_action(
                lead_data, lag_data,
                window=pair['zscore_window'],
                entry_threshold=pair['entry_threshold'],
                exit_threshold=pair['exit_threshold'],
            )
            pair['current_zscore'] = current_z
            pair['action'] = action

        # --- Evaluate watchlist candidates ---
        # Re-check the action signal for pairs that previously passed all discovery
        # gates but didn't have a buy signal. This is cheap (no cointegration or
        # simulation re-runs) and converts idle-cash days into buy opportunities.
        today = end_date.date()
        position_symbols = {p.symbol for p in self.get_positions()}
        stale_watchlist = []
        for symbol, candidate in self._watchlist.items():
            ttl = candidate.get('watchlist_ttl', self._watchlist_ttl_days)
            if (today - candidate['watchlist_date']).days > ttl:
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

            action, current_z = stock_evaluator.get_zscore_action(
                lead_data, lag_data,
                window=candidate['zscore_window'],
                entry_threshold=candidate['entry_threshold'],
                exit_threshold=candidate['exit_threshold'],
            )
            if action == 'buy':
                candidate['action'] = 'buy'
                candidate['current_zscore'] = current_z
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
            pair['action'], pair['current_zscore'] = stock_evaluator.get_zscore_action(
                lead_data, lag_data,
                window=pair['zscore_window'],
                entry_threshold=pair['entry_threshold'],
                exit_threshold=pair['exit_threshold'],
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

        if self.use_clusters:
            # Cluster the full ticker universe by 6-month return similarity.
            # Clustering always runs on all tickers so the structure reflects the
            # full universe — ticker_limit is never applied before this step.
            # Clusters are ranked by expected yield (avg intra-cluster correlation ×
            # size) so the most fertile clusters are searched first.
            clusters = self._clusterer.get_clusters(
                tickers, as_of=end_date, recompute_days=self.cluster_recompute_days
            )

            if self.ticker_limit:
                # Fill a ticker_limit-sized bucket from the top clusters in yield
                # order. Each cluster is shuffled before sampling so the specific
                # tickers drawn rotate randomly each day while always coming from
                # the highest-yield clusters. Dip into subsequent clusters only
                # when the top cluster has fewer tickers than the remaining quota.
                bucket: list[str] = []
                remaining = self.ticker_limit
                for cluster in clusters:
                    if remaining <= 0:
                        break
                    sample = list(cluster)
                    random.shuffle(sample)
                    bucket.extend(sample[:remaining])
                    remaining -= len(sample[:remaining])
                pair_iter = itertools.combinations(bucket, 2)
            else:
                # Full universe: search within each cluster only, in yield order.
                # Shuffle within each cluster copy for daily pair-order variety.
                shuffled_clusters = [list(c) for c in clusters]
                for c in shuffled_clusters:
                    random.shuffle(c)
                pair_iter = (
                    (s1, s2)
                    for cluster in shuffled_clusters
                    for s1, s2 in itertools.combinations(cluster, 2)
                )
        else:
            # Original path: shuffle all tickers, apply optional ticker_limit, then
            # iterate all combinations. ticker_limit is only meaningful here.
            random.shuffle(tickers)
            if self.ticker_limit:
                tickers = tickers[:self.ticker_limit]
            pair_iter = itertools.combinations(tickers, 2)

        new_candidates = 0
        gate_counts = {'penny': 0, 'correlation': 0, 'cointegration': 0, 'simulation': 0, 'sharpe': 0, 'holdout': 0, 'action': 0}
        pairs_scanned = 0
        candidates_found = 0
        candidates_buy_ready = 0
        new_penny_stocks: set[str] = set()

        for stock1, stock2 in pair_iter:
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

            # Walk-forward holdout: optimise Z-score params on the train split
            # (first ~67% of lookback) and validate on the holdout split (last
            # ~33%). Rejects pairs that overfit to recent history by requiring
            # profitable performance on data the optimiser never saw.
            sim_result, holdout_return, holdout_days_to_first_signal = simulator.optimize_zscore_with_holdout(s1, s2)
            if sim_result.total_return <= 0 or sim_result.num_trades < 2:
                gate_counts['simulation'] += 1
                continue

            if sim_result.sharpe < self.min_sharpe:
                gate_counts['sharpe'] += 1
                continue

            if holdout_return <= 0:
                gate_counts['holdout'] += 1
                continue

            candidates_found += 1

            # Use the best-corr lag for reporting; z-score doesn't use lag directly.
            corr_at_opt_lag = best_corr

            action, current_z = stock_evaluator.get_zscore_action(
                s1, s2,
                window=sim_result.zscore_window,
                entry_threshold=sim_result.entry_threshold,
                exit_threshold=sim_result.exit_threshold,
            )
            if action != 'buy':
                gate_counts['action'] += 1
                # Park in watchlist rather than discard — re-check action signal
                # cheaply each day until buy-ready or TTL expires.
                # TTL is derived from the holdout simulation: how many days did
                # the signal take to fire on unseen data? Add 1 day of buffer.
                # Fall back to the global default when the signal never fired.
                if stock2 not in self._watchlist and stock2 not in self.pairs:
                    ttl = (holdout_days_to_first_signal + 1
                           if holdout_days_to_first_signal > 0
                           else self._watchlist_ttl_days)
                    self._watchlist[stock2] = {
                        'lead_stock':       stock1,
                        'lag_stock':        stock2,
                        'corr':             corr_at_opt_lag,
                        'simulated_return': sim_result.total_return,
                        'sim_sharpe':       sim_result.sharpe,
                        'watchlist_date':   today,
                        'watchlist_ttl':    ttl,
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
                'corr':             corr_at_opt_lag,
                'action':           action,
                'simulated_return': sim_result.total_return,
                'sim_sharpe':       sim_result.sharpe,
                'current_zscore':   current_z,
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

        # Confidence-weighted allocation: each new position receives a budget
        # scaled by a composite signal score (Z-score depth, correlation, Sharpe).
        # A deployment-gap boost adds extra capital when the portfolio is below
        # target_deployed_pct, divided evenly across remaining buy candidates.
        # Pairs are funded in descending confidence order; continue (not break) is
        # used when a pair's budget exceeds available cash so cheaper candidates
        # further down the ranked list can still be executed.
        new_buy_symbols: set[str] = set()
        no_price_symbols: list[str] = []
        daily_new_buys = 0
        if buy_pairs:
            portfolio_value = self.portfolio_value
            available_cash = self.get_cash()
            current_deployed = portfolio_value - available_cash
            deployment_gap = max(0.0, self.target_deployed_pct * portfolio_value - current_deployed)
            n_candidates = len(buy_pairs)

            for pair in buy_pairs:
                pair['confidence_score'] = self._compute_confidence(pair)
            buy_pairs_ranked = sorted(buy_pairs, key=lambda p: p['confidence_score'], reverse=True)

            for pair in buy_pairs_ranked:
                confidence = pair['confidence_score']
                base_budget = (
                    self.min_position_pct + confidence * (self.max_position_pct - self.min_position_pct)
                ) * portfolio_value
                if deployment_gap > 0 and n_candidates > 0:
                    gap_share = deployment_gap / n_candidates
                    base_budget = min(base_budget + gap_share, self.max_position_pct * portfolio_value)
                per_stock_budget = base_budget

                if available_cash < per_stock_budget:
                    continue  # budget exceeds cash; try the next (cheaper) candidate

                price = self.get_last_price(pair['lag_stock'])
                if price and price > 0:
                    quantity = round(per_stock_budget / price, 6)
                    if quantity > 0:
                        order = self.create_order(pair['lag_stock'], quantity, 'buy')
                        self.submit_order(order)
                        available_cash -= per_stock_budget
                        deployment_gap = max(0.0, deployment_gap - per_stock_budget)
                        n_candidates -= 1
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
                    n_candidates -= 1

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

        watchlist_ttls = [
            c.get('watchlist_ttl', self._watchlist_ttl_days)
            for c in self._watchlist.values()
        ]
        avg_watchlist_ttl = (
            round(sum(watchlist_ttls) / len(watchlist_ttls), 1)
            if watchlist_ttls else None
        )

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
        if avg_watchlist_ttl is not None:
            self.add_line("avg_watchlist_ttl", avg_watchlist_ttl)

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
            avg_watchlist_ttl=avg_watchlist_ttl,
        )

    def on_strategy_end(self):
        self._db.close_run(self._run_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_confidence(self, pair: dict) -> float:
        """
        Returns a [0.0, 1.0] confidence score for a buy signal on *pair*.

        Three components are combined with fixed weights:
          - Z-score depth  (0.4): how far the spread has exceeded the entry
            threshold; normalised by the threshold itself so a z of -4.0 with
            entry=2.0 scores 1.0, while exactly at threshold scores 0.0.
          - Correlation    (0.4): pair's correlation normalised over the range
            [min_correlation, 1.0]; pairs just above the minimum gate score 0.0.
          - Simulated Sharpe (0.2): sim Sharpe normalised over [min_sharpe,
            2×min_sharpe]; caps at 1.0 when Sharpe reaches twice the minimum.

        Missing values fall back to their respective minimum gates so a pair
        without a stored z-score or Sharpe receives the lowest possible score
        for that component rather than an error.
        """
        z = pair.get('current_zscore')
        entry = pair.get('entry_threshold', self.entry_threshold)
        z_score = min((abs(z) - entry) / entry, 1.0) if z is not None else 0.0
        z_score = max(z_score, 0.0)

        corr = pair.get('corr', self.min_correlation)
        corr_score = min((corr - self.min_correlation) / (1.0 - self.min_correlation), 1.0)
        corr_score = max(corr_score, 0.0)

        sharpe = pair.get('sim_sharpe', self.min_sharpe)
        sharpe_score = min((sharpe - self.min_sharpe) / self.min_sharpe, 1.0)
        sharpe_score = max(sharpe_score, 0.0)

        return 0.4 * z_score + 0.4 * corr_score + 0.2 * sharpe_score

    @staticmethod
    def _is_penny_stock(series) -> bool:
        """Return True if the most recent close price in *series* is below $5."""
        return not series.empty and float(series.iloc[-1]) < 5
