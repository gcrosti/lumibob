import logging
import os
import secrets
from datetime import datetime, timedelta
from statistics import median

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from lumibot.strategies import Strategy

from AlpacaClient import AlpacaClient
from DatabaseClient import DatabaseClient
from StockDataCache import StockDataCache
from StockEvaluator import StockEvaluator
from TickerClusterer import TickerClusterer

load_dotenv()

SIC_SECTORS = {
    range(100, 1000):   'Agriculture, Forestry & Fishing',
    range(1000, 1500):  'Mining',
    range(1500, 1800):  'Construction',
    range(2000, 4000):  'Manufacturing',
    range(4000, 5000):  'Transportation & Utilities',
    range(5000, 5200):  'Wholesale Trade',
    range(5200, 6000):  'Retail Trade',
    range(6000, 6800):  'Finance, Insurance & Real Estate',
    range(7000, 9000):  'Services',
    range(9100, 9730):  'Public Administration',
}


def _sic_to_sector(sic_code) -> str | None:
    if sic_code is None:
        return None
    try:
        sic = int(sic_code)
    except (ValueError, TypeError):
        return None
    for sic_range, sector in SIC_SECTORS.items():
        if sic in sic_range:
            return sector
    return 'Other'


class BobsBrain(Strategy):
    """
    Score-and-rank pairs trading strategy.

    Lifecycle per trading day:
    - before_market_opens(): score all candidate pairs and existing positions,
      build a unified ranked list, and determine the target portfolio (top K).
    - on_trading_iteration(): execute sells (positions displaced from the target
      portfolio) then buys (new pairs entering the target portfolio).

    Requires DB_URL, ALPACA_API_KEY, and ALPACA_API_SECRET to be set (via .env
    or environment). Raises EnvironmentError on startup if any are missing.
    """

    def initialize(self):
        self.sleeptime = '1D'
        # Calendar days of price history for scoring (must span corr windows in bars).
        self.lookback_window = self.parameters.get('lookback_window', 130)
        # Min days between cluster recomputes; None = recompute only when cache cold.
        self.cluster_recompute_days = self.parameters.get('cluster_recompute_days', None)

        self._ticker_metadata: dict[str, dict] = {}
        self._metadata_loaded = False

        # Position size bounds as a fraction of portfolio (from composite score).
        self.min_position_pct = self.parameters.get('min_position_pct', 0.03)
        self.max_position_pct = self.parameters.get('max_position_pct', 0.20)
        # Target fraction deployed; shortfall increases per-buy allocation.
        self.target_deployed_pct = self.parameters.get('target_deployed_pct', 0.60)

        # Spread z-score bands for entry signal vs exit / shallow regime.
        self.entry_threshold = self.parameters.get('entry_threshold', 2.0)
        self.exit_threshold = self.parameters.get('exit_threshold', 0.5)
        # Rolling bars for spread z-score.
        self.zscore_window = self.parameters.get('zscore_window', 20)

        # Log-return correlation windows (bars); long vs short horizon.
        self.corr_long_window = self.parameters.get('corr_long_window', 90)
        self.corr_short_window = self.parameters.get('corr_short_window', 20)
        # Composite score weights (corr_long, corr_short, z_depth); should sum to 1.
        self.w_corr_long = self.parameters.get('w_corr_long', 0.3)
        self.w_corr_short = self.parameters.get('w_corr_short', 0.5)
        self.w_z_depth = self.parameters.get('w_z_depth', 0.2)

        # Max new pairs scored per day (global budget).
        self.max_daily_candidates = self.parameters.get('max_daily_candidates', 200)
        # Days before the same unordered pair can be scored again.
        self.cooldown_days = self.parameters.get('cooldown_days', 7)

        # Minimum price for a ticker to pass the penny-stock filter.
        self.penny_threshold = self.parameters.get('penny_threshold', 5.0)

        # Hard ceiling on target portfolio size (Tier 2 tunable).
        # K floats between max_k * quality_scale_min and max_k based on daily
        # pool quality.  The buy loop's cash check enforces affordability; K
        # itself is purely a quality / concentration target.
        self.max_k = self.parameters.get('max_k', 20)

        # Quality-scale curve for dynamic-K: pool_corr is divided by the pivot
        # and the result is clamped to [min, max], then multiplied by max_k.
        self.quality_scale_pivot = self.parameters.get('quality_scale_pivot', 0.7)
        self.quality_scale_min = self.parameters.get('quality_scale_min', 0.5)
        self.quality_scale_max = self.parameters.get('quality_scale_max', 1.5)

        # TickerClusterer parameters (passed through at construction time).
        self.cluster_lookback_days = self.parameters.get('cluster_lookback_days', 126)
        self.hdbscan_min_cluster_size = self.parameters.get('hdbscan_min_cluster_size', 5)
        self.hdbscan_min_samples = self.parameters.get('hdbscan_min_samples', 2)
        self.pca_variance = self.parameters.get('pca_variance', 0.95)
        self.min_coverage = self.parameters.get('min_coverage', 0.5)
        self.hdbscan_metric = self.parameters.get('hdbscan_metric', 'precomputed')
        self.hdbscan_selection_method = self.parameters.get('hdbscan_selection_method', 'eom')
        self.hdbscan_cluster_selection_epsilon = self.parameters.get(
            'hdbscan_cluster_selection_epsilon', 0.0,
        )
        self.min_intra_cluster_corr = self.parameters.get('min_intra_cluster_corr', 0.3)

        self._run_mode = os.getenv('RUN_MODE', 'backtest')
        self._spy_start_price = None
        self._starting_portfolio_value = None
        self._pairs_scanned = 0
        self._candidates_found = 0
        self._candidates_buy_ready = 0
        self._pair_evaluated_at: dict[frozenset, datetime] = {}
        self._next_cluster_idx: int = 0

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
        self._db.migrate_ticker_metadata()
        self._db.migrate_failed_tickers()
        self._alpaca = AlpacaClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=os.getenv('ALPACA_IS_PAPER', 'true').lower() == 'true',
            mode=self._run_mode,
        )
        self._cache = StockDataCache(self._db, self._alpaca)
        self._clusterer = TickerClusterer(
            db=self._db,
            get_prices=self._cache.get_prices,
            lookback_days=self.cluster_lookback_days,
            min_cluster_size=self.hdbscan_min_cluster_size,
            pca_variance=self.pca_variance,
            min_coverage=self.min_coverage,
            hdbscan_min_samples=self.hdbscan_min_samples,
            hdbscan_metric=self.hdbscan_metric,
            hdbscan_selection_method=self.hdbscan_selection_method,
            hdbscan_cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
            min_intra_cluster_corr=self.min_intra_cluster_corr,
        )
        self._failed_tickers: set[str] = set(self._db.get_failed_tickers())

        self._run_id = secrets.token_hex(3)
        self.pairs: dict[str, dict] = self._db.load_active_pairs(self._run_id)

        tickers = self._db.get_tickers()
        if not tickers:
            print("[BobsBrain] Tickers table empty, fetching tradeable assets from Alpaca...")
            tickers = self._alpaca.get_tradeable_assets()
            self._db.upsert_tickers(tickers, 'ALPACA')
        tickers = [t for t in tickers if t not in self._failed_tickers]

        self._load_ticker_metadata(tickers)
        self._metadata_loaded = True

        # Cluster warm-up deferred to the first before_market_opens() call,
        # which has access to the simulated backtest date via get_datetime().
        # Calling here with datetime.utcnow() would use real-world prices
        # instead of backtest-window prices, producing garbage clusters.

        self._db.create_run(
            run_id=self._run_id,
            mode=self._run_mode,
            settings={
                # Data windows
                'lookback_window': self.lookback_window,
                'cluster_recompute_days': self.cluster_recompute_days,
                # Position sizing
                'max_k': self.max_k,
                'min_position_pct': self.min_position_pct,
                'max_position_pct': self.max_position_pct,
                'target_deployed_pct': self.target_deployed_pct,
                # Signal
                'entry_threshold': self.entry_threshold,
                'exit_threshold': self.exit_threshold,
                'zscore_window': self.zscore_window,
                # Scoring
                'corr_long_window': self.corr_long_window,
                'corr_short_window': self.corr_short_window,
                'w_corr_long': self.w_corr_long,
                'w_corr_short': self.w_corr_short,
                'w_z_depth': self.w_z_depth,
                # Discovery
                'max_daily_candidates': self.max_daily_candidates,
                'cooldown_days': self.cooldown_days,
                # Filters
                'penny_threshold': self.penny_threshold,
                # Dynamic-K quality scale
                'quality_scale_pivot': self.quality_scale_pivot,
                'quality_scale_min': self.quality_scale_min,
                'quality_scale_max': self.quality_scale_max,
                # Clustering / HDBSCAN
                'cluster_lookback_days': self.cluster_lookback_days,
                'hdbscan_min_cluster_size': self.hdbscan_min_cluster_size,
                'hdbscan_min_samples': self.hdbscan_min_samples,
                'pca_variance': self.pca_variance,
                'min_coverage': self.min_coverage,
                'hdbscan_metric': self.hdbscan_metric,
                'hdbscan_selection_method': self.hdbscan_selection_method,
                'hdbscan_cluster_selection_epsilon': self.hdbscan_cluster_selection_epsilon,
                'min_intra_cluster_corr': self.min_intra_cluster_corr,
            },
        )

    def before_market_opens(self):
        """
        Score-and-rank pipeline. Runs once per trading day.

        Phase 1: Hard gates (penny, sector) reduce the within-cluster universe.
        Phase 2: Score all eligible new candidates AND existing positions with a
                 composite score (corr_long, corr_short, z_depth).
        Phase 3: Build unified ranked list, determine target portfolio of top K,
                 and set actions (buy/sell) for on_trading_iteration().
        """
        evaluator = StockEvaluator()
        end_date = self.get_datetime()
        start_date = end_date - timedelta(days=self.lookback_window)

        _series_cache: dict[str, pd.Series | None] = {}

        def _get_series(symbol: str) -> pd.Series | None:
            if symbol not in _series_cache:
                df = self._cache.get_prices([symbol], start_date, end_date)
                if df.empty or symbol not in df.columns:
                    _series_cache[symbol] = None
                else:
                    series = df[symbol].dropna()
                    _series_cache[symbol] = series if not series.empty else None
            return _series_cache[symbol]

        position_symbols = {p.symbol for p in self.get_positions()}

        # --- Phase 2a: Re-score existing positions ---
        for symbol in list(self.pairs.keys()):
            pair = self.pairs[symbol]
            lag_data = _get_series(pair['lag_stock'])
            lead_data = _get_series(pair['lead_stock'])
            if lag_data is None or lead_data is None:
                pair['action'] = 'sell'
                pair['exit_reason'] = 'data_missing'
                pair['composite_score'] = -1.0
                continue

            corr_long, corr_short = evaluator.get_correlation_dual(
                lead_data, lag_data, self.corr_long_window, self.corr_short_window,
            )
            z_depth, z_raw = evaluator.compute_z_depth(
                lead_data, lag_data, self.zscore_window,
                self.entry_threshold, self.exit_threshold,
            )

            action, current_z = evaluator.get_zscore_action(
                lead_data, lag_data,
                window=self.zscore_window,
                entry_threshold=self.entry_threshold,
                exit_threshold=self.exit_threshold,
            )

            pair['corr_long'] = corr_long
            pair['corr_short'] = corr_short
            pair['z_depth'] = z_depth
            pair['current_zscore'] = current_z if current_z is not None else z_raw
            pair['composite_score'] = self._composite_score(corr_long, corr_short, z_depth)

            pid = pair.get('pair_id')
            if pid is not None and np.isfinite(corr_long):
                self._db.update_pair_correlation(int(pid), float(corr_long))

            if symbol in position_symbols and action == 'sell':
                pair['action'] = 'sell'
                pair['exit_reason'] = 'zscore_exit'
            else:
                pair['action'] = 'hold'

        # --- Phase 1 + Phase 2b: Discover and score new candidates ---
        tickers = [
            t for t in self._db.get_tickers() if t not in self._failed_tickers
        ]

        clusters = self._clusterer.get_clusters(
            tickers,
            as_of=end_date,
            recompute_days=self.cluster_recompute_days,
            ticker_metadata=self._ticker_metadata,
        )

        today = end_date.date() if hasattr(end_date, 'date') else end_date

        pairs_scanned = 0
        candidates_found = 0
        gate_counts = {'penny': 0, 'cooldown': 0}
        new_penny_stocks: set[str] = set()
        scored_candidates: list[dict] = []
        budget_remaining = self.max_daily_candidates

        n_clusters = len(clusters)
        start_idx = 0
        clusters_tried = 0
        if n_clusters > 0:
            start_idx = self._next_cluster_idx % n_clusters

            while budget_remaining > 0 and clusters_tried < n_clusters:
                ci = (start_idx + clusters_tried) % n_clusters
                cluster = clusters[ci]
                top_pairs = self._clusterer.get_top_pairs_by_corr(
                    cluster, n=len(cluster) * (len(cluster) - 1) // 2,
                )

                for stock1, stock2, cluster_corr in top_pairs:
                    if budget_remaining <= 0:
                        break

                    if stock2 in self.pairs or stock2 in position_symbols:
                        continue
                    if stock1 in self.pairs or stock1 in position_symbols:
                        continue

                    pair_key = frozenset((stock1, stock2))
                    if self.cooldown_days and pair_key in self._pair_evaluated_at:
                        last_eval = self._pair_evaluated_at[pair_key]
                        last_date = last_eval.date() if hasattr(last_eval, 'date') else last_eval
                        if (today - last_date).days < self.cooldown_days:
                            gate_counts['cooldown'] += 1
                            continue

                    s1 = _get_series(stock1)
                    if s1 is None or float(s1.iloc[-1]) < self.penny_threshold:
                        gate_counts['penny'] += 1
                        if s1 is not None:
                            new_penny_stocks.add(stock1)
                        continue

                    s2 = _get_series(stock2)
                    if s2 is None or float(s2.iloc[-1]) < self.penny_threshold:
                        gate_counts['penny'] += 1
                        if s2 is not None:
                            new_penny_stocks.add(stock2)
                        continue

                    pairs_scanned += 1
                    self._pair_evaluated_at[pair_key] = end_date

                    corr_long, corr_short = evaluator.get_correlation_dual(
                        s1, s2, self.corr_long_window, self.corr_short_window,
                    )
                    z_depth, z_raw = evaluator.compute_z_depth(
                        s1, s2, self.zscore_window,
                        self.entry_threshold, self.exit_threshold,
                    )
                    score = self._composite_score(corr_long, corr_short, z_depth)

                    candidates_found += 1
                    budget_remaining -= 1
                    scored_candidates.append({
                        'lead_stock': stock1,
                        'lag_stock': stock2,
                        'corr_long': corr_long,
                        'corr_short': corr_short,
                        'z_depth': z_depth,
                        'z_raw': z_raw,
                        'composite_score': score,
                    })

                clusters_tried += 1

            self._next_cluster_idx = (start_idx + 1) % n_clusters

        for sym in new_penny_stocks:
            self._failed_tickers.add(sym)
            self._db.mark_ticker_failed(sym, 'penny stock')

        # --- Phase 3: Unified portfolio construction ---
        existing_scored = [
            (symbol, pair)
            for symbol, pair in self.pairs.items()
            if pair.get('composite_score', -1) >= 0 and pair.get('action') != 'sell'
        ]

        all_scored: list[tuple[str, float, dict | None, str | None]] = []

        for symbol, pair in existing_scored:
            all_scored.append((symbol, pair['composite_score'], None, 'existing'))

        for cand in scored_candidates:
            all_scored.append((
                cand['lag_stock'],
                cand['composite_score'],
                cand,
                'candidate',
            ))

        all_scored.sort(key=lambda x: x[1], reverse=True)

        corr_short_values = [
            c['corr_short'] for c in scored_candidates
            if c['z_depth'] > 0 and not np.isnan(c['corr_short'])
        ]
        for symbol, pair in existing_scored:
            cs = pair.get('corr_short')
            if cs is not None and not np.isnan(cs) and pair.get('z_depth', 0) > 0:
                corr_short_values.append(cs)

        pool_corr = median(corr_short_values) if corr_short_values else 0.0

        # K is a quality-scaled fraction of max_k.  pool_corr / pivot gives the
        # raw scale; clamped to [quality_scale_min, quality_scale_max] so K
        # stays between max_k×min and max_k regardless of pool quality extremes.
        # Affordability is not encoded here — the buy loop's cash check handles it.
        quality_scale = max(
            self.quality_scale_min,
            min(pool_corr / self.quality_scale_pivot, self.quality_scale_max),
        )
        k_target = max(1, round(self.max_k * quality_scale))

        target_portfolio: dict[str, dict] = {}
        candidates_buy_ready = 0

        for symbol, score, cand_data, source in all_scored:
            if len(target_portfolio) >= k_target:
                break
            if source == 'existing':
                target_portfolio[symbol] = self.pairs[symbol]
            elif source == 'candidate' and cand_data is not None:
                if symbol in target_portfolio:
                    continue

                new_pair = {
                    'lead_stock': cand_data['lead_stock'],
                    'lag_stock': cand_data['lag_stock'],
                    'corr_long': cand_data['corr_long'],
                    'corr_short': cand_data['corr_short'],
                    'z_depth': cand_data['z_depth'],
                    'composite_score': cand_data['composite_score'],
                    'current_zscore': cand_data.get('z_raw'),
                    'action': 'buy',
                    'signal_type': 'zscore',
                    'zscore_window': self.zscore_window,
                    'entry_threshold': self.entry_threshold,
                    'exit_threshold': self.exit_threshold,
                }
                new_pair['pair_id'] = self._db.save_pair(new_pair, self._run_id)
                self.pairs[symbol] = new_pair
                target_portfolio[symbol] = new_pair
                candidates_buy_ready += 1
                print(
                    f"New candidate: {cand_data['lead_stock']} -> {symbol} | "
                    f"score={score:.3f} corr_s={cand_data['corr_short']:.3f} "
                    f"z_depth={cand_data['z_depth']:.2f}"
                )

        for symbol in list(self.pairs.keys()):
            pair = self.pairs[symbol]
            if pair.get('action') == 'sell':
                continue
            if symbol not in target_portfolio and symbol in position_symbols:
                pair['action'] = 'sell'
                pair['exit_reason'] = 'displaced'
                print(f"Displaced from target portfolio: {symbol} (score={pair.get('composite_score', 0):.3f})")
            elif symbol in target_portfolio and symbol not in position_symbols:
                pair['action'] = 'buy'

        print(
            f"Cluster {start_idx}/{n_clusters} "
            f"({clusters_tried} tried). "
            f"Scanned {pairs_scanned} pairs. "
            f"Gates: {gate_counts}. "
            f"Candidates scored: {candidates_found}. "
            f"Target K: {k_target}. "
            f"Pool quality (median corr_short): {pool_corr:.3f}. "
            f"New buys queued: {candidates_buy_ready}."
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
                            exit_reason=pair.get('exit_reason'),
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
        no_price_symbols: list[str] = []
        daily_new_buys = 0
        if buy_pairs:
            portfolio_value = self.portfolio_value
            available_cash = self.get_cash()
            current_deployed = portfolio_value - available_cash
            deployment_gap = max(0.0, self.target_deployed_pct * portfolio_value - current_deployed)
            n_candidates = len(buy_pairs)

            buy_pairs_ranked = sorted(
                buy_pairs, key=lambda p: p.get('composite_score', 0), reverse=True,
            )

            for pair in buy_pairs_ranked:
                score = pair.get('composite_score', 0.0)
                base_budget = (
                    self.min_position_pct + score * (self.max_position_pct - self.min_position_pct)
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
        long_corrs: list[float] = []
        for p in active_pairs:
            v = p.get('corr_long')
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv):
                long_corrs.append(fv)
        avg_corr = float(np.mean(long_corrs)) if long_corrs else 0.0

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
            avg_watchlist_ttl=None,
        )

    def on_strategy_end(self):
        self._db.close_run(self._run_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_ticker_metadata(self, tickers: list[str]) -> None:
        """
        Populate self._ticker_metadata with sector and ETF classification.

        Data is read from the DB first.  Tickers with no existing row are
        looked up via SEC EDGAR (CIK mapping + submissions API for SIC codes).
        Results are stored in the DB so subsequent runs pay nothing.
        """
        existing_df = self._db.get_ticker_metadata(tickers)
        known = set(existing_df['symbol'].tolist()) if not existing_df.empty else set()
        missing = [t for t in tickers if t not in known]

        if missing:
            self._fetch_sec_metadata(missing)
            new_df = self._db.get_ticker_metadata(missing)
            existing_df = (
                pd.concat([existing_df, new_df], ignore_index=True)
                if not existing_df.empty else new_df
            )

        self._ticker_metadata = {}
        if not existing_df.empty:
            for _, row in existing_df.iterrows():
                self._ticker_metadata[row['symbol']] = {
                    'sector': row['sector'] if pd.notna(row['sector']) else None,
                    'is_etf': bool(row['is_etf']),
                }

        self._validate_ticker_metadata(tickers)

    def _fetch_sec_metadata(self, tickers: list[str]) -> None:
        """
        Fetch sector metadata from SEC EDGAR for tickers missing from the DB.

        1. Download the bulk CIK-to-ticker mapping from SEC.
        2. For matched tickers, batch-fetch SIC codes from the submissions API
           (rate-limited to ~9 req/s to comply with SEC fair-access policy).
        3. Store results via DatabaseClient.upsert_sec_metadata.
        """
        import time as _time

        headers = {'User-Agent': 'LumiBob research@lumibob.local'}
        print(f"[BobsBrain] Fetching SEC EDGAR metadata for {len(tickers)} symbols...")

        try:
            resp = requests.get(
                'https://www.sec.gov/files/company_tickers_exchange.json',
                headers=headers, timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:
            print(f"[BobsBrain] SEC EDGAR bulk download failed: {exc}")
            self._store_empty_metadata(tickers)
            return

        sec_data = resp.json()
        fields = sec_data['fields']
        sec_df = pd.DataFrame(sec_data['data'], columns=fields)
        sec_df.columns = [c.lower() for c in sec_df.columns]
        cik_map = sec_df.drop_duplicates(subset='ticker').set_index(
            sec_df['ticker'].str.upper()
        )['cik'].to_dict()

        fetched_at = datetime.utcnow()
        records: list[dict] = []
        unmatched: list[str] = []

        for i, symbol in enumerate(tickers):
            cik = cik_map.get(symbol.upper())
            if cik is None:
                unmatched.append(symbol)
                continue

            cik_padded = str(int(cik)).zfill(10)
            url = f'https://data.sec.gov/submissions/CIK{cik_padded}.json'
            try:
                r = requests.get(url, headers=headers, timeout=5)
                if r.status_code == 200:
                    meta = r.json()
                    sic = meta.get('sic')
                    sic_sector = _sic_to_sector(sic)
                    records.append({
                        'symbol': symbol,
                        'sic_code': int(sic) if sic else None,
                        'sic_sector': sic_sector,
                        'is_etf': False,
                        'fetched_at': fetched_at,
                    })
            except Exception:
                unmatched.append(symbol)

            if (i + 1) % 100 == 0:
                print(f"[BobsBrain] SEC metadata progress: {i + 1}/{len(tickers)}")
                if records:
                    self._db.upsert_sec_metadata(records)
                    records = []
            _time.sleep(0.11)

        if records:
            self._db.upsert_sec_metadata(records)

        if unmatched:
            self._store_empty_metadata(unmatched)

        total_with_sector = len(tickers) - len(unmatched)
        print(
            f"[BobsBrain] SEC EDGAR: {total_with_sector}/{len(tickers)} tickers "
            f"matched, {len(unmatched)} unmatched"
        )

    def _store_empty_metadata(self, tickers: list[str]) -> None:
        """Store placeholder rows for tickers with no SEC data so we don't re-fetch."""
        fetched_at = datetime.utcnow()
        records = [
            {'symbol': t, 'sector': None, 'is_etf': False, 'fetched_at': fetched_at}
            for t in tickers
        ]
        self._db.upsert_ticker_metadata(records)

    def _validate_ticker_metadata(self, tickers: list[str]) -> None:
        """
        Log a coverage summary for the loaded metadata and warn when the
        unknown-sector partition is unexpectedly large.

        Thresholds (sector coverage fraction):
          < 20%  → ERROR  — sector partitioning is effectively useless; all
                             tickers land in the unknown bucket.
          20–50% → WARNING — notable unknown partition; quality may be degraded.
          >= 50% → INFO    — coverage sufficient for meaningful partitioning.
        """
        total = len(tickers)
        if total == 0:
            return
        with_sector = sum(
            1 for t in tickers if self._ticker_metadata.get(t, {}).get('sector')
        )
        etf_count = sum(
            1 for t in tickers if self._ticker_metadata.get(t, {}).get('is_etf')
        )
        coverage_pct = with_sector / total
        unknown_count = total - with_sector
        msg = (
            f"[BobsBrain] Ticker metadata: {with_sector}/{total} tickers have "
            f"sector data ({coverage_pct:.0%}), {etf_count} ETFs, "
            f"{total - etf_count} stocks"
        )
        if coverage_pct < 0.20:
            logging.error(
                "%s — coverage critically low; %d tickers will land in the "
                "unknown partition, making sector pre-partition ineffective. "
                "Clear the ticker_metadata table and re-run to trigger a fresh "
                "SEC EDGAR fetch.",
                msg,
                unknown_count,
            )
        elif coverage_pct < 0.50:
            logging.warning(
                "%s — coverage low; %d tickers will be clustered in the unknown "
                "partition, which may have poor intra-cluster correlation. "
                "Clear the ticker_metadata table and re-run to trigger a fresh "
                "SEC EDGAR fetch if this number is unexpectedly high.",
                msg,
                unknown_count,
            )
        else:
            print(msg)

    def _composite_score(
        self,
        corr_long: float,
        corr_short: float,
        z_depth: float,
    ) -> float:
        """
        Weighted composite of the three scoring components.  Each correlation
        value is clamped to [0, 1] before weighting; z_depth is already in
        that range by construction.
        """
        cl = max(corr_long, 0.0) if not np.isnan(corr_long) else 0.0
        cs = max(corr_short, 0.0) if not np.isnan(corr_short) else 0.0
        return (
            self.w_corr_long * min(cl, 1.0)
            + self.w_corr_short * min(cs, 1.0)
            + self.w_z_depth * z_depth
        )
