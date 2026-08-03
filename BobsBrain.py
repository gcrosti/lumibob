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
from StockEvaluator import StockEvaluator, halflife_to_score
from TickerClusterer import TickerClusterer

load_dotenv()

# ADF p-value ceiling used to normalise coint_score: pairs at or above this
# value score 0 on the cointegration component.  Shared between the discovery
# loop and the re-score loop so both produce comparable values.
_COINT_PVALUE_CEILING = 0.20

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
        # Opaque token set by the tuning engine so a parallel Optuna worker can
        # recover exactly its own run from backtest_runs.settings. Not a
        # strategy parameter; None outside tuning runs.
        self.tuning_trial_token = self.parameters.get('tuning_trial_token', None)
        # True during tuning studies: prices come from the DB cache only, no
        # Alpaca fetches or failure-marking. Operational param, not tunable.
        self.price_cache_only = bool(self.parameters.get('price_cache_only', False))
        # Calendar days of price history for scoring (must span corr windows in bars).
        self.lookback_window = self.parameters.get('lookback_window', 130)
        # Min days between cluster recomputes; None = recompute only when cache cold.
        self.cluster_recompute_days = self.parameters.get('cluster_recompute_days', None)

        self._ticker_metadata: dict[str, dict] = {}
        self._metadata_loaded = False

        # Position size as a fraction of portfolio.  Sizing is FLAT within the
        # selected book: nothing in the entry criteria ranks pair *quality*, so
        # betting more on any qualified pair is unvalidated.  min_position_pct
        # is the flat allocation; max_position_pct caps the deployment-gap boost.
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
        # Correlations are persisted for observability and cluster health; they
        # no longer select or rank candidates (PR #50: the composite did not
        # select positively, and post-gate correlation ranks outcomes
        # negatively — see docs/plans/2026-08-01_entry-criteria-overhaul.md).
        self.corr_long_window = self.parameters.get('corr_long_window', 90)
        self.corr_short_window = self.parameters.get('corr_short_window', 20)
        # Ceiling for half-life scoring: pairs with halflife >= this score 0.
        self.max_halflife_days = self.parameters.get('max_halflife_days', 60)

        # Emergency floor: a candidate whose expected reversion is worth less
        # than this many bps of gross notional is never bought, however it
        # ranks.  Cost-viability backstop, not a selector — sized at ~1x the
        # round-trip friction estimate so it rarely binds.
        self.min_expected_gross_bps = self.parameters.get('min_expected_gross_bps', 25.0)

        # Max new pairs QUALIFIED (gates passed, fully scored) per day.
        self.max_daily_candidates = self.parameters.get('max_daily_candidates', 200)
        # Max pairs EXAMINED per day.  The entry gates run before the expensive
        # cointegration work, so most examined pairs cost only one z-score
        # computation; this cap bounds that cheap scan.
        self.max_daily_examined = self.parameters.get('max_daily_examined', 2000)
        # Days before the same unordered pair can be scored again.
        self.cooldown_days = self.parameters.get('cooldown_days', 7)

        # Minimum price for a ticker to pass the penny-stock filter.
        self.penny_threshold = self.parameters.get('penny_threshold', 5.0)

        # Target portfolio size.  Fixed: the former dynamic-K quality_scale
        # clipped to 1.0 on every scoring date measured, so K always equalled
        # max_k anyway, and every dynamic scheme tested underperformed a fixed
        # K — reducing K hurts (diversification is doing real work) and the
        # natural scaling signals peak on regime-break dates.  See
        # docs/plans/2026-08-01_entry-criteria-overhaul.md §4.
        self.max_k = self.parameters.get('max_k', 20)

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

        # H1: continuous short-leg scaling.
        # short_leg_fraction ∈ [0.0, 1.0]: fraction of the long notional to short the
        # lead stock.  0.0 = long-only; 1.0 = full dollar-neutral hedge.
        # Backward compat: if the deprecated enable_short_leg=True is set and
        # short_leg_fraction is not explicitly provided, default to 1.0 (full hedge).
        _enable_short_leg_legacy = bool(self.parameters.get('enable_short_leg', False))
        _slf_default = 1.0 if _enable_short_leg_legacy else 0.0
        self.short_leg_fraction = float(
            self.parameters.get('short_leg_fraction', _slf_default)
        )
        self.short_leg_fraction = max(0.0, min(1.0, self.short_leg_fraction))

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
        self._db.migrate_pairs_score_components()
        self._db.migrate_ticker_metadata()
        self._db.migrate_failed_tickers()
        self._db.migrate_coint_cache()
        self._db.migrate_short_leg()
        self._db.migrate_snapshot_deployment()
        self._alpaca = AlpacaClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=os.getenv('ALPACA_IS_PAPER', 'true').lower() == 'true',
            mode=self._run_mode,
        )
        self._cache = StockDataCache(
            self._db, self._alpaca, cache_only=self.price_cache_only,
        )
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
        # Window-independent failures only (penny stocks, dead quotes).
        # Window-scoped 'no data' rows are applied per-fetch by StockDataCache
        # — loading them globally here poisoned the whole universe once a mass
        # fetch failure was recorded (2026-07-14 incident).
        self._failed_tickers: set[str] = set(self._db.get_failed_tickers_global())

        self._run_id = secrets.token_hex(3)
        self.pairs: dict[str, dict] = self._db.load_active_pairs(self._run_id)

        tickers = self._db.get_tickers()
        if not tickers:
            print("[BobsBrain] Tickers table empty, fetching tradeable assets from Alpaca...")
            tickers = self._alpaca.get_tradeable_assets()
            self._db.upsert_tickers(tickers, 'ALPACA')
        raw_universe_count = len(tickers)
        tickers = [t for t in tickers if t not in self._failed_tickers]
        self._check_universe_health(raw_universe_count, len(tickers))

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
                # Scoring / observability windows
                'corr_long_window': self.corr_long_window,
                'corr_short_window': self.corr_short_window,
                'max_halflife_days': self.max_halflife_days,
                # Entry criteria
                'min_expected_gross_bps': self.min_expected_gross_bps,
                # Discovery
                'max_daily_candidates': self.max_daily_candidates,
                'max_daily_examined': self.max_daily_examined,
                'cooldown_days': self.cooldown_days,
                # Filters
                'penny_threshold': self.penny_threshold,
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
                # short_leg_fraction replaces the deprecated enable_short_leg boolean.
                'short_leg_fraction': self.short_leg_fraction,
                # Tuning-run attribution (null outside Optuna studies).
                'tuning_trial_token': self.tuning_trial_token,
                'price_cache_only': self.price_cache_only,
            },
        )

    @staticmethod
    def _check_universe_health(raw_count: int, filtered_count: int) -> None:
        """
        Abort startup when the failed-ticker filter collapses the universe.

        A run with a near-empty universe scans nothing, trades nothing, and
        still *completes* — a silent failure that poisons downstream analysis
        and, in tuning studies, burns trial quota on meaningless scores
        (2026-07-14 incident: 89 of 100 trials). Failing loudly here turns
        that into an immediately visible crash instead.
        """
        if raw_count > 0 and filtered_count == 0:
            raise RuntimeError(
                f'Universe collapsed: 0 of {raw_count} tickers survived the '
                f'failed-ticker filter — refusing to run with an empty universe.'
            )
        if raw_count >= 500 and filtered_count < 0.2 * raw_count:
            raise RuntimeError(
                f'Universe collapsed: only {filtered_count} of {raw_count} '
                f'tickers survived the failed-ticker filter (<20%). This '
                f'indicates poisoned failed_tickers state, not real data gaps.'
            )

    def before_market_opens(self):
        """
        Gate-and-rank pipeline. Runs once per trading day.

        Phase 1: Cheap gates (penny, cooldown) reduce the within-cluster universe.
        Phase 2: Entry gates — direction + dislocation (z <= -entry_threshold)
                 and the magnitude floor (expected_gross >= min_expected_gross_bps)
                 — applied BEFORE the expensive cointegration work.  Survivors
                 are fully scored; existing positions are re-valued on their
                 remaining expected reversion.
        Phase 3: Rank everything by expected_gross_bps, take the top max_k, and
                 set actions (buy/sell) for on_trading_iteration().

        Design rationale: docs/plans/2026-08-01_entry-criteria-overhaul.md
        """
        evaluator = StockEvaluator()
        end_date = self.get_datetime()
        start_date = end_date - timedelta(days=self.lookback_window)
        today = end_date.date() if hasattr(end_date, 'date') else end_date

        # Load cointegration cache for today's window upfront; avoids repeated
        # ADF tests for pairs that have already been evaluated on this date.
        self._coint_cache: dict[tuple[str, str], tuple[float, float | None]] = \
            self._db.load_coint_cache(today, self.lookback_window)
        self._coint_cache_new: dict[tuple[str, str], tuple[float, float | None]] = {}

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
                pair['expected_gross_bps'] = -1.0
                continue

            corr_long, corr_short = evaluator.get_correlation_dual(
                lead_data, lag_data, self.corr_long_window, self.corr_short_window,
            )

            action, current_z = evaluator.get_zscore_action(
                lead_data, lag_data,
                window=self.zscore_window,
                entry_threshold=self.entry_threshold,
                exit_threshold=self.exit_threshold,
            )

            # An open position competes for its slot on the reversion it has
            # LEFT, not the one it was opened on: a pair that has already run
            # most of the way to its exit is worth less than a fresh
            # dislocation of the same quality.  (The exit rule still closes it
            # on its own terms once z reverts past exit_threshold.)
            metrics = evaluator.compute_entry_metrics(
                lead_data, lag_data, self.zscore_window, self.exit_threshold,
            )
            pair['corr_long'] = corr_long
            pair['corr_short'] = corr_short
            pair['current_zscore'] = (
                current_z if current_z is not None
                else (metrics.z if metrics else None)
            )
            pair['spread_std_bps'] = metrics.spread_std_bps if metrics else None
            pair['expected_gross_bps'] = (
                metrics.expected_gross_bps if metrics else 0.0
            )

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

        pairs_scanned = 0
        candidates_found = 0
        gate_counts = {'penny': 0, 'cooldown': 0, 'dislocation': 0, 'magnitude': 0}
        new_penny_stocks: set[str] = set()
        scored_candidates: list[dict] = []
        budget_remaining = self.max_daily_candidates
        examined_remaining = self.max_daily_examined

        n_clusters = len(clusters)
        start_idx = 0
        clusters_tried = 0
        if n_clusters > 0:
            start_idx = self._next_cluster_idx % n_clusters

            while (budget_remaining > 0 and examined_remaining > 0
                   and clusters_tried < n_clusters):
                ci = (start_idx + clusters_tried) % n_clusters
                cluster = clusters[ci]
                top_pairs = self._clusterer.get_top_pairs_by_corr(
                    cluster, n=len(cluster) * (len(cluster) - 1) // 2,
                )

                for stock1, stock2, cluster_corr in top_pairs:
                    if budget_remaining <= 0 or examined_remaining <= 0:
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

                    examined_remaining -= 1

                    # --- Entry gates, BEFORE the expensive cointegration work ---
                    # Both gates come from one z-score computation; the ADF test
                    # below costs ~10x more, so gating first keeps the daily scan
                    # cheap (measured in tuning/studies/scoring_replay.py).
                    metrics = evaluator.compute_entry_metrics(
                        s1, s2, self.zscore_window, self.exit_threshold,
                    )
                    if metrics is None:
                        continue

                    # Gate 1 — direction + dislocation.  The strategy buys the
                    # lag leg when it is CHEAP relative to the lead, so only a
                    # sufficiently negative z is tradeable.  Nothing checked
                    # this before: entry was purely rank-based, and only 6% of
                    # live entries actually met it.
                    if metrics.z > -self.entry_threshold:
                        gate_counts['dislocation'] += 1
                        continue

                    # Gate 2 — emergency floor on trade magnitude.
                    if metrics.expected_gross_bps < self.min_expected_gross_bps:
                        gate_counts['magnitude'] += 1
                        continue

                    pairs_scanned += 1
                    self._pair_evaluated_at[pair_key] = end_date

                    corr_long, corr_short = evaluator.get_correlation_dual(
                        s1, s2, self.corr_long_window, self.corr_short_window,
                    )

                    # Cointegration / half-life scores — cache-first.
                    cache_key = (stock1, stock2)
                    if cache_key in self._coint_cache:
                        coint_pvalue, halflife_days = self._coint_cache[cache_key]
                    else:
                        ss = evaluator.compute_spread_scores(
                            s1, s2, max_halflife_days=float(self.max_halflife_days),
                        )
                        coint_pvalue, halflife_days = ss.coint_pvalue, ss.halflife_days
                        self._coint_cache[cache_key] = (coint_pvalue, halflife_days)
                        self._coint_cache_new[cache_key] = (coint_pvalue, halflife_days)

                    # Observability only — none of these select or rank
                    # candidates any more (PR #50; entry-criteria overhaul).
                    coint_score = max(0.0, 1.0 - coint_pvalue / _COINT_PVALUE_CEILING)
                    halflife_score = halflife_to_score(halflife_days, self.max_halflife_days)

                    candidates_found += 1
                    budget_remaining -= 1
                    scored_candidates.append({
                        'lead_stock': stock1,
                        'lag_stock': stock2,
                        'corr_long': corr_long,
                        'corr_short': corr_short,
                        'z_raw': metrics.z,
                        'coint_pvalue': coint_pvalue,
                        'halflife_days': halflife_days,
                        # Selection quantity: bps of gross notional expected
                        # from reverting to the exit threshold.
                        'expected_gross_bps': metrics.expected_gross_bps,
                        'spread_std_bps': metrics.spread_std_bps,
                        # Component scores — stored for post-hoc analysis
                        'score_corr_long': min(max(corr_long, 0.0), 1.0),
                        'score_corr_short': min(max(corr_short, 0.0), 1.0),
                        'score_coint': coint_score,
                        'score_halflife': halflife_score,
                        'min_expected_gross_bps': self.min_expected_gross_bps,
                    })

                clusters_tried += 1

            self._next_cluster_idx = (start_idx + 1) % n_clusters

        # Persist newly computed cointegration results to the DB cache.
        if self._coint_cache_new:
            self._db.write_coint_cache(self._coint_cache_new, today, self.lookback_window)
            self._coint_cache_new = {}

        for sym in new_penny_stocks:
            self._failed_tickers.add(sym)
            self._db.mark_ticker_failed(sym, 'penny stock')

        # --- Phase 3: Unified portfolio construction ---
        # Existing positions and new candidates compete on the same quantity:
        # expected remaining reversion, in bps of gross notional.
        existing_scored = [
            (symbol, pair)
            for symbol, pair in self.pairs.items()
            if pair.get('expected_gross_bps', -1) >= 0 and pair.get('action') != 'sell'
        ]

        all_scored: list[tuple[str, float, dict | None, str | None]] = []

        for symbol, pair in existing_scored:
            all_scored.append((symbol, pair['expected_gross_bps'], None, 'existing'))

        for cand in scored_candidates:
            all_scored.append((
                cand['lag_stock'],
                cand['expected_gross_bps'],
                cand,
                'candidate',
            ))

        all_scored.sort(key=lambda x: x[1], reverse=True)

        # Pool correlation is retained as an observability metric only; it no
        # longer scales K (see the max_k comment in initialize()).
        corr_short_values = [
            c['corr_short'] for c in scored_candidates
            if not np.isnan(c['corr_short'])
        ]
        for symbol, pair in existing_scored:
            cs = pair.get('corr_short')
            if cs is not None and not np.isnan(cs):
                corr_short_values.append(cs)

        pool_corr = median(corr_short_values) if corr_short_values else 0.0
        k_target = self.max_k

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
                    'expected_gross_bps': cand_data['expected_gross_bps'],
                    'spread_std_bps': cand_data['spread_std_bps'],
                    'current_zscore': cand_data.get('z_raw'),
                    'coint_pvalue': cand_data.get('coint_pvalue', 1.0),
                    'halflife_days': cand_data.get('halflife_days'),
                    'action': 'buy',
                    'signal_type': 'zscore',
                    'zscore_window': self.zscore_window,
                    'entry_threshold': self.entry_threshold,
                    'exit_threshold': self.exit_threshold,
                    # Observability components — forwarded for DB storage
                    'score_corr_long': cand_data.get('score_corr_long'),
                    'score_corr_short': cand_data.get('score_corr_short'),
                    'score_coint': cand_data.get('score_coint'),
                    'score_halflife': cand_data.get('score_halflife'),
                    'min_expected_gross_bps': cand_data.get('min_expected_gross_bps'),
                }
                new_pair['lead_short_qty'] = None
                new_pair['pair_id'] = self._db.save_pair(new_pair, self._run_id)
                self.pairs[symbol] = new_pair
                target_portfolio[symbol] = new_pair
                candidates_buy_ready += 1
                print(
                    f"New candidate: {cand_data['lead_stock']} -> {symbol} | "
                    f"exp_gross={score:.0f}bps z={cand_data['z_raw']:.2f} "
                    f"corr_s={cand_data['corr_short']:.3f}"
                )

        for symbol in list(self.pairs.keys()):
            pair = self.pairs[symbol]
            if pair.get('action') == 'sell':
                continue
            if symbol not in target_portfolio and symbol in position_symbols:
                pair['action'] = 'sell'
                pair['exit_reason'] = 'displaced'
                print(f"Displaced from target portfolio: {symbol} "
                      f"(exp_gross={pair.get('expected_gross_bps', 0):.0f}bps)")
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
                            leg='long',
                        )
                    else:
                        print(f"Warning: could not log sell trade for {symbol} — price unavailable.")

                if self.short_leg_fraction > 0.0:
                    lead_sym = pair['lead_stock']
                    sq = pair.get('lead_short_qty')
                    if sq is not None and float(sq) > 0:
                        cover_qty = float(sq)
                        order_lead = self.create_order(lead_sym, cover_qty, 'buy')
                        self.submit_order(order_lead)
                        lead_px = self.get_last_price(lead_sym)
                        if lead_px and lead_px > 0:
                            self._db.log_trade(
                                run_id=self._run_id,
                                symbol=lead_sym,
                                side='buy',
                                quantity=cover_qty,
                                price=float(lead_px),
                                filled_at=now,
                                pair_id=pair.get('pair_id'),
                                exit_reason=pair.get('exit_reason'),
                                leg='short',
                            )
                        else:
                            print(
                                f"Warning: could not log cover trade for {lead_sym} — price unavailable."
                            )

                self._db.deactivate_pair(symbol, self._run_id)
                to_remove.append(symbol)

        for symbol in to_remove:
            self.pairs.pop(symbol)

        # --- Execute buys ---
        existing_positions = self.get_positions()
        existing_position_symbols = {p.symbol for p in existing_positions}
        held_long = {p.symbol for p in existing_positions if p.quantity > 0}
        held_short = {p.symbol for p in existing_positions if p.quantity < 0}
        buy_pairs = [
            pair for symbol, pair in self.pairs.items()
            if pair['action'] == 'buy' and symbol not in existing_position_symbols
        ]

        new_buy_symbols: set[str] = set()
        new_short_symbols: set[str] = set()
        no_price_symbols: list[str] = []
        daily_new_buys = 0
        if buy_pairs:
            portfolio_value = self.portfolio_value
            gross_short_notional = sum(
                abs(pos.quantity) * (self.get_last_price(pos.symbol) or 0)
                for pos in existing_positions if pos.quantity < 0
            )
            available_cash = self.get_cash() - gross_short_notional
            current_deployed = portfolio_value - available_cash
            deployment_gap = max(0.0, self.target_deployed_pct * portfolio_value - current_deployed)
            n_candidates = len(buy_pairs)

            buy_pairs_ranked = sorted(
                buy_pairs, key=lambda p: p.get('expected_gross_bps', 0), reverse=True,
            )

            for pair in buy_pairs_ranked:
                # FLAT sizing.  Size previously scaled with composite_score, but
                # nothing in the entry criteria ranks pair *quality* — and the
                # selection quantity (expected_gross) correlates with disaster
                # rate, so sizing by it would put the most capital in the
                # fattest-tailed trades.  Order still follows expected_gross so
                # that, under a cash constraint, the largest opportunities fill
                # first.  See docs/plans/2026-08-01_entry-criteria-overhaul.md §3.
                base_budget = self.min_position_pct * portfolio_value
                if deployment_gap > 0 and n_candidates > 0:
                    gap_share = deployment_gap / n_candidates
                    base_budget = min(base_budget + gap_share, self.max_position_pct * portfolio_value)
                per_stock_budget = base_budget

                # short_leg_fraction scales the short notional relative to the long leg.
                # effective_cost = long budget + short budget so the cash gate sees
                # the full capital required for both legs.
                effective_cost = per_stock_budget * (1.0 + self.short_leg_fraction)
                if available_cash < effective_cost:
                    continue  # budget exceeds cash; try the next (cheaper) candidate

                if self.short_leg_fraction > 0.0 and (
                    pair['lag_stock'] in held_short
                    or pair['lead_stock'] in held_long
                    or pair['lag_stock'] in new_short_symbols
                    or pair['lead_stock'] in new_buy_symbols
                ):
                    print(
                        f"Skipping {pair['lag_stock']}/{pair['lead_stock']}: "
                        f"mirror-pair conflict — same stock held in opposite direction."
                    )
                    continue

                lead_qty: float | None = None
                lead_px: float | None = None
                if self.short_leg_fraction > 0.0:
                    lp = self.get_last_price(pair['lead_stock'])
                    if not lp or lp <= 0:
                        continue
                    lead_px = float(lp)
                    # short notional = per_stock_budget * short_leg_fraction;
                    # short_qty = that notional / lead price
                    short_notional = per_stock_budget * self.short_leg_fraction
                    lead_qty = round(short_notional / lead_px, 6)
                    if lead_qty <= 0:
                        continue

                price = self.get_last_price(pair['lag_stock'])
                if price and price > 0:
                    quantity = round(per_stock_budget / price, 6)
                    if quantity > 0:
                        order = self.create_order(pair['lag_stock'], quantity, 'buy')
                        self.submit_order(order)
                        available_cash -= effective_cost
                        deployment_gap = max(0.0, deployment_gap - effective_cost)
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
                            leg='long',
                        )
                        if self.short_leg_fraction > 0.0 and lead_qty is not None and lead_qty > 0 and lead_px:
                            order_s = self.create_order(pair['lead_stock'], lead_qty, 'sell')
                            self.submit_order(order_s)
                            self._db.log_trade(
                                run_id=self._run_id,
                                symbol=pair['lead_stock'],
                                side='sell',
                                quantity=float(lead_qty),
                                price=float(lead_px),
                                filled_at=now,
                                pair_id=pair.get('pair_id'),
                                leg='short',
                            )
                            pair['lead_short_qty'] = float(lead_qty)
                            new_short_symbols.add(pair['lead_stock'])
                            pid = pair.get('pair_id')
                            if pid is not None:
                                self._db.update_pair_lead_short_qty(int(pid), float(lead_qty))
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

        end_positions = self.get_positions()
        gross_long = sum(
            pos.quantity * (self.get_last_price(pos.symbol) or 0)
            for pos in end_positions if pos.quantity > 0
        )
        gross_short = sum(
            abs(pos.quantity) * (self.get_last_price(pos.symbol) or 0)
            for pos in end_positions if pos.quantity < 0
        )
        gross_long_pct = round(gross_long / portfolio_value, 4) if portfolio_value else 0.0
        gross_short_pct = round(gross_short / portfolio_value, 4) if portfolio_value else 0.0

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
        self.add_line("gross_long_pct",  gross_long_pct)
        self.add_line("gross_short_pct", gross_short_pct)

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
            gross_long_pct=gross_long_pct,
            gross_short_pct=gross_short_pct,
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

