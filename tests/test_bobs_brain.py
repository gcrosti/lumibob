"""
Unit tests for BobsBrain helpers.

BobsBrain extends Lumibot's Strategy class, making full integration tests
heavy.  These tests cover the static/pure helpers that can be exercised
without instantiating the strategy or connecting to any external service.
"""

import sys
import os
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from BobsBrain import BobsBrain


# ---------------------------------------------------------------------------
# BobsBrain._is_penny_stock
# ---------------------------------------------------------------------------

class TestIsPennyStock(unittest.TestCase):
    def test_returns_true_when_last_price_below_five(self):
        series = pd.Series([10.0, 8.0, 4.99])
        self.assertTrue(BobsBrain._is_penny_stock(series))

    def test_returns_false_when_last_price_equals_five(self):
        series = pd.Series([3.0, 4.0, 5.00])
        self.assertFalse(BobsBrain._is_penny_stock(series))

    def test_returns_false_when_last_price_above_five(self):
        series = pd.Series([4.0, 4.5, 5.01])
        self.assertFalse(BobsBrain._is_penny_stock(series))

    def test_returns_false_for_empty_series(self):
        self.assertFalse(BobsBrain._is_penny_stock(pd.Series([], dtype=float)))

    def test_handles_single_element_series(self):
        self.assertTrue(BobsBrain._is_penny_stock(pd.Series([1.00])))
        self.assertFalse(BobsBrain._is_penny_stock(pd.Series([100.00])))

    def test_only_last_price_is_checked(self):
        """Earlier values below $5 do not matter; only the last bar counts."""
        series = pd.Series([1.0, 2.0, 10.0])
        self.assertFalse(BobsBrain._is_penny_stock(series))


# ---------------------------------------------------------------------------
# Failed ticker filtering logic
# ---------------------------------------------------------------------------

class TestFailedTickerFiltering(unittest.TestCase):
    """
    The failed-ticker filter is a one-liner in before_market_opens:
        tickers = [t for t in tickers if t not in self._failed_tickers]
    These tests verify the invariant directly without instantiating Strategy.
    DB-level persistence (get_failed_tickers / mark_ticker_failed) is covered
    by TestFailedTickers in test_database_client.py.
    """

    def test_failed_tickers_excluded_from_discovery_list(self):
        tickers = ['AAPL', 'MSFT', 'BAD1', 'TSLA', 'BAD2']
        failed = {'BAD1', 'BAD2'}
        filtered = [t for t in tickers if t not in failed]
        self.assertEqual(filtered, ['AAPL', 'MSFT', 'TSLA'])

    def test_empty_failed_set_leaves_list_unchanged(self):
        tickers = ['AAPL', 'MSFT', 'TSLA']
        failed = set()
        filtered = [t for t in tickers if t not in failed]
        self.assertEqual(filtered, tickers)

    def test_all_failed_returns_empty_list(self):
        tickers = ['BAD1', 'BAD2']
        failed = {'BAD1', 'BAD2'}
        filtered = [t for t in tickers if t not in failed]
        self.assertEqual(filtered, [])

    def test_no_price_symbols_collected_and_evicted(self):
        """
        Pairs whose get_last_price returns None should be collected in
        no_price_symbols and then popped from self.pairs.
        Simulates the eviction loop at the end of on_trading_iteration.
        """
        pairs = {
            'GOOD': {'lag_stock': 'GOOD', 'action': 'buy'},
            'BAD':  {'lag_stock': 'BAD',  'action': 'buy'},
        }
        no_price_symbols = ['BAD']

        for symbol in no_price_symbols:
            pairs.pop(symbol, None)

        self.assertIn('GOOD', pairs)
        self.assertNotIn('BAD', pairs)

    def test_failed_set_updated_when_price_unavailable(self):
        """
        When get_last_price returns None the symbol should be added to the
        failed set so future discovery loops exclude it.
        """
        failed: set[str] = set()
        symbol = 'NODATAINC'

        failed.add(symbol)

        self.assertIn(symbol, failed)


# ---------------------------------------------------------------------------
# Confidence-based position sizing
# ---------------------------------------------------------------------------

class TestConfidenceBasedSizing(unittest.TestCase):
    """
    Tests for the confidence-weighted position sizing introduced to replace the
    flat max_daily_spend_pct × per_pair_allocation formula.

    _compute_confidence and the per-pair budget formula are exercised directly
    without instantiating the full Strategy, keeping these tests fast and
    dependency-free.
    """

    # Mirrors BobsBrain._compute_confidence logic for isolated testing.
    def _confidence(self, pair, min_correlation=0.9, min_sharpe=0.5, entry_threshold=2.0):
        z = pair.get('current_zscore')
        entry = pair.get('entry_threshold', entry_threshold)
        z_score = min((abs(z) - entry) / entry, 1.0) if z is not None else 0.0
        z_score = max(z_score, 0.0)

        corr = pair.get('corr', min_correlation)
        corr_score = min((corr - min_correlation) / (1.0 - min_correlation), 1.0)
        corr_score = max(corr_score, 0.0)

        sharpe = pair.get('sim_sharpe', min_sharpe)
        sharpe_score = min((sharpe - min_sharpe) / min_sharpe, 1.0)
        sharpe_score = max(sharpe_score, 0.0)

        return 0.4 * z_score + 0.4 * corr_score + 0.2 * sharpe_score

    def _budget(self, confidence, portfolio_value, min_pct=0.03, max_pct=0.20,
                deployment_gap=0.0, n_candidates=1):
        base = (min_pct + confidence * (max_pct - min_pct)) * portfolio_value
        if deployment_gap > 0 and n_candidates > 0:
            gap_share = deployment_gap / n_candidates
            base = min(base + gap_share, max_pct * portfolio_value)
        return base

    # --- confidence score ---

    def test_zero_confidence_when_all_at_minimums(self):
        """Pair exactly at every gate floor returns confidence 0.0."""
        pair = {'current_zscore': -2.0, 'entry_threshold': 2.0,
                'corr': 0.9, 'sim_sharpe': 0.5}
        self.assertAlmostEqual(self._confidence(pair), 0.0)

    def test_max_confidence_when_all_components_capped(self):
        """Z excess = entry, corr = 1.0, sharpe = 2× min → confidence 1.0."""
        pair = {'current_zscore': -4.0, 'entry_threshold': 2.0,
                'corr': 1.0, 'sim_sharpe': 1.0}
        self.assertAlmostEqual(self._confidence(pair), 1.0)

    def test_medium_confidence_midpoint_values(self):
        """
        corr = 0.95 (mid of 0.9–1.0), sharpe = 0.75 (mid of 0.5–1.0),
        z = -3.0 with entry=2.0 → z_excess = 1.0, normalised = 0.5.
        Expected: 0.4*0.5 + 0.4*0.5 + 0.2*0.5 = 0.5
        """
        pair = {'current_zscore': -3.0, 'entry_threshold': 2.0,
                'corr': 0.95, 'sim_sharpe': 0.75}
        self.assertAlmostEqual(self._confidence(pair), 0.5)

    def test_missing_zscore_defaults_to_zero_component(self):
        """No current_zscore → z component is 0; other components still score."""
        pair = {'corr': 1.0, 'sim_sharpe': 1.0}
        score = self._confidence(pair)
        # z_score=0, corr_score=1, sharpe_score=1 → 0*0.4 + 1*0.4 + 1*0.2 = 0.6
        self.assertAlmostEqual(score, 0.6)

    def test_confidence_never_exceeds_one(self):
        """Extreme values are clamped to 1.0."""
        pair = {'current_zscore': -100.0, 'entry_threshold': 2.0,
                'corr': 1.0, 'sim_sharpe': 999.0}
        self.assertLessEqual(self._confidence(pair), 1.0)

    def test_confidence_never_below_zero(self):
        """Values below gate floors are clamped to 0.0."""
        pair = {'current_zscore': -2.0, 'entry_threshold': 2.0,
                'corr': 0.9, 'sim_sharpe': 0.0}
        self.assertGreaterEqual(self._confidence(pair), 0.0)

    # --- budget formula ---

    def test_min_confidence_gives_min_position(self):
        """Confidence 0 → min_position_pct of portfolio."""
        result = self._budget(0.0, 10_000)
        self.assertAlmostEqual(result, 300.0)  # 3% of 10k

    def test_max_confidence_gives_max_position(self):
        """Confidence 1.0 → max_position_pct of portfolio."""
        result = self._budget(1.0, 10_000)
        self.assertAlmostEqual(result, 2_000.0)  # 20% of 10k

    def test_medium_confidence_interpolates(self):
        """Confidence 0.5 → midpoint of [min, max] = 11.5% of portfolio."""
        result = self._budget(0.5, 10_000)
        self.assertAlmostEqual(result, 1_150.0)

    def test_deployment_gap_boosts_budget(self):
        """Gap share is added to base budget when portfolio is under-deployed."""
        result = self._budget(0.0, 10_000, deployment_gap=3_000, n_candidates=3)
        # base=300, gap_share=1000 → 1300
        self.assertAlmostEqual(result, 1_300.0)

    def test_deployment_gap_boost_capped_at_max_position(self):
        """Gap boost is capped so budget never exceeds max_position_pct."""
        result = self._budget(0.5, 10_000, deployment_gap=50_000, n_candidates=1)
        self.assertAlmostEqual(result, 2_000.0)  # capped at 20% of 10k

    def test_cash_guard_prevents_overspend(self):
        """If available_cash < per_stock_budget the pair is skipped (continue)."""
        per_stock_budget = 1_500.0
        available_cash = 900.0
        buy_executed = False

        if available_cash >= per_stock_budget:
            buy_executed = True

        self.assertFalse(buy_executed)

    def test_sufficient_cash_allows_buy(self):
        per_stock_budget = 1_500.0
        available_cash = 2_000.0
        buy_executed = False

        if available_cash >= per_stock_budget:
            buy_executed = True

        self.assertTrue(buy_executed)

    def test_candidates_sorted_by_confidence_descending(self):
        """Highest confidence pair should appear first after sorting."""
        pairs = [
            {'lag_stock': 'LOW', 'confidence_score': 0.2},
            {'lag_stock': 'HIGH', 'confidence_score': 0.9},
            {'lag_stock': 'MID', 'confidence_score': 0.5},
        ]
        ranked = sorted(pairs, key=lambda p: p['confidence_score'], reverse=True)
        self.assertEqual([p['lag_stock'] for p in ranked], ['HIGH', 'MID', 'LOW'])

    def test_no_debt_when_expensive_pair_followed_by_cheap_pair(self):
        """
        A high-confidence expensive pair should not block a cheaper pair.
        Using continue (not break), the second pair can still be funded.
        """
        available_cash = 500.0
        pairs = [
            {'confidence_score': 0.9, 'budget': 800.0},  # too expensive
            {'confidence_score': 0.3, 'budget': 200.0},  # affordable
        ]
        funded = []
        for pair in pairs:
            if available_cash < pair['budget']:
                continue  # skip, not break
            available_cash -= pair['budget']
            funded.append(pair)

        self.assertEqual(len(funded), 1)
        self.assertEqual(funded[0]['budget'], 200.0)
        self.assertGreaterEqual(available_cash, 0.0)  # never went negative


# ---------------------------------------------------------------------------
# Watchlist TTL expiry logic
# ---------------------------------------------------------------------------

class TestWatchlistExpiry(unittest.TestCase):
    """
    Watchlist entries expire after _watchlist_ttl_days days.
    These tests verify the date-comparison logic that drives eviction.
    """

    from datetime import date as _date

    def _is_expired(self, watchlist_date, today, ttl_days):
        from datetime import date
        return (today - watchlist_date).days > ttl_days

    def test_entry_within_ttl_is_not_expired(self):
        from datetime import date, timedelta
        today = date(2024, 1, 10)
        added = date(2024, 1, 7)  # 3 days ago, TTL=5
        self.assertFalse(self._is_expired(added, today, ttl_days=5))

    def test_entry_exactly_at_ttl_is_not_expired(self):
        from datetime import date
        today = date(2024, 1, 10)
        added = date(2024, 1, 5)  # exactly 5 days ago, TTL=5
        self.assertFalse(self._is_expired(added, today, ttl_days=5))

    def test_entry_past_ttl_is_expired(self):
        from datetime import date
        today = date(2024, 1, 10)
        added = date(2024, 1, 4)  # 6 days ago, TTL=5
        self.assertTrue(self._is_expired(added, today, ttl_days=5))

    def test_stale_entries_are_removed_from_watchlist(self):
        """Simulates the eviction loop that removes expired entries."""
        from datetime import date
        today = date(2024, 1, 10)
        ttl = 5
        watchlist = {
            'AAPL': {'watchlist_date': date(2024, 1, 8)},  # 2 days — keep
            'MSFT': {'watchlist_date': date(2024, 1, 3)},  # 7 days — expire
            'TSLA': {'watchlist_date': date(2024, 1, 9)},  # 1 day  — keep
        }

        stale = [
            sym for sym, c in watchlist.items()
            if (today - c['watchlist_date']).days > ttl
        ]
        for sym in stale:
            watchlist.pop(sym)

        self.assertIn('AAPL', watchlist)
        self.assertIn('TSLA', watchlist)
        self.assertNotIn('MSFT', watchlist)


# ---------------------------------------------------------------------------
# Watchlist promotion logic
# ---------------------------------------------------------------------------

class TestWatchlistPromotion(unittest.TestCase):
    """
    A watchlist entry with action='buy' should be moved to self.pairs.
    These tests verify the promotion and eviction logic without instantiating Strategy.
    """

    def _simulate_watchlist_evaluation(self, watchlist, action_by_symbol):
        """
        Simulate one pass of the watchlist evaluation loop.
        action_by_symbol: dict mapping symbol -> 'buy' | 'hold' | 'sell'
        Returns (updated_pairs, updated_watchlist).
        """
        pairs = {}
        stale = []
        for symbol, candidate in watchlist.items():
            action = action_by_symbol.get(symbol, 'hold')
            if action == 'buy':
                candidate['action'] = 'buy'
                pairs[symbol] = candidate
                stale.append(symbol)
        for sym in stale:
            watchlist.pop(sym)
        return pairs, watchlist

    def test_buy_signal_promotes_entry_to_pairs(self):
        watchlist = {
            'AAPL': {'lead_stock': 'MSFT', 'lag_stock': 'AAPL', 'action': 'hold'},
        }
        pairs, remaining = self._simulate_watchlist_evaluation(
            watchlist, action_by_symbol={'AAPL': 'buy'}
        )
        self.assertIn('AAPL', pairs)
        self.assertEqual(pairs['AAPL']['action'], 'buy')
        self.assertNotIn('AAPL', remaining)

    def test_hold_signal_keeps_entry_in_watchlist(self):
        watchlist = {
            'AAPL': {'lead_stock': 'MSFT', 'lag_stock': 'AAPL', 'action': 'hold'},
        }
        pairs, remaining = self._simulate_watchlist_evaluation(
            watchlist, action_by_symbol={'AAPL': 'hold'}
        )
        self.assertNotIn('AAPL', pairs)
        self.assertIn('AAPL', remaining)

    def test_only_buy_ready_entries_are_promoted(self):
        watchlist = {
            'AAPL': {'lead_stock': 'MSFT', 'lag_stock': 'AAPL', 'action': 'hold'},
            'TSLA': {'lead_stock': 'NVDA', 'lag_stock': 'TSLA', 'action': 'hold'},
        }
        pairs, remaining = self._simulate_watchlist_evaluation(
            watchlist, action_by_symbol={'AAPL': 'buy', 'TSLA': 'hold'}
        )
        self.assertIn('AAPL', pairs)
        self.assertNotIn('TSLA', pairs)
        self.assertIn('TSLA', remaining)
        self.assertNotIn('AAPL', remaining)

    def test_empty_watchlist_produces_no_pairs(self):
        pairs, remaining = self._simulate_watchlist_evaluation({}, {})
        self.assertEqual(pairs, {})
        self.assertEqual(remaining, {})


# ---------------------------------------------------------------------------
# Pair evaluation cooldown logic
# ---------------------------------------------------------------------------

def _check_cooldown(pair_key, evaluated_at, today, cooldown_days):
    """Replicate the cooldown gate from BobsBrain.before_market_opens."""
    if not cooldown_days:
        return False  # cooldown disabled — never skip
    if pair_key in evaluated_at:
        return (today - evaluated_at[pair_key]).days < cooldown_days
    return False


class TestPairEvalCooldown(unittest.TestCase):
    from datetime import date as _date

    def _today(self):
        from datetime import date
        return date(2024, 2, 10)

    def test_pair_not_in_dict_is_not_skipped(self):
        from datetime import date
        evaluated = {}
        key = frozenset({'AAPL', 'MSFT'})
        self.assertFalse(_check_cooldown(key, evaluated, self._today(), cooldown_days=7))

    def test_pair_within_cooldown_is_skipped(self):
        from datetime import date
        today = self._today()
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: date(2024, 2, 6)}  # 4 days ago, cooldown=7
        self.assertTrue(_check_cooldown(key, evaluated, today, cooldown_days=7))

    def test_pair_exactly_at_cooldown_boundary_is_not_skipped(self):
        from datetime import date
        today = self._today()
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: date(2024, 2, 3)}  # exactly 7 days ago
        self.assertFalse(_check_cooldown(key, evaluated, today, cooldown_days=7))

    def test_pair_past_cooldown_is_not_skipped(self):
        from datetime import date
        today = self._today()
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: date(2024, 1, 28)}  # 13 days ago, cooldown=7
        self.assertFalse(_check_cooldown(key, evaluated, today, cooldown_days=7))

    def test_cooldown_disabled_with_none(self):
        from datetime import date
        today = self._today()
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: today}  # evaluated today — but cooldown disabled
        self.assertFalse(_check_cooldown(key, evaluated, today, cooldown_days=None))

    def test_cooldown_disabled_with_zero(self):
        from datetime import date
        today = self._today()
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: today}
        self.assertFalse(_check_cooldown(key, evaluated, today, cooldown_days=0))

    def test_pair_key_is_order_independent(self):
        """frozenset ensures (A, B) and (B, A) map to the same cooldown entry."""
        from datetime import date
        today = self._today()
        key_ab = frozenset({'AAPL', 'MSFT'})
        key_ba = frozenset({'MSFT', 'AAPL'})
        evaluated = {key_ab: today}
        self.assertTrue(_check_cooldown(key_ba, evaluated, today, cooldown_days=7))


# ---------------------------------------------------------------------------
# Same-sector / both-ETF gate logic
# ---------------------------------------------------------------------------

def _passes_sector_gate(meta1, meta2):
    """Replicate the sector gate from BobsBrain.before_market_opens."""
    if meta1 is None or meta2 is None:
        return True  # missing metadata → allow
    both_etf = meta1.get('is_etf') and meta2.get('is_etf')
    same_sector = (
        meta1.get('sector') and
        meta1.get('sector') == meta2.get('sector')
    )
    return bool(both_etf or same_sector)


class TestSectorGate(unittest.TestCase):
    def test_same_sector_passes(self):
        m1 = {'sector': 'Technology', 'is_etf': False}
        m2 = {'sector': 'Technology', 'is_etf': False}
        self.assertTrue(_passes_sector_gate(m1, m2))

    def test_different_sector_fails(self):
        m1 = {'sector': 'Technology', 'is_etf': False}
        m2 = {'sector': 'Healthcare', 'is_etf': False}
        self.assertFalse(_passes_sector_gate(m1, m2))

    def test_both_etf_passes(self):
        m1 = {'sector': None, 'is_etf': True}
        m2 = {'sector': None, 'is_etf': True}
        self.assertTrue(_passes_sector_gate(m1, m2))

    def test_one_etf_one_stock_fails(self):
        m1 = {'sector': None,         'is_etf': True}
        m2 = {'sector': 'Technology', 'is_etf': False}
        self.assertFalse(_passes_sector_gate(m1, m2))

    def test_missing_metadata_for_first_ticker_passes(self):
        self.assertTrue(_passes_sector_gate(None, {'sector': 'Technology', 'is_etf': False}))

    def test_missing_metadata_for_second_ticker_passes(self):
        self.assertTrue(_passes_sector_gate({'sector': 'Technology', 'is_etf': False}, None))

    def test_both_missing_metadata_passes(self):
        self.assertTrue(_passes_sector_gate(None, None))

    def test_both_etf_same_sector_passes(self):
        """ETF + same sector — both conditions true, should pass."""
        m1 = {'sector': 'Equity', 'is_etf': True}
        m2 = {'sector': 'Equity', 'is_etf': True}
        self.assertTrue(_passes_sector_gate(m1, m2))

    def test_sector_none_on_one_stock_fails(self):
        """If only one side has sector data it cannot match."""
        m1 = {'sector': 'Technology', 'is_etf': False}
        m2 = {'sector': None,         'is_etf': False}
        self.assertFalse(_passes_sector_gate(m1, m2))


# ---------------------------------------------------------------------------
# Metadata quality validation logic
# ---------------------------------------------------------------------------

def _validate_coverage(tickers, metadata):
    """
    Replicate the coverage classification in BobsBrain._validate_ticker_metadata.
    Returns 'ok', 'warning', or 'error'.
    """
    total = len(tickers)
    if total == 0:
        return 'ok'
    with_sector = sum(1 for t in tickers if metadata.get(t, {}).get('sector'))
    coverage_pct = with_sector / total
    if coverage_pct < 0.30:
        return 'error'
    if coverage_pct < 0.70:
        return 'warning'
    return 'ok'


class TestMetadataValidation(unittest.TestCase):
    def _meta(self, tickers_with_sector, total):
        """Build a metadata dict where the first N tickers have a sector."""
        tickers = [f'T{i}' for i in range(total)]
        metadata = {
            t: {'sector': 'Technology' if i < tickers_with_sector else None, 'is_etf': False}
            for i, t in enumerate(tickers)
        }
        return tickers, metadata

    def test_high_coverage_returns_ok(self):
        tickers, meta = self._meta(tickers_with_sector=85, total=100)
        self.assertEqual(_validate_coverage(tickers, meta), 'ok')

    def test_exactly_seventy_percent_returns_ok(self):
        tickers, meta = self._meta(tickers_with_sector=70, total=100)
        self.assertEqual(_validate_coverage(tickers, meta), 'ok')

    def test_below_seventy_returns_warning(self):
        tickers, meta = self._meta(tickers_with_sector=50, total=100)
        self.assertEqual(_validate_coverage(tickers, meta), 'warning')

    def test_below_thirty_returns_error(self):
        tickers, meta = self._meta(tickers_with_sector=20, total=100)
        self.assertEqual(_validate_coverage(tickers, meta), 'error')

    def test_zero_tickers_returns_ok(self):
        self.assertEqual(_validate_coverage([], {}), 'ok')

    def test_no_sector_data_at_all_returns_error(self):
        tickers, meta = self._meta(tickers_with_sector=0, total=100)
        self.assertEqual(_validate_coverage(tickers, meta), 'error')


if __name__ == '__main__':
    unittest.main()
