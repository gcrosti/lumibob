"""
Unit tests for BobsBrain helpers.

BobsBrain extends Lumibot's Strategy class, making full integration tests
heavy.  These tests cover the static/pure helpers that can be exercised
without instantiating the strategy or connecting to any external service.
"""

import sys
import os
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from BobsBrain import BobsBrain, _sic_to_sector


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
        failed: set[str] = set()
        symbol = 'NODATAINC'
        failed.add(symbol)
        self.assertIn(symbol, failed)


# ---------------------------------------------------------------------------
# Composite-score-based position sizing
# ---------------------------------------------------------------------------

class TestCompositeScoreSizing(unittest.TestCase):
    """
    Tests for composite-score-weighted position sizing.

    The budget formula interpolates between min_position_pct and
    max_position_pct based on the pair's composite_score, with a
    deployment-gap boost when the portfolio is under-deployed.
    """

    def _budget(self, score, portfolio_value, min_pct=0.03, max_pct=0.20,
                deployment_gap=0.0, n_candidates=1):
        base = (min_pct + score * (max_pct - min_pct)) * portfolio_value
        if deployment_gap > 0 and n_candidates > 0:
            gap_share = deployment_gap / n_candidates
            base = min(base + gap_share, max_pct * portfolio_value)
        return base

    def test_zero_score_gives_min_position(self):
        result = self._budget(0.0, 10_000)
        self.assertAlmostEqual(result, 300.0)

    def test_max_score_gives_max_position(self):
        result = self._budget(1.0, 10_000)
        self.assertAlmostEqual(result, 2_000.0)

    def test_medium_score_interpolates(self):
        result = self._budget(0.5, 10_000)
        self.assertAlmostEqual(result, 1_150.0)

    def test_deployment_gap_boosts_budget(self):
        result = self._budget(0.0, 10_000, deployment_gap=3_000, n_candidates=3)
        self.assertAlmostEqual(result, 1_300.0)

    def test_deployment_gap_boost_capped_at_max_position(self):
        result = self._budget(0.5, 10_000, deployment_gap=50_000, n_candidates=1)
        self.assertAlmostEqual(result, 2_000.0)

    def test_cash_guard_prevents_overspend(self):
        per_stock_budget = 1_500.0
        available_cash = 900.0
        self.assertFalse(available_cash >= per_stock_budget)

    def test_sufficient_cash_allows_buy(self):
        per_stock_budget = 1_500.0
        available_cash = 2_000.0
        self.assertTrue(available_cash >= per_stock_budget)

    def test_candidates_sorted_by_score_descending(self):
        pairs = [
            {'lag_stock': 'LOW', 'composite_score': 0.2},
            {'lag_stock': 'HIGH', 'composite_score': 0.9},
            {'lag_stock': 'MID', 'composite_score': 0.5},
        ]
        ranked = sorted(pairs, key=lambda p: p['composite_score'], reverse=True)
        self.assertEqual([p['lag_stock'] for p in ranked], ['HIGH', 'MID', 'LOW'])


# ---------------------------------------------------------------------------
# _composite_score logic (mirrors BobsBrain._composite_score, 5 components)
# ---------------------------------------------------------------------------

def _composite_score(corr_long, corr_short, z_depth,
                     coint_score=0.0, halflife_score=0.0,
                     w_long=0.3, w_short=0.5, w_z=0.2,
                     w_coint=0.0, w_halflife=0.0):
    cl = max(corr_long, 0.0) if not np.isnan(corr_long) else 0.0
    cs = max(corr_short, 0.0) if not np.isnan(corr_short) else 0.0
    return (
        w_long * min(cl, 1.0)
        + w_short * min(cs, 1.0)
        + w_z * z_depth
        + w_coint * coint_score
        + w_halflife * halflife_score
    )


class TestCompositeScore(unittest.TestCase):
    def test_all_zeros(self):
        self.assertAlmostEqual(_composite_score(0.0, 0.0, 0.0), 0.0)

    def test_all_ones(self):
        self.assertAlmostEqual(_composite_score(1.0, 1.0, 1.0), 1.0)

    def test_corr_short_dominates(self):
        score = _composite_score(0.0, 1.0, 0.0)
        self.assertAlmostEqual(score, 0.5)

    def test_negative_correlations_clamped_to_zero(self):
        score = _composite_score(-0.5, -0.3, 0.0)
        self.assertAlmostEqual(score, 0.0)

    def test_nan_correlations_treated_as_zero(self):
        score = _composite_score(float('nan'), float('nan'), 0.5)
        self.assertAlmostEqual(score, 0.1)

    def test_corr_above_one_clamped(self):
        score = _composite_score(1.5, 1.5, 1.0)
        self.assertAlmostEqual(score, 1.0)

    def test_z_depth_only(self):
        score = _composite_score(0.0, 0.0, 1.0)
        self.assertAlmostEqual(score, 0.2)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(0.3 + 0.5 + 0.2, 1.0)

    def test_coint_score_raises_composite(self):
        """A perfect coint_score should raise composite above the 3-weight baseline."""
        base = _composite_score(0.5, 0.5, 0.5, coint_score=0.0, w_coint=0.0)
        with_coint = _composite_score(0.5, 0.5, 0.5, coint_score=1.0, w_coint=0.25)
        self.assertGreater(with_coint, base)

    def test_halflife_score_raises_composite(self):
        """A perfect halflife_score should raise composite above the 3-weight baseline."""
        base = _composite_score(0.5, 0.5, 0.5, halflife_score=0.0, w_halflife=0.0)
        with_hl = _composite_score(0.5, 0.5, 0.5, halflife_score=1.0, w_halflife=0.15)
        self.assertGreater(with_hl, base)

    def test_zero_coint_and_halflife_unchanged(self):
        """When both new scores are 0, result equals the 3-component formula."""
        old = _composite_score(0.6, 0.7, 0.8)
        new = _composite_score(0.6, 0.7, 0.8, coint_score=0.0, halflife_score=0.0,
                               w_coint=0.25, w_halflife=0.15)
        self.assertAlmostEqual(old, new)


# ---------------------------------------------------------------------------
# Composite score weight normalization helpers
# ---------------------------------------------------------------------------

class TestWeightNormalization(unittest.TestCase):
    """Verify that normalize_weights handles the expanded 5-weight set."""

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from tuning.parameter_space import normalize_weights
        self.normalize_weights = normalize_weights

    def test_five_weights_sum_to_one_after_normalisation(self):
        params = {
            'w_corr_long': 0.3,
            'w_corr_short': 0.5,
            'w_z_depth': 0.2,
            'w_coint': 0.25,
            'w_halflife': 0.15,
        }
        result = self.normalize_weights(params)
        total = sum(result[k] for k in params)
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_old_params_without_coint_still_normalise(self):
        """Old param dicts missing w_coint/w_halflife should be normalised correctly."""
        old_params = {'w_corr_long': 0.3, 'w_corr_short': 0.5, 'w_z_depth': 0.2}
        result = self.normalize_weights(old_params)
        # Result should contain only old keys normalised (new defaults absorbed in denominator)
        present = {k for k in result if k.startswith('w_')}
        self.assertTrue(present.issubset({'w_corr_long', 'w_corr_short', 'w_z_depth'}))
        # Values must be positive and < 1
        for k in old_params:
            self.assertGreater(result[k], 0.0)
            self.assertLessEqual(result[k], 1.0)


# ---------------------------------------------------------------------------
# Same-sector / both-ETF gate logic
# ---------------------------------------------------------------------------

def _passes_sector_gate(meta1, meta2):
    """Replicate the sector gate from BobsBrain.before_market_opens."""
    if meta1 is None or meta2 is None:
        return True  # missing metadata -> allow
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
        m1 = {'sector': 'Equity', 'is_etf': True}
        m2 = {'sector': 'Equity', 'is_etf': True}
        self.assertTrue(_passes_sector_gate(m1, m2))

    def test_sector_none_on_one_stock_fails(self):
        m1 = {'sector': 'Technology', 'is_etf': False}
        m2 = {'sector': None,         'is_etf': False}
        self.assertFalse(_passes_sector_gate(m1, m2))


# ---------------------------------------------------------------------------
# SIC-to-sector mapping
# ---------------------------------------------------------------------------

class TestSicToSector(unittest.TestCase):
    def test_manufacturing_sic(self):
        self.assertEqual(_sic_to_sector(3674), 'Manufacturing')

    def test_finance_sic(self):
        self.assertEqual(_sic_to_sector(6020), 'Finance, Insurance & Real Estate')

    def test_services_sic(self):
        self.assertEqual(_sic_to_sector(7372), 'Services')

    def test_none_returns_none(self):
        self.assertIsNone(_sic_to_sector(None))

    def test_invalid_string_returns_none(self):
        self.assertIsNone(_sic_to_sector('not_a_number'))

    def test_unmapped_sic_returns_other(self):
        self.assertEqual(_sic_to_sector(9999), 'Other')

    def test_string_sic_converted(self):
        self.assertEqual(_sic_to_sector('3674'), 'Manufacturing')


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


# ---------------------------------------------------------------------------
# Dynamic K determination
# ---------------------------------------------------------------------------

class TestDynamicK(unittest.TestCase):
    """
    K (target number of positions) is determined by:
        k_base = available_cash / (target_pos_pct * portfolio_value)
        quality_scale = clamp(pool_corr / 0.7, 0.5, 1.5)
        k_target = max(1, round(k_base * quality_scale))
    """

    def _compute_k(self, available_cash, portfolio_value,
                   min_pct=0.03, max_pct=0.20, pool_corr=0.0, n_existing=0):
        target_pos_pct = (min_pct + max_pct) / 2
        k_base = max(1, int(available_cash / (target_pos_pct * portfolio_value))) if portfolio_value > 0 else 1
        quality_scale = max(0.5, min(pool_corr / 0.7, 1.5))
        k_target = max(1, round(k_base * quality_scale))
        return max(k_target, n_existing)

    def test_full_cash_reasonable_k(self):
        k = self._compute_k(10_000, 10_000, pool_corr=0.7)
        self.assertGreaterEqual(k, 1)
        self.assertLessEqual(k, 100)

    def test_zero_pool_corr_halves_k(self):
        k_high = self._compute_k(10_000, 10_000, pool_corr=0.7)
        k_low = self._compute_k(10_000, 10_000, pool_corr=0.0)
        self.assertLessEqual(k_low, k_high)

    def test_k_never_below_one(self):
        k = self._compute_k(0, 10_000, pool_corr=0.0)
        self.assertGreaterEqual(k, 1)

    def test_k_never_below_existing_positions(self):
        k = self._compute_k(1_000, 10_000, pool_corr=0.0, n_existing=5)
        self.assertGreaterEqual(k, 5)

    def test_high_quality_scales_up(self):
        k_mid = self._compute_k(10_000, 10_000, pool_corr=0.7)
        k_high = self._compute_k(10_000, 10_000, pool_corr=1.05)
        self.assertGreaterEqual(k_high, k_mid)


# ---------------------------------------------------------------------------
# Pair evaluation cooldown logic
# ---------------------------------------------------------------------------

def _check_cooldown(pair_key, evaluated_at, today, cooldown_days):
    """Replicate the cooldown gate from BobsBrain.before_market_opens."""
    if not cooldown_days:
        return False
    if pair_key in evaluated_at:
        last = evaluated_at[pair_key]
        last_date = last.date() if hasattr(last, 'date') else last
        today_date = today.date() if hasattr(today, 'date') else today
        return (today_date - last_date).days < cooldown_days
    return False


class TestPairEvalCooldown(unittest.TestCase):
    from datetime import date as _date

    def _today(self):
        from datetime import date
        return date(2024, 2, 10)

    def test_pair_not_in_dict_is_not_skipped(self):
        key = frozenset({'AAPL', 'MSFT'})
        self.assertFalse(_check_cooldown(key, {}, self._today(), cooldown_days=7))

    def test_pair_within_cooldown_is_skipped(self):
        from datetime import date
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: date(2024, 2, 6)}
        self.assertTrue(_check_cooldown(key, evaluated, self._today(), cooldown_days=7))

    def test_pair_exactly_at_cooldown_boundary_is_not_skipped(self):
        from datetime import date
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: date(2024, 2, 3)}
        self.assertFalse(_check_cooldown(key, evaluated, self._today(), cooldown_days=7))

    def test_pair_past_cooldown_is_not_skipped(self):
        from datetime import date
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: date(2024, 1, 28)}
        self.assertFalse(_check_cooldown(key, evaluated, self._today(), cooldown_days=7))

    def test_cooldown_disabled_with_none(self):
        from datetime import date
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: self._today()}
        self.assertFalse(_check_cooldown(key, evaluated, self._today(), cooldown_days=None))

    def test_cooldown_disabled_with_zero(self):
        key = frozenset({'AAPL', 'MSFT'})
        evaluated = {key: self._today()}
        self.assertFalse(_check_cooldown(key, evaluated, self._today(), cooldown_days=0))

    def test_pair_key_is_order_independent(self):
        from datetime import date
        key_ab = frozenset({'AAPL', 'MSFT'})
        key_ba = frozenset({'MSFT', 'AAPL'})
        evaluated = {key_ab: self._today()}
        self.assertTrue(_check_cooldown(key_ba, evaluated, self._today(), cooldown_days=7))


# ---------------------------------------------------------------------------
# Daily candidate budget
# ---------------------------------------------------------------------------

class TestDailyBudget(unittest.TestCase):
    def test_budget_limits_scored_candidates(self):
        """Round-robin loop should stop once daily budget is exhausted."""
        budget = 5
        scored = 0
        pairs = [(f'A{i}', f'B{i}') for i in range(20)]
        for a, b in pairs:
            if scored >= budget:
                break
            scored += 1
        self.assertEqual(scored, budget)

    def test_budget_zero_scores_nothing(self):
        budget = 0
        scored = 0
        for _ in range(10):
            if scored >= budget:
                break
            scored += 1
        self.assertEqual(scored, 0)


if __name__ == '__main__':
    unittest.main()
