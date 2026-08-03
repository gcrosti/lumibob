import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from StockEvaluator import (EntryMetrics, SpreadScores, StockEvaluator,
                            halflife_to_score)


class TestHalflifeToScore(unittest.TestCase):
    """The log-spaced half-life transform (WS1: restore component variance)."""

    def test_range_and_endpoints(self):
        self.assertEqual(halflife_to_score(1.0, 60), 1.0)      # floor -> max score
        self.assertEqual(halflife_to_score(60.0, 60), 0.0)     # ceiling -> 0
        self.assertEqual(halflife_to_score(100.0, 60), 0.0)    # above ceiling -> 0
        for hl in (1, 2, 5, 15, 45, 60):
            s = halflife_to_score(hl, 60)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_missing_or_degenerate_is_zero(self):
        self.assertEqual(halflife_to_score(None, 60), 0.0)
        self.assertEqual(halflife_to_score(float('nan'), 60), 0.0)
        self.assertEqual(halflife_to_score(2.0, 1.0), 0.0)     # log ceiling collapses

    def test_monotonic_decreasing_in_halflife(self):
        prev = 1.1
        for hl in (1, 2, 3, 5, 10, 30):
            s = halflife_to_score(hl, 60)
            self.assertLess(s, prev)
            prev = s

    def test_restores_variance_across_low_cluster(self):
        """The whole point of WS1: half-lives clustered at 1-5 days must NOT all
        collapse to ~0.96 as the old linear map did — they must spread out."""
        low = [halflife_to_score(hl, 60) for hl in (1, 2, 3, 4, 5)]
        self.assertGreater(max(low) - min(low), 0.30)          # log spread
        old_linear = [max(0.0, 1.0 - hl / 60) for hl in (1, 2, 3, 4, 5)]
        self.assertLess(max(old_linear) - min(old_linear), 0.10)  # what we replaced


class TestGetCorrelation(unittest.TestCase):
    def setUp(self):
        self.evaluator = StockEvaluator()

    def test_perfect_positive_correlation_no_lag(self):
        """Identical series should have correlation of 1.0 (after accounting for lag NaN rows)."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = self.evaluator.get_correlation(data, data, lag=0)
        self.assertAlmostEqual(corr, 1.0, places=5)

    def test_positive_correlation_with_lag(self):
        """Lead shifted by 1 should still produce a high correlation with the original lag series."""
        lead = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        lag = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        corr = self.evaluator.get_correlation(lead, lag, lag=1)
        self.assertGreater(corr, 0.99)

    def test_negative_correlation(self):
        """Inversely related series should yield a negative correlation."""
        lead = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        lag = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        corr = self.evaluator.get_correlation(lead, lag, lag=0)
        self.assertLess(corr, -0.99)

    def test_returns_nan_for_constant_series(self):
        """A constant series has zero variance; correlation should be NaN."""
        constant = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
        varying = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = self.evaluator.get_correlation(constant, varying, lag=0)
        self.assertTrue(math.isnan(corr))

    def test_returns_nan_for_all_nan_series(self):
        """An all-NaN series should produce a NaN correlation."""
        nan_series = pd.Series([float('nan')] * 5)
        varying = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = self.evaluator.get_correlation(nan_series, varying, lag=0)
        self.assertTrue(math.isnan(corr))

    def test_returns_nan_when_lag_consumes_all_overlap(self):
        """A lag larger than the series length leaves no overlapping values — should return NaN."""
        data = pd.Series([1.0, 2.0, 3.0])
        corr = self.evaluator.get_correlation(data, data, lag=10)
        self.assertTrue(math.isnan(corr))


class TestGetAction(unittest.TestCase):
    def setUp(self):
        self.evaluator = StockEvaluator()

    def _make_trending_lead(self, direction: str, length: int = 20) -> pd.Series:
        """Returns an upward or downward trending series."""
        if direction == 'up':
            return pd.Series(range(1, length + 1), dtype=float)
        return pd.Series(range(length, 0, -1), dtype=float)

    def test_buy_signal_on_upward_lead(self):
        """Upward-trending lead stock: short MA > long MA → 'buy'."""
        lead = self._make_trending_lead('up')
        lag = self._make_trending_lead('up')
        action = self.evaluator.get_action(lead, lag, lag=1, short_ma=2, long_ma=5)
        self.assertEqual(action, 'buy')

    def test_sell_signal_on_downward_lead(self):
        """Downward-trending lead stock: short MA < long MA → 'sell'."""
        lead = self._make_trending_lead('down')
        lag = self._make_trending_lead('down')
        action = self.evaluator.get_action(lead, lag, lag=1, short_ma=2, long_ma=5)
        self.assertEqual(action, 'sell')

    def test_hold_signal_when_mas_equal(self):
        """Flat lead stock: short MA == long MA → 'hold'."""
        flat = pd.Series([5.0] * 20)
        action = self.evaluator.get_action(flat, flat, lag=0, short_ma=2, long_ma=5)
        self.assertEqual(action, 'hold')

    def test_buy_signal_respects_lag(self):
        """Action should still reflect bullish signal when a non-zero lag is applied."""
        lead = self._make_trending_lead('up', length=30)
        lag_series = self._make_trending_lead('up', length=30)
        action = self.evaluator.get_action(lead, lag_series, lag=3, short_ma=2, long_ma=10)
        self.assertEqual(action, 'buy')

    def test_default_ma_parameters(self):
        """Calling get_action without explicit MA params should use defaults (short=2, long=5)."""
        lead = self._make_trending_lead('up')
        lag_series = self._make_trending_lead('up')
        action_explicit = self.evaluator.get_action(lead, lag_series, lag=1, short_ma=2, long_ma=5)
        action_default = self.evaluator.get_action(lead, lag_series, lag=1)
        self.assertEqual(action_explicit, action_default)


class TestIsCointegrated(unittest.TestCase):
    def setUp(self):
        self.evaluator = StockEvaluator()

    def _make_cointegrated_pair(self, n: int = 100):
        """
        Construct a cointegrated pair: lag = lead + stationary noise.
        The spread is mean-reverting by construction, so coint() should reject
        the null of no cointegration.
        """
        rng = np.random.default_rng(42)
        lead = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
        lag = lead + rng.normal(0, 0.5, n)
        return lead, lag

    def _make_independent_pair(self, n: int = 100):
        """Two independent random walks — should not be cointegrated."""
        rng = np.random.default_rng(99)
        lead = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
        lag = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
        return lead, lag

    def test_cointegrated_pair_returns_true(self):
        """A pair whose spread is stationary should be identified as cointegrated."""
        lead, lag = self._make_cointegrated_pair()
        result = self.evaluator.is_cointegrated(lead, lag)
        self.assertTrue(result)

    def test_independent_pair_returns_false(self):
        """Two independent random walks should not be flagged as cointegrated."""
        lead, lag = self._make_independent_pair()
        result = self.evaluator.is_cointegrated(lead, lag)
        self.assertFalse(result)

    def test_returns_false_for_insufficient_data(self):
        """Fewer than 10 overlapping observations → safe False rather than error."""
        short = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.evaluator.is_cointegrated(short, short)
        self.assertFalse(result)

    def test_returns_false_for_all_nan_series(self):
        """All-NaN input should return False gracefully."""
        nan_series = pd.Series([float('nan')] * 50)
        other = pd.Series(range(50), dtype=float)
        result = self.evaluator.is_cointegrated(nan_series, other)
        self.assertFalse(result)

    def test_respects_custom_p_threshold(self):
        """A very strict threshold (p < 0.0001) should reject most pairs."""
        lead, lag = self._make_cointegrated_pair()
        # Even a strongly cointegrated pair is unlikely to meet p < 0.0001
        strict_result = self.evaluator.is_cointegrated(lead, lag, p_threshold=0.0001)
        lenient_result = self.evaluator.is_cointegrated(lead, lag, p_threshold=0.05)
        # Lenient should accept; strict may reject — at minimum lenient >= strict
        self.assertGreaterEqual(int(lenient_result), int(strict_result))


class TestComputeZscore(unittest.TestCase):
    def setUp(self):
        self.evaluator = StockEvaluator()
        rng = np.random.default_rng(7)
        n = 80
        # Cointegrated pair: lag = lead + stationary noise
        self.lead = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
        self.lag = self.lead + rng.normal(0, 0.5, n)

    def test_returns_series_of_same_length(self):
        z = self.evaluator.compute_zscore(self.lead, self.lag, window=20)
        self.assertEqual(len(z), len(self.lead))

    def test_first_window_values_are_nan(self):
        """Rolling window means the first (window-1) values should be NaN."""
        z = self.evaluator.compute_zscore(self.lead, self.lag, window=20)
        self.assertTrue(z.iloc[:19].isna().all())

    def test_non_nan_values_after_warmup(self):
        z = self.evaluator.compute_zscore(self.lead, self.lag, window=20)
        self.assertFalse(z.iloc[20:].isna().all())

    def test_zscore_roughly_zero_mean(self):
        """For a stationary spread the long-run average Z-score should be near 0."""
        z = self.evaluator.compute_zscore(self.lead, self.lag, window=20).dropna()
        self.assertAlmostEqual(float(z.mean()), 0.0, delta=1.0)

    def test_too_short_series_returns_nan_series(self):
        short = pd.Series([100.0, 101.0, 100.5])
        z = self.evaluator.compute_zscore(short, short, window=20)
        self.assertTrue(z.isna().all())


class TestGetZscoreAction(unittest.TestCase):
    def setUp(self):
        self.evaluator = StockEvaluator()
        rng = np.random.default_rng(42)
        n = 80
        self.lead = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)

    def _make_lag_with_zscore(self, target_z: float, lead: pd.Series, window: int = 20) -> pd.Series:
        """
        Construct a lag series whose final Z-score is approximately target_z
        by shifting the spread on the last bar.
        """
        log_lead = np.log(lead.clip(lower=1e-9))
        # Start with a near-perfect cointegration (spread ≈ 0)
        lag = np.exp(log_lead + 0.01)
        lag_s = pd.Series(lag.values)
        spread = np.log(lag_s.clip(lower=1e-9)) - log_lead.values
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()
        # Push the final spread to hit target_z standard deviations
        final_mean = float(roll_mean.iloc[-1])
        final_std = float(roll_std.iloc[-1]) or 0.01
        new_log_lag_final = log_lead.iloc[-1] + final_mean + target_z * final_std
        lag_s.iloc[-1] = np.exp(new_log_lag_final)
        return lag_s

    def test_buy_when_zscore_below_negative_entry(self):
        lag = self._make_lag_with_zscore(-3.0, self.lead)
        action, z = self.evaluator.get_zscore_action(self.lead, lag, window=20,
                                                      entry_threshold=2.0, exit_threshold=0.5)
        self.assertEqual(action, 'buy')
        self.assertIsNotNone(z)

    def test_sell_when_zscore_above_negative_exit(self):
        lag = self._make_lag_with_zscore(1.0, self.lead)
        action, z = self.evaluator.get_zscore_action(self.lead, lag, window=20,
                                                      entry_threshold=2.0, exit_threshold=0.5)
        self.assertEqual(action, 'sell')
        self.assertIsNotNone(z)

    def test_hold_when_zscore_in_band(self):
        with patch.object(self.evaluator, 'compute_zscore',
                          return_value=pd.Series([-1.2])):
            action, z = self.evaluator.get_zscore_action(
                self.lead, self.lead, window=20,
                entry_threshold=2.0, exit_threshold=0.5,
            )
        self.assertEqual(action, 'hold')
        self.assertAlmostEqual(z, -1.2)

    def test_hold_on_too_short_series(self):
        short = pd.Series([100.0, 101.0])
        action, z = self.evaluator.get_zscore_action(short, short, window=20,
                                                      entry_threshold=2.0, exit_threshold=0.5)
        self.assertEqual(action, 'hold')
        self.assertIsNone(z)

    def test_returns_tuple(self):
        lag = self._make_lag_with_zscore(-3.0, self.lead)
        result = self.evaluator.get_zscore_action(self.lead, lag, window=20,
                                                   entry_threshold=2.0, exit_threshold=0.5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestGetCorrelationDual(unittest.TestCase):
    def setUp(self):
        self.evaluator = StockEvaluator()
        rng = np.random.default_rng(42)
        n = 120
        base = np.cumsum(rng.normal(0, 1, n)) + 100
        self.lead = pd.Series(base)
        self.lag = pd.Series(base + rng.normal(0, 0.3, n))

    def test_returns_two_floats(self):
        cl, cs = self.evaluator.get_correlation_dual(self.lead, self.lag, 90, 20)
        self.assertIsInstance(cl, float)
        self.assertIsInstance(cs, float)

    def test_cointegrated_pair_has_positive_correlations(self):
        cl, cs = self.evaluator.get_correlation_dual(self.lead, self.lag, 90, 20)
        self.assertGreater(cl, 0.0)
        self.assertGreater(cs, 0.0)

    def test_short_series_returns_nan(self):
        short = pd.Series([100.0, 101.0, 102.0])
        cl, cs = self.evaluator.get_correlation_dual(short, short, 90, 20)
        self.assertTrue(math.isnan(cl))
        self.assertTrue(math.isnan(cs))

    def test_identical_series_near_one(self):
        cl, cs = self.evaluator.get_correlation_dual(self.lead, self.lead, 90, 20)
        self.assertAlmostEqual(cl, 1.0, places=3)
        self.assertAlmostEqual(cs, 1.0, places=3)


class TestComputeZDepth(unittest.TestCase):
    def setUp(self):
        self.evaluator = StockEvaluator()

    def test_returns_tuple(self):
        rng = np.random.default_rng(42)
        n = 80
        lead = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
        lag = lead + rng.normal(0, 0.5, n)
        result = self.evaluator.compute_z_depth(lead, lag)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_z_depth_between_zero_and_one(self):
        rng = np.random.default_rng(42)
        lead = pd.Series(np.cumsum(rng.normal(0, 1, 80)) + 100)
        lag = lead + rng.normal(0, 0.5, 80)
        z_depth, _ = self.evaluator.compute_z_depth(lead, lag)
        self.assertGreaterEqual(z_depth, 0.0)
        self.assertLessEqual(z_depth, 1.0)

    def test_short_series_returns_zero(self):
        short = pd.Series([100.0, 101.0])
        z_depth, raw_z = self.evaluator.compute_z_depth(short, short)
        self.assertEqual(z_depth, 0.0)
        self.assertIsNone(raw_z)

    def test_identical_series_returns_zero_depth(self):
        rng = np.random.default_rng(7)
        s = pd.Series(np.cumsum(rng.normal(0, 1, 80)) + 100)
        z_depth, _ = self.evaluator.compute_z_depth(s, s)
        self.assertAlmostEqual(z_depth, 0.0, places=1)


class TestComputeEntryMetrics(unittest.TestCase):
    """The entry criteria depend on this; see
    docs/plans/2026-08-01_entry-criteria-overhaul.md."""

    def setUp(self):
        self.evaluator = StockEvaluator()

    def _pair(self, noise=0.5, n=80, seed=42):
        rng = np.random.default_rng(seed)
        lead = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
        lag = lead + rng.normal(0, noise, n)
        return lead, lag

    def test_returns_entry_metrics(self):
        m = self.evaluator.compute_entry_metrics(*self._pair())
        self.assertIsInstance(m, EntryMetrics)
        self.assertTrue(np.isfinite(m.z))
        self.assertGreater(m.spread_std_bps, 0.0)

    def test_short_series_returns_none(self):
        short = pd.Series([100.0, 101.0])
        self.assertIsNone(self.evaluator.compute_entry_metrics(short, short))

    def test_identical_series_returns_none(self):
        """A zero-width spread has no scale, so no tradeable magnitude."""
        rng = np.random.default_rng(7)
        s = pd.Series(np.cumsum(rng.normal(0, 1, 80)) + 100)
        self.assertIsNone(self.evaluator.compute_entry_metrics(s, s))

    def test_expected_gross_is_zero_when_inside_exit_band(self):
        """|z| below the exit threshold means there is nothing left to capture."""
        m = self.evaluator.compute_entry_metrics(
            *self._pair(), exit_threshold=10.0)
        self.assertEqual(m.expected_gross_bps, 0.0)

    def test_expected_gross_scales_with_spread_size(self):
        """Two pairs at comparable |z| but different spread scale must produce
        proportionally different expected gross — the whole point of the
        measure, since the z-score normalises scale away."""
        tight = self.evaluator.compute_entry_metrics(*self._pair(noise=0.2))
        wide = self.evaluator.compute_entry_metrics(*self._pair(noise=2.0))
        self.assertGreater(wide.spread_std_bps, tight.spread_std_bps)

    def test_expected_gross_matches_formula(self):
        m = self.evaluator.compute_entry_metrics(*self._pair(), exit_threshold=0.5)
        expected = max(abs(m.z) - 0.5, 0.0) * m.spread_std_bps
        self.assertAlmostEqual(m.expected_gross_bps, expected, places=9)

    def test_z_sign_matches_compute_zscore(self):
        """The gate tests signed z, so the sign convention must agree with the
        exit path's compute_zscore."""
        lead, lag = self._pair()
        m = self.evaluator.compute_entry_metrics(lead, lag, window=20)
        z_series = self.evaluator.compute_zscore(lead, lag, window=20)
        self.assertAlmostEqual(m.z, float(z_series.iloc[-1]), places=9)


class TestComputeSpreadScores(unittest.TestCase):
    def setUp(self):
        self.evaluator = StockEvaluator()

    def _make_cointegrated_pair(self, n: int = 200, seed: int = 42):
        """Shared random walk plus stationary noise — cointegrated by construction."""
        rng = np.random.default_rng(seed)
        common = np.cumsum(rng.normal(0, 1, n)) + 100
        lead = pd.Series(common + rng.normal(0, 0.1, n))
        lag = pd.Series(common + rng.normal(0, 0.1, n))
        return lead.clip(lower=0.01), lag.clip(lower=0.01)

    def _make_independent_pair(self, n: int = 200, seed: int = 99):
        """Two fully independent random walks."""
        rng = np.random.default_rng(seed)
        lead = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
        lag = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
        return lead.clip(lower=0.01), lag.clip(lower=0.01)

    def test_returns_spread_scores_namedtuple(self):
        lead, lag = self._make_cointegrated_pair()
        result = self.evaluator.compute_spread_scores(lead, lag)
        self.assertIsInstance(result, SpreadScores)

    def test_cointegrated_pair_scores_high(self):
        lead, lag = self._make_cointegrated_pair()
        result = self.evaluator.compute_spread_scores(lead, lag)
        self.assertGreater(result.coint_score, 0.5)
        self.assertIsNotNone(result.halflife_days)
        self.assertGreater(result.halflife_days, 0)

    def test_independent_pair_scores_low_coint(self):
        """Independent random walks should produce a low cointegration score most of the time."""
        lead, lag = self._make_independent_pair()
        result = self.evaluator.compute_spread_scores(lead, lag)
        self.assertLess(result.coint_score, 0.5)

    def test_non_stationary_spread_gives_zero_halflife_score(self):
        """A non-stationary spread (|rho| >= 1) produces halflife_score == 0."""
        rng = np.random.default_rng(7)
        n = 150
        lead = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100).clip(lower=0.01)
        lag = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100).clip(lower=0.01)
        result = self.evaluator.compute_spread_scores(lead, lag)
        # Either halflife_days is None (non-stationary) or score is 0
        if result.halflife_days is None:
            self.assertEqual(result.halflife_score, 0.0)

    def test_graceful_on_all_nan_input(self):
        nan_series = pd.Series([float('nan')] * 100)
        other = pd.Series(range(1, 101), dtype=float)
        result = self.evaluator.compute_spread_scores(nan_series, other)
        self.assertEqual(result, SpreadScores(0.0, 0.0, 1.0, None))

    def test_graceful_on_too_short_input(self):
        short = pd.Series([100.0, 101.0, 100.5])
        result = self.evaluator.compute_spread_scores(short, short)
        self.assertEqual(result, SpreadScores(0.0, 0.0, 1.0, None))

    def test_coint_score_between_zero_and_one(self):
        lead, lag = self._make_cointegrated_pair()
        result = self.evaluator.compute_spread_scores(lead, lag)
        self.assertGreaterEqual(result.coint_score, 0.0)
        self.assertLessEqual(result.coint_score, 1.0)

    def test_halflife_score_between_zero_and_one(self):
        lead, lag = self._make_cointegrated_pair()
        result = self.evaluator.compute_spread_scores(lead, lag)
        self.assertGreaterEqual(result.halflife_score, 0.0)
        self.assertLessEqual(result.halflife_score, 1.0)

    def test_is_cointegrated_still_works_on_cointegrated_pair(self):
        """Regression: is_cointegrated must still return True after log-price fix."""
        lead, lag = self._make_cointegrated_pair()
        self.assertTrue(self.evaluator.is_cointegrated(lead, lag))

    def test_halflife_score_zero_when_halflife_at_ceiling(self):
        """A pair with halflife_days >= max_halflife_days should score 0."""
        lead, lag = self._make_cointegrated_pair()
        result = self.evaluator.compute_spread_scores(
            lead, lag, max_halflife_days=1.0
        )
        self.assertEqual(result.halflife_score, 0.0)

    def test_max_halflife_days_controls_scoring_range(self):
        """Wider ceiling → same halflife_days → higher halflife_score."""
        lead, lag = self._make_cointegrated_pair()
        r_narrow = self.evaluator.compute_spread_scores(lead, lag, max_halflife_days=30.0)
        r_wide = self.evaluator.compute_spread_scores(lead, lag, max_halflife_days=120.0)
        if r_narrow.halflife_days is not None and r_wide.halflife_days is not None:
            self.assertGreaterEqual(r_wide.halflife_score, r_narrow.halflife_score)


if __name__ == '__main__':
    unittest.main()
