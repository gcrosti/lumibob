import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from StockEvaluator import StockEvaluator


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


if __name__ == '__main__':
    unittest.main()
