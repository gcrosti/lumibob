"""
Unit tests for PairSimulator.

All tests use deterministic synthetic price series so they are fast and
reproducible without any external data or DB dependencies.
"""

import math
import sys
import os
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PairSimulator import PairSimulator, SimResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trending(n: int = 60, start: float = 100.0, slope: float = 1.0) -> pd.Series:
    """Monotonically trending series with a fixed slope."""
    return pd.Series([start + i * slope for i in range(n)], dtype=float)


def _flat(n: int = 60, value: float = 50.0) -> pd.Series:
    return pd.Series([value] * n, dtype=float)


def _oscillating(n: int = 60, base: float = 100.0, amplitude: float = 5.0) -> pd.Series:
    """Series that alternates up and down, generating buy/sell crossovers."""
    rng = np.random.default_rng(7)
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] + rng.choice([-amplitude, amplitude]))
    return pd.Series(prices, dtype=float)


# ---------------------------------------------------------------------------
# SimResult dataclass
# ---------------------------------------------------------------------------

class TestSimResult(unittest.TestCase):
    def test_is_dataclass_with_expected_fields(self):
        r = SimResult(
            total_return=0.05, sharpe=1.2, max_drawdown=-0.02,
            win_rate=0.6, num_trades=5, avg_holding_days=4.0,
            lag=1, short_ma=2, long_ma=5,
        )
        self.assertAlmostEqual(r.total_return, 0.05)
        self.assertEqual(r.lag, 1)
        self.assertEqual(r.short_ma, 2)
        self.assertEqual(r.long_ma, 5)


# ---------------------------------------------------------------------------
# PairSimulator.run()
# ---------------------------------------------------------------------------

class TestPairSimulatorRun(unittest.TestCase):
    def setUp(self):
        self.sim = PairSimulator()

    def test_raises_when_short_ma_not_less_than_long_ma(self):
        lead = _trending()
        lag = _trending()
        with self.assertRaises(ValueError):
            self.sim.run(lead, lag, lag=1, short_ma=5, long_ma=5)

    def test_returns_sim_result_instance(self):
        lead = _trending()
        lag = _trending()
        result = self.sim.run(lead, lag, lag=1, short_ma=2, long_ma=5)
        self.assertIsInstance(result, SimResult)

    def test_stores_params_on_result(self):
        lead = _trending()
        lag = _trending()
        result = self.sim.run(lead, lag, lag=2, short_ma=1, long_ma=3)
        self.assertEqual(result.lag, 2)
        self.assertEqual(result.short_ma, 1)
        self.assertEqual(result.long_ma, 3)

    def test_uptrending_lead_produces_positive_return(self):
        """
        A steadily uptrending lead creates a permanent buy signal; the lag
        stock also trends up, so the strategy should produce a positive return.
        """
        lead = _trending(n=60, slope=1.0)
        lag = _trending(n=60, slope=0.8)
        result = self.sim.run(lead, lag, lag=1, short_ma=2, long_ma=5)
        self.assertGreater(result.total_return, 0)

    def test_flat_lead_produces_zero_trades(self):
        """A completely flat lead never crosses MAs — no trades executed."""
        lead = _flat()
        lag = _trending()
        result = self.sim.run(lead, lag, lag=1, short_ma=2, long_ma=5)
        self.assertEqual(result.num_trades, 0)

    def test_max_drawdown_is_non_positive(self):
        lead = _oscillating()
        lag = _oscillating()
        result = self.sim.run(lead, lag, lag=1, short_ma=2, long_ma=5)
        self.assertLessEqual(result.max_drawdown, 0.0)

    def test_win_rate_is_nan_when_no_trades(self):
        lead = _flat()
        lag = _flat()
        result = self.sim.run(lead, lag, lag=1, short_ma=2, long_ma=5)
        self.assertEqual(result.num_trades, 0)
        self.assertTrue(math.isnan(result.win_rate))

    def test_win_rate_between_zero_and_one(self):
        lead = _oscillating()
        lag = _oscillating()
        result = self.sim.run(lead, lag, lag=1, short_ma=2, long_ma=5)
        if result.num_trades > 0:
            self.assertGreaterEqual(result.win_rate, 0.0)
            self.assertLessEqual(result.win_rate, 1.0)

    def test_avg_holding_days_is_positive_when_trades_exist(self):
        lead = _oscillating()
        lag = _oscillating()
        result = self.sim.run(lead, lag, lag=1, short_ma=2, long_ma=5)
        if result.num_trades > 0:
            self.assertGreater(result.avg_holding_days, 0)

    def test_sharpe_is_nan_for_zero_variance_strategy(self):
        """When the strategy never changes (e.g. always in cash), std = 0 → nan Sharpe."""
        lead = _flat()
        lag = _flat()
        result = self.sim.run(lead, lag, lag=1, short_ma=2, long_ma=5)
        self.assertTrue(math.isnan(result.sharpe))


# ---------------------------------------------------------------------------
# PairSimulator.optimize()
# ---------------------------------------------------------------------------

class TestPairSimulatorOptimize(unittest.TestCase):
    def setUp(self):
        self.sim = PairSimulator()

    def test_returns_sim_result(self):
        lead = _trending()
        lag = _trending()
        result = self.sim.optimize(lead, lag, max_lag=3)
        self.assertIsInstance(result, SimResult)

    def test_lag_within_bounds(self):
        lead = _trending()
        lag = _trending()
        result = self.sim.optimize(lead, lag, max_lag=3)
        self.assertGreaterEqual(result.lag, 1)
        self.assertLessEqual(result.lag, 3)

    def test_ma_params_are_valid(self):
        lead = _trending()
        lag = _trending()
        result = self.sim.optimize(lead, lag, max_lag=5)
        self.assertGreaterEqual(result.short_ma, 1)
        self.assertLess(result.short_ma, result.long_ma)
        self.assertLessEqual(result.long_ma, 5)

    def test_result_is_best_among_all_combos(self):
        """optimize() should return at least as good a return as any single run."""
        lead = _oscillating()
        lag = _oscillating()
        best = self.sim.optimize(lead, lag, max_lag=3)

        for lag_val in range(1, 4):
            for short_ma in range(1, 5):
                for long_ma in range(short_ma + 1, 6):
                    r = self.sim.run(lead, lag, lag=lag_val, short_ma=short_ma, long_ma=long_ma)
                    self.assertGreaterEqual(
                        best.total_return, r.total_return - 1e-9,
                        msg=f"optimize() missed a better combo: lag={lag_val} {short_ma}/{long_ma}"
                    )

    def test_max_lag_one_still_returns_result(self):
        lead = _trending()
        lag = _trending()
        result = self.sim.optimize(lead, lag, max_lag=1)
        self.assertEqual(result.lag, 1)


if __name__ == '__main__':
    unittest.main()
