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


if __name__ == '__main__':
    unittest.main()
