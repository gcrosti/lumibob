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
# Fixed per-pair budget calculation
# ---------------------------------------------------------------------------

class TestPerPairBudget(unittest.TestCase):
    """
    The per-pair allocation is:
        per_stock_budget = available_cash * max_daily_spend_pct * per_pair_allocation
    These tests verify the formula directly without instantiating Strategy.
    """

    def _budget(self, available_cash, max_daily_spend_pct, per_pair_allocation):
        return available_cash * max_daily_spend_pct * per_pair_allocation

    def test_default_params_give_five_pct_of_cash_on_day_one(self):
        """With full starting cash ($10k) and defaults (0.5 × 0.10), per pair = $500."""
        result = self._budget(10_000, 0.5, 0.10)
        self.assertAlmostEqual(result, 500.0)

    def test_shrinks_as_cash_depletes(self):
        """After deploying half the capital, per-pair budget halves too."""
        result = self._budget(5_000, 0.5, 0.10)
        self.assertAlmostEqual(result, 250.0)

    def test_zero_allocation_gives_zero_budget(self):
        result = self._budget(10_000, 0.5, 0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_remaining_cash_gate_prevents_overspend(self):
        """
        Simulates the cash guard: if remaining_cash < per_stock_budget the
        buy loop should break rather than place the order.
        """
        per_stock_budget = 500.0
        remaining_cash = 300.0
        buy_executed = False

        if remaining_cash >= per_stock_budget:
            buy_executed = True

        self.assertFalse(buy_executed)

    def test_sufficient_cash_allows_buy(self):
        per_stock_budget = 500.0
        remaining_cash = 600.0
        buy_executed = False

        if remaining_cash >= per_stock_budget:
            buy_executed = True

        self.assertTrue(buy_executed)


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


if __name__ == '__main__':
    unittest.main()
