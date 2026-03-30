"""
Unit tests for TickerClusterer.

DatabaseClient is fully mocked — no live DB or API calls required.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TickerClusterer import TickerClusterer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AS_OF = datetime(2025, 6, 1)
TICKERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


def _make_prices(symbols: list[str], n_days: int = 80, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic price DataFrame with two distinct movement groups
    so HDBSCAN has a clear clustering signal to work with.

    Symbols in the first half move together (correlated); symbols in the
    second half form a second correlated group with a different drift.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=AS_OF, periods=n_days, freq='B')
    mid = len(symbols) // 2

    data = {}
    for i, sym in enumerate(symbols):
        if i < mid:
            # Group 1: mild upward drift
            noise = rng.normal(0.001, 0.01, n_days)
        else:
            # Group 2: mild downward drift with different variance
            noise = rng.normal(-0.001, 0.015, n_days)
        prices = 100 * np.exp(np.cumsum(noise))
        data[sym] = prices

    return pd.DataFrame(data, index=dates)


def _make_clusterer(prices: pd.DataFrame) -> tuple[TickerClusterer, MagicMock]:
    mock_db = MagicMock()
    mock_db.get_prices.return_value = prices
    clusterer = TickerClusterer(db=mock_db, min_cluster_size=2)
    return clusterer, mock_db


# ---------------------------------------------------------------------------
# Basic clustering behaviour
# ---------------------------------------------------------------------------

class TestGetClusters:
    def test_returns_list_of_lists(self):
        prices = _make_prices(TICKERS)
        clusterer, _ = _make_clusterer(prices)
        clusters = clusterer.get_clusters(TICKERS, as_of=AS_OF)
        assert isinstance(clusters, list)
        assert all(isinstance(c, list) for c in clusters)

    def test_all_tickers_present_in_some_cluster(self):
        prices = _make_prices(TICKERS)
        clusterer, _ = _make_clusterer(prices)
        clusters = clusterer.get_clusters(TICKERS, as_of=AS_OF)
        found = {sym for cluster in clusters for sym in cluster}
        assert found == set(TICKERS)

    def test_no_ticker_appears_twice(self):
        prices = _make_prices(TICKERS)
        clusterer, _ = _make_clusterer(prices)
        clusters = clusterer.get_clusters(TICKERS, as_of=AS_OF)
        all_tickers = [sym for cluster in clusters for sym in cluster]
        assert len(all_tickers) == len(set(all_tickers))

    def test_clusters_ranked_by_yield_descending(self):
        """
        Clusters are sorted by avg_corr × size descending.

        HDBSCAN is mocked to assign deterministic labels so the test never
        skips. Labels are deliberately returned in reverse-yield order to
        verify that _compute sorts them correctly before returning.
        """
        from unittest.mock import patch

        prices = _make_prices(TICKERS)  # group1=TICKERS[:5], group2=TICKERS[5:]
        clusterer, mock_db = _make_clusterer(prices)

        # HDBSCAN labels: group1 = label 1, group2 = label 0 (insertion order
        # means label 0 would be returned first by cluster_map.values() if not
        # sorted — we rely on this to verify the yield-based sort actually fires).
        mock_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        mock_hdbscan = MagicMock()
        mock_hdbscan.fit_predict.return_value = mock_labels

        with patch('TickerClusterer.hdbscan.HDBSCAN', return_value=mock_hdbscan):
            clusters = clusterer.get_clusters(TICKERS, as_of=AS_OF)

        assert len(clusters) == 2

        # Verify the output is ordered by yield (avg_corr × size) descending.
        log_ret = np.log(prices).diff().dropna()

        def _yield(members):
            sub = log_ret[members].corr().values
            n = len(members)
            return (sub.sum() - n) / (n * (n - 1)) * n

        yields = [_yield(c) for c in clusters]
        assert yields[0] >= yields[1]


# ---------------------------------------------------------------------------
# Fallback behaviour on bad / missing data
# ---------------------------------------------------------------------------

class TestFallbacks:
    def test_empty_prices_returns_single_cluster_with_all_tickers(self):
        mock_db = MagicMock()
        mock_db.get_prices.return_value = pd.DataFrame()
        clusterer = TickerClusterer(db=mock_db, min_cluster_size=2)
        clusters = clusterer.get_clusters(TICKERS, as_of=AS_OF)
        assert len(clusters) == 1
        assert set(clusters[0]) == set(TICKERS)

    def test_single_ticker_returns_single_cluster(self):
        prices = _make_prices(['A'])
        clusterer, _ = _make_clusterer(prices)
        clusters = clusterer.get_clusters(['A'], as_of=AS_OF)
        assert len(clusters) == 1
        assert clusters[0] == ['A']

    def test_sparse_tickers_dropped_before_clustering(self):
        """Tickers with fewer than 50% non-null observations are excluded."""
        prices = _make_prices(TICKERS[:4])
        # Add a very sparse ticker — only 1 non-null observation
        sparse_col = pd.Series(np.nan, index=prices.index)
        sparse_col.iloc[-1] = 10.0
        prices = prices.copy()
        prices['SPARSE'] = sparse_col

        mock_db = MagicMock()
        mock_db.get_prices.return_value = prices
        clusterer = TickerClusterer(db=mock_db, min_cluster_size=2)
        clusters = clusterer.get_clusters(TICKERS[:4] + ['SPARSE'], as_of=AS_OF)

        all_tickers = {sym for cluster in clusters for sym in cluster}
        assert 'SPARSE' not in all_tickers


# ---------------------------------------------------------------------------
# Caching / recompute logic
# ---------------------------------------------------------------------------

class TestCaching:
    def test_result_is_cached_on_second_call_same_universe(self):
        prices = _make_prices(TICKERS)
        clusterer, mock_db = _make_clusterer(prices)

        clusterer.get_clusters(TICKERS, as_of=AS_OF)
        clusterer.get_clusters(TICKERS, as_of=AS_OF)

        # DB queried exactly once across both calls.
        assert mock_db.get_prices.call_count == 1

    def test_recomputes_when_ticker_universe_changes(self):
        prices = _make_prices(TICKERS)
        clusterer, mock_db = _make_clusterer(prices)

        clusterer.get_clusters(TICKERS, as_of=AS_OF)
        clusterer.get_clusters(TICKERS[:-1], as_of=AS_OF)  # different universe

        assert mock_db.get_prices.call_count == 2

    def test_recomputes_after_recompute_days_elapsed(self):
        prices = _make_prices(TICKERS)
        clusterer, mock_db = _make_clusterer(prices)

        clusterer.get_clusters(TICKERS, as_of=AS_OF, recompute_days=30)
        later = AS_OF + timedelta(days=31)
        clusterer.get_clusters(TICKERS, as_of=later, recompute_days=30)

        assert mock_db.get_prices.call_count == 2

    def test_does_not_recompute_before_recompute_days(self):
        prices = _make_prices(TICKERS)
        clusterer, mock_db = _make_clusterer(prices)

        clusterer.get_clusters(TICKERS, as_of=AS_OF, recompute_days=30)
        soon = AS_OF + timedelta(days=10)
        clusterer.get_clusters(TICKERS, as_of=soon, recompute_days=30)

        assert mock_db.get_prices.call_count == 1

    def test_recompute_days_none_never_recomputes(self):
        prices = _make_prices(TICKERS)
        clusterer, mock_db = _make_clusterer(prices)

        clusterer.get_clusters(TICKERS, as_of=AS_OF, recompute_days=None)
        # Even a year later, no recomputation when recompute_days=None.
        much_later = AS_OF + timedelta(days=365)
        clusterer.get_clusters(TICKERS, as_of=much_later, recompute_days=None)

        assert mock_db.get_prices.call_count == 1


# ---------------------------------------------------------------------------
# Look-ahead safety
# ---------------------------------------------------------------------------

class TestLookaheadSafety:
    def test_prices_fetched_with_as_of_as_end_date(self):
        prices = _make_prices(TICKERS)
        clusterer, mock_db = _make_clusterer(prices)

        clusterer.get_clusters(TICKERS, as_of=AS_OF)

        _, call_kwargs = mock_db.get_prices.call_args
        # end date passed to DB must equal as_of
        assert call_kwargs.get('end') == AS_OF or mock_db.get_prices.call_args[0][2] == AS_OF
