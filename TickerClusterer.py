"""
TickerClusterer — groups tickers by 6-month return similarity for efficient pair discovery.

Reduces the O(N²) pair search space by only evaluating pairs within clusters of
similarly-moving tickers. Clusters are ranked by expected pair yield
(avg_intra_cluster_corr × cluster_size) so BobsBrain searches the most fertile
clusters first.

Data is read entirely from the local stock_prices table — no API calls are made.
"""

import math
from datetime import datetime, timedelta

import hdbscan
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from DatabaseClient import DatabaseClient


class TickerClusterer:
    """
    Clusters tickers by movement similarity and ranks clusters by expected
    pair-discovery yield.

    Usage:
        clusterer = TickerClusterer(db)
        clusters = clusterer.get_clusters(tickers, as_of=datetime.now())
        # clusters is a List[List[str]] sorted by expected yield, highest first

    Each call to get_clusters() returns a cached result unless the ticker
    universe has changed or recompute_days have elapsed since the last run.
    Pass recompute_days=None (the default) to compute once and hold — the
    recommended setting for backtests where recomputation overhead matters.
    """

    def __init__(
        self,
        db: DatabaseClient,
        lookback_days: int = 126,
        min_cluster_size: int = 5,
        pca_variance: float = 0.95,
    ):
        """
        Parameters
        ----------
        db : DatabaseClient
            Source for historical close prices.
        lookback_days : int
            Calendar days of price history used to build clusters (~6 trading months).
        min_cluster_size : int
            Minimum tickers to form a cluster (HDBSCAN parameter).
            Smaller values produce more, tighter clusters.
        pca_variance : float
            Fraction of variance to retain after PCA dimensionality reduction.
        """
        self._db = db
        self.lookback_days = lookback_days
        self.min_cluster_size = min_cluster_size
        self.pca_variance = pca_variance

        self._clusters: list[list[str]] = []
        self._last_computed: datetime | None = None
        self._last_tickers: set[str] = set()
        self._corr_matrix: pd.DataFrame | None = None
        self._symbols: list[str] = []

    def get_clusters(
        self,
        tickers: list[str],
        as_of: datetime,
        recompute_days: int | None = None,
    ) -> list[list[str]]:
        """
        Return tickers grouped into movement-similarity clusters, ranked by
        expected pair yield (avg intra-cluster correlation × cluster size).

        Clusters are recomputed when:
          - No prior result exists, or
          - The ticker universe has changed, or
          - recompute_days is not None and that many days have elapsed.

        Parameters
        ----------
        tickers : list[str]
            Full ticker universe to cluster (failed/penny tickers already excluded).
        as_of : datetime
            Upper bound for price data — prevents look-ahead bias in backtests.
        recompute_days : int | None
            Minimum days between recomputations. None = compute once, cache forever.

        Returns
        -------
        list[list[str]]
            Clusters sorted descending by expected yield. The final cluster, when
            present, contains tickers HDBSCAN could not assign to any coherent group.
        """
        ticker_set = set(tickers)

        def _to_naive(dt: datetime) -> datetime:
            return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

        elapsed = (
            (_to_naive(as_of) - _to_naive(self._last_computed)).days
            if self._last_computed is not None
            else math.inf
        )
        needs_recompute = (
            not self._clusters
            or ticker_set != self._last_tickers
            or (recompute_days is not None and elapsed >= recompute_days)
        )

        if needs_recompute:
            self._clusters = self._compute(tickers, as_of)
            self._last_computed = as_of
            self._last_tickers = ticker_set

        return self._clusters

    @property
    def corr_matrix(self) -> pd.DataFrame | None:
        """
        The log-return correlation matrix from the most recent clustering run.
        Indexed and columned by ticker symbol.  Returns None before first compute.
        """
        return self._corr_matrix

    def get_top_pairs_by_corr(
        self,
        cluster: list[str],
        n: int = 500,
    ) -> list[tuple[str, str, float]]:
        """
        Return the top-n within-cluster pairs ranked by correlation from the
        cached matrix.  Falls back to all pairs when n exceeds availability.

        Returns list of (symbol_a, symbol_b, correlation) tuples, descending.
        """
        if self._corr_matrix is None:
            import itertools
            return [(a, b, float('nan')) for a, b in itertools.combinations(cluster, 2)]

        pairs: list[tuple[str, str, float]] = []
        available = [s for s in cluster if s in self._corr_matrix.columns]
        for i, a in enumerate(available):
            for b in available[i + 1:]:
                pairs.append((a, b, float(self._corr_matrix.loc[a, b])))
        pairs.sort(key=lambda x: x[2] if not np.isnan(x[2]) else -1, reverse=True)
        return pairs[:n]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute(self, tickers: list[str], as_of: datetime) -> list[list[str]]:
        """Fetch returns up to as_of, run HDBSCAN, rank by expected yield."""
        end = as_of
        start = end - timedelta(days=self.lookback_days)

        prices = self._db.get_prices(tickers, start, end)
        if prices.empty:
            return [list(tickers)]

        # Require ≥50% coverage; forward-fill gaps then drop any column still NaN.
        # Cast to float64 explicitly — DB values come back as Python floats which
        # numpy ufuncs (np.log) cannot operate on via the array protocol.
        min_obs = max(2, int(prices.shape[0] * 0.5))
        prices = prices.dropna(axis=1, thresh=min_obs).ffill().astype(float)
        log_returns = np.log(prices).diff().dropna()
        log_returns = log_returns.dropna(axis=1)

        if log_returns.shape[1] < 2:
            remaining = list(log_returns.columns) if not log_returns.empty else list(tickers)
            return [remaining]

        symbols = list(log_returns.columns)

        # Standardise per-ticker returns then reduce with PCA before clustering.
        X = StandardScaler().fit_transform(log_returns.values.T)  # (n_tickers, n_days)
        X_reduced = self._pca_reduce(X)

        labels = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
        ).fit_predict(X_reduced)

        cluster_map: dict[int, list[str]] = {}
        noise: list[str] = []
        for symbol, label in zip(symbols, labels):
            if label == -1:
                noise.append(symbol)
            else:
                cluster_map.setdefault(label, []).append(symbol)

        if not cluster_map:
            return [symbols]

        # Rank each cluster by avg_intra_corr × size.
        corr_df = log_returns[symbols].corr()
        self._corr_matrix = corr_df
        self._symbols = symbols
        corr_matrix = corr_df.values
        idx_map = {s: i for i, s in enumerate(symbols)}

        ranked: list[tuple[float, list[str]]] = []
        for members in cluster_map.values():
            if len(members) < 2:
                noise.extend(members)
                continue
            idxs = [idx_map[s] for s in members]
            sub = corr_matrix[np.ix_(idxs, idxs)]
            n = len(idxs)
            avg_corr = (sub.sum() - n) / (n * (n - 1))
            ranked.append((avg_corr * n, members))

        ranked.sort(key=lambda x: x[0], reverse=True)
        result = [members for _, members in ranked]

        # Noise tickers appended as a low-priority tail cluster.
        if noise:
            result.append(noise)

        n_clusters = len(result)
        n_noise = len(noise)
        total = sum(len(c) for c in result)
        print(
            f"TickerClusterer: {total} tickers → {n_clusters} clusters "
            f"(min_size={self.min_cluster_size}, {n_noise} noise/tail tickers)"
        )
        return result

    def _pca_reduce(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce X (n_samples × n_features) to the minimum number of principal
        components that explain at least pca_variance of total variance.
        Falls back to the original X when PCA is not applicable (too few samples).
        """
        n_samples, n_features = X.shape
        max_components = min(n_samples - 1, n_features, 50)
        if max_components < 2:
            return X

        pca = PCA(n_components=max_components, random_state=42)
        X_pca = pca.fit_transform(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, self.pca_variance)) + 1
        return X_pca[:, :n_keep]
