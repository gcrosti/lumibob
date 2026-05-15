"""
TickerClusterer — groups tickers by movement similarity for efficient pair discovery.

Reduces the O(N²) pair search space by only evaluating pairs within clusters of
similarly-moving tickers. Clusters are ranked by expected pair yield
(avg_intra_cluster_corr × cluster_size) so BobsBrain searches the most fertile
clusters first.

Phase 2 improvements over the original:
  - Sector pre-partition: tickers are split by SIC sector (ETFs and unknowns
    each get their own partition) before HDBSCAN runs. This guarantees every
    cluster is intra-sector, eliminating the downstream sector gate entirely.
  - Correlation distance: HDBSCAN runs on ``1 - corr`` when
    ``hdbscan_metric='precomputed'``, directly optimising for co-movement rather
    than euclidean proximity in PCA space.  PCA is still used for the euclidean
    path and for the Ward agglomerative fallback.
  - Per-cluster sanity gate: clusters whose median intra-correlation falls below
    ``min_intra_cluster_corr`` are dissolved into the noise tail rather than
    forwarded to pair discovery.

Price history is loaded via an injectable ``get_prices`` callable; in BobsBrain
this is ``StockDataCache.get_prices`` so clustering matches the strategy's
DB + gap-fill path.
"""

import math
from collections.abc import Callable
from datetime import datetime, timedelta

# min_cluster_size scales with universe size so clustering density is consistent
# regardless of how many symbols the failed_tickers filter removes.
# Formula: max(_MCS_FLOOR, round(n_symbols * _MCS_FRACTION))
# At 5 000 symbols → mcs=5 (matches old fixed default).
# At 3 000 symbols → mcs=3 (prevents over-fragmentation on a smaller universe).
_MCS_FRACTION: float = 0.001
_MCS_FLOOR: int = 3

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from DatabaseClient import DatabaseClient


class TickerClusterer:
    """
    Clusters tickers by movement similarity and ranks clusters by expected
    pair-discovery yield.

    Usage:
        clusterer = TickerClusterer(db)
        clusters = clusterer.get_clusters(
            tickers, as_of=datetime.now(), ticker_metadata=metadata,
        )
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
        get_prices: Callable[[list[str], datetime, datetime], pd.DataFrame] | None = None,
        min_coverage: float = 0.5,
        hdbscan_min_samples: int = 2,
        hdbscan_metric: str = 'precomputed',
        hdbscan_selection_method: str = 'eom',
        hdbscan_cluster_selection_epsilon: float = 0.0,
        min_intra_cluster_corr: float = 0.3,
    ):
        """
        Parameters
        ----------
        db : DatabaseClient
            Used when ``get_prices`` is omitted (tests, DB-only workflows).
        lookback_days : int
            Calendar days of price history used to build clusters (~6 trading months).
        min_cluster_size : int
            Minimum tickers to form a cluster (HDBSCAN parameter).
        pca_variance : float
            Fraction of variance to retain after PCA (used only for the euclidean
            metric path and the Ward agglomerative fallback).
        get_prices : callable, optional
            ``(symbols, start, end) -> DataFrame`` price loader.
        min_coverage : float
            Minimum fraction of non-NaN bars required to keep a ticker.
        hdbscan_min_samples : int
            HDBSCAN ``min_samples`` — lower values declare less noise.
        hdbscan_metric : str
            ``'precomputed'``: HDBSCAN runs on ``1 - corr`` distance matrix
            (recommended; aligns clustering with the co-movement signal).
            ``'euclidean'``: HDBSCAN runs on PCA-reduced standardised returns
            (legacy behaviour).
        hdbscan_selection_method : str
            ``'eom'`` (larger clusters) or ``'leaf'`` (finer clusters).
        hdbscan_cluster_selection_epsilon : float
            Non-zero values merge nearby clusters, reducing noise at the cost
            of resolution.
        min_intra_cluster_corr : float
            Clusters whose median pairwise correlation falls below this threshold
            are dissolved into the noise tail (sanity gate).
        """
        self._db = db
        self._get_prices = get_prices if get_prices is not None else db.get_prices
        self.lookback_days = lookback_days
        self.min_cluster_size = min_cluster_size
        self.pca_variance = pca_variance
        self.min_coverage = min_coverage
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_metric = hdbscan_metric
        self.hdbscan_selection_method = hdbscan_selection_method
        self.hdbscan_cluster_selection_epsilon = hdbscan_cluster_selection_epsilon
        self.min_intra_cluster_corr = min_intra_cluster_corr

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
        ticker_metadata: dict[str, dict] | None = None,
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
        ticker_metadata : dict[str, dict] | None
            ``{symbol: {'sector': str | None, 'is_etf': bool}}`` — used to
            build sector partitions before clustering.  When None, all tickers
            are clustered together (legacy behaviour).

        Returns
        -------
        list[list[str]]
            Clusters sorted descending by expected yield.  The final cluster,
            when present, contains tickers that could not be assigned to any
            coherent group (noise tail).
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
            self._clusters = self._compute(tickers, as_of, ticker_metadata)
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
        order = list(available)
        seed = hash(('pairgen', tuple(sorted(cluster)))) % (2**32 - 1)
        np.random.default_rng(seed if seed > 0 else 1).shuffle(order)
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                pairs.append((a, b, float(self._corr_matrix.loc[a, b])))

        def _sort_key(t: tuple[str, str, float]) -> tuple:
            c = t[2]
            corr = float(c) if not np.isnan(c) else -1.0
            tie = hash(t[0]) ^ hash(t[1])
            return (-corr, tie)

        pairs.sort(key=_sort_key)
        return pairs[:n]

    # ------------------------------------------------------------------
    # Internal — orchestration
    # ------------------------------------------------------------------

    def _compute(
        self,
        tickers: list[str],
        as_of: datetime,
        ticker_metadata: dict[str, dict] | None,
    ) -> list[list[str]]:
        """Fetch returns, run sector-partitioned HDBSCAN, rank by expected yield."""
        end = as_of
        start = end - timedelta(days=self.lookback_days)

        prices = self._get_prices(tickers, start, end)
        if prices.empty:
            print(
                '[TickerClusterer] No prices in clustering window '
                f'({start.date()} .. {end.date()}); '
                'falling back to single cluster (HDBSCAN skipped).',
            )
            return [list(tickers)]

        # Require min_coverage of bars to be non-NaN; forward-fill remaining gaps.
        min_obs = max(2, int(prices.shape[0] * self.min_coverage))
        prices = prices.dropna(axis=1, thresh=min_obs).ffill().astype(float)
        log_returns = np.log(prices).diff().dropna().dropna(axis=1)

        if log_returns.shape[1] < 2:
            remaining = list(log_returns.columns) if not log_returns.empty else list(tickers)
            return [remaining]

        symbols = list(log_returns.columns)

        # Scale min_cluster_size with universe size so cluster density is
        # consistent regardless of how many symbols failed_tickers removes.
        self._dynamic_mcs = max(_MCS_FLOOR, round(len(symbols) * _MCS_FRACTION))

        # Compute correlation matrix once — used for HDBSCAN distance (precomputed
        # path) and for cluster ranking / get_top_pairs_by_corr.
        corr_df = log_returns[symbols].corr()
        self._corr_matrix = corr_df
        self._symbols = symbols
        corr_matrix = corr_df.values
        idx_map = {s: i for i, s in enumerate(symbols)}

        # Build sector partitions; fall back to single partition when no metadata.
        partitions = self._build_partitions(symbols, ticker_metadata)

        # Cluster each partition independently.
        all_valid_clusters: list[list[str]] = []
        all_noise: list[str] = []

        for partition_name, partition_syms in partitions:
            clusters, noise = self._cluster_partition(
                partition_syms, log_returns, corr_df,
            )
            all_valid_clusters.extend(clusters)
            all_noise.extend(noise)

        # Per-cluster sanity gate: dissolve clusters with low median intra-corr.
        passed: list[list[str]] = []
        for members in all_valid_clusters:
            idxs = [idx_map[s] for s in members if s in idx_map]
            if len(idxs) < 2:
                all_noise.extend(members)
                continue
            sub = corr_matrix[np.ix_(idxs, idxs)]
            n = len(idxs)
            # Median of upper triangle (excluding diagonal)
            upper = sub[np.triu_indices(n, k=1)]
            median_corr = float(np.median(upper))
            if median_corr < self.min_intra_cluster_corr:
                all_noise.extend(members)
            else:
                passed.append(members)

        # Rank passing clusters by avg_intra_corr × size.
        ranked: list[tuple[float, list[str]]] = []
        for members in passed:
            idxs = [idx_map[s] for s in members]
            sub = corr_matrix[np.ix_(idxs, idxs)]
            n = len(idxs)
            avg_corr = (sub.sum() - n) / (n * (n - 1))
            ranked.append((avg_corr * n, members))

        ranked.sort(key=lambda x: x[0], reverse=True)
        result = [members for _, members in ranked]

        if all_noise:
            result.append(all_noise)

        n_clusters = len(result)
        n_noise = len(all_noise)
        total = sum(len(c) for c in result)
        n_partitions = len(partitions)
        print(
            f'TickerClusterer: {total} tickers → {n_clusters} clusters '
            f'across {n_partitions} sector partitions '
            f'(min_size={self._dynamic_mcs} [universe={len(symbols)}], metric={self.hdbscan_metric}, '
            f'{n_noise} noise/tail tickers, '
            f'{len(passed)} clusters passed sanity gate)'
        )
        return result

    # ------------------------------------------------------------------
    # Internal — sector partitioning
    # ------------------------------------------------------------------

    def _build_partitions(
        self,
        symbols: list[str],
        ticker_metadata: dict[str, dict] | None,
    ) -> list[tuple[str, list[str]]]:
        """
        Split ``symbols`` into sector partitions using ``ticker_metadata``.

        Groups:
          - ETFs — clustered together regardless of SIC sector
          - Known sectors — one partition per SIC sector label
          - Unknown — tickers with no metadata or no sector assigned (Option B)

        When ``ticker_metadata`` is None, returns a single ``('all', symbols)``
        partition (legacy behaviour).
        """
        if not ticker_metadata:
            return [('all', symbols)]

        etf_group: list[str] = []
        sector_groups: dict[str, list[str]] = {}
        unknown_group: list[str] = []

        for sym in symbols:
            meta = ticker_metadata.get(sym)
            if meta is None:
                unknown_group.append(sym)
            elif meta.get('is_etf'):
                etf_group.append(sym)
            elif meta.get('sector'):
                sector_groups.setdefault(meta['sector'], []).append(sym)
            else:
                unknown_group.append(sym)

        partitions: list[tuple[str, list[str]]] = []
        if etf_group:
            partitions.append(('ETFs', etf_group))
        for sector in sorted(sector_groups):
            partitions.append((sector, sector_groups[sector]))
        if unknown_group:
            partitions.append(('Unknown', unknown_group))

        # Log partition sizes.
        summary = ', '.join(
            f'{name}:{len(syms)}' for name, syms in partitions
        )
        print(f'TickerClusterer: sector partitions — {summary}')
        return partitions

    # ------------------------------------------------------------------
    # Internal — per-partition clustering
    # ------------------------------------------------------------------

    def _cluster_partition(
        self,
        partition_syms: list[str],
        log_returns: pd.DataFrame,
        corr_df: pd.DataFrame,
    ) -> tuple[list[list[str]], list[str]]:
        """
        Run HDBSCAN on a single sector partition.

        Returns ``(clusters, noise)`` where ``clusters`` is a list of member
        lists (each with >= 2 tickers) and ``noise`` is the list of unassigned
        tickers.  Singletons from any cluster label are moved to noise.

        The metric path is chosen by ``self.hdbscan_metric``:
          - ``'precomputed'``: distance matrix = ``clip(1 - sub_corr, 0, 2)``
          - ``'euclidean'``:   PCA-reduced standardised log-returns
        Both paths share the same Ward agglomerative fallback.
        """
        n_sym = len(partition_syms)

        if n_sym < 2:
            return [], partition_syms

        # Build HDBSCAN input matrix and euclidean fallback (for Ward).
        sub_returns = log_returns[partition_syms]
        X_euclidean = self._pca_reduce(
            StandardScaler().fit_transform(sub_returns.values.T)
        )

        if self.hdbscan_metric == 'precomputed':
            sub_corr = corr_df.loc[partition_syms, partition_syms].values
            # NaN correlations (constant-return or zero-variance tickers) are
            # treated as uncorrelated (corr=0 → distance=1, i.e. far apart).
            sub_corr = np.nan_to_num(sub_corr, nan=0.0)
            dist = np.clip(1.0 - sub_corr, 0.0, 2.0)
            np.fill_diagonal(dist, 0.0)
            X_hdbscan = dist
        else:
            X_hdbscan = X_euclidean

        # First HDBSCAN attempt with universe-scaled min_cluster_size.
        labels = self._run_hdbscan(X_hdbscan, self._dynamic_mcs, self.hdbscan_min_samples)
        cluster_map, noise = self._labels_to_map(partition_syms, labels)

        # Retry with relaxed params if all tickers went to noise.
        if not cluster_map:
            relaxed_mcs = min(3, self._dynamic_mcs)
            print(
                f'TickerClusterer: HDBSCAN all-noise on partition of {n_sym} tickers; '
                f'retrying with min_cluster_size={relaxed_mcs}, min_samples=1',
            )
            labels = self._run_hdbscan(X_hdbscan, relaxed_mcs, 1)
            cluster_map, noise = self._labels_to_map(partition_syms, labels)

        # Ward agglomerative fallback (always on euclidean X).
        if not cluster_map:
            k = max(2, min(max(2, n_sym // 25), min(100, n_sym)))
            print(
                f'TickerClusterer: HDBSCAN still all-noise; '
                f'Ward fallback with n_clusters={k} on partition of {n_sym} tickers',
            )
            ward_labels = AgglomerativeClustering(
                n_clusters=k, linkage='ward',
            ).fit_predict(X_euclidean)
            cluster_map, noise = self._labels_to_map(partition_syms, ward_labels)

        # Move singletons to noise.
        clusters: list[list[str]] = []
        for members in cluster_map.values():
            if len(members) >= 2:
                clusters.append(members)
            else:
                noise.extend(members)

        return clusters, noise

    def _run_hdbscan(self, X: np.ndarray, mcs: int, ms: int) -> np.ndarray:
        return hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric=self.hdbscan_metric,
            cluster_selection_method=self.hdbscan_selection_method,
            cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
        ).fit_predict(X)

    @staticmethod
    def _labels_to_map(
        symbols: list[str],
        labels: np.ndarray,
    ) -> tuple[dict[int, list[str]], list[str]]:
        """Split HDBSCAN label array into a cluster map and noise list."""
        cluster_map: dict[int, list[str]] = {}
        noise: list[str] = []
        for sym, label in zip(symbols, labels):
            if label == -1:
                noise.append(sym)
            else:
                cluster_map.setdefault(int(label), []).append(sym)
        return cluster_map, noise

    # ------------------------------------------------------------------
    # Internal — PCA reduction
    # ------------------------------------------------------------------

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
