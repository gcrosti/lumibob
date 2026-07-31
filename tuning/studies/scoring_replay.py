"""Scoring-quality replay harness — Pass A v4, WS2.

Reconstructs the full candidate pool via the strategy's *own* clusterer at sampled
scoring dates, scores each candidate's discovery-time components with the real
``StockEvaluator``, and computes its P&L-free **forward gross spread outcome** under a
frozen exit rule.  Emits a cached observation table so the WS3 scoring study can
reweight it — and re-window the corr components — in seconds.  No lumibot, no
portfolio, no costs.

This is the machinery that removes the old Pass A gate's range restriction: it scores
every dislocated candidate, not just the top-K the composite already selected.

Look-ahead discipline (critical):
  * components use only prices up to and including the scoring date T;
  * the hedge ratio is frozen on the pre-T lookback;
  * the forward outcome uses only (T, T+HORIZON].

Frozen "outcome contract" (never tuned by WS3): ``ZSCORE_WINDOW``, the exit rule,
the hedge lookback, ``COINT_CEIL``, ``MAX_HALFLIFE``.  The corr windows are *not*
frozen — WS3 tunes them, so we cache the trailing log-returns for per-trial recompute.

See docs/plans/2026-07-22_pass-a-v4-scoring-study.md.
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from DatabaseClient import DatabaseClient
from StockEvaluator import StockEvaluator, halflife_to_score
from TickerClusterer import TickerClusterer

# --- Frozen outcome contract + fixed component params (from the gate-run settings) ---
ZSCORE_WINDOW = 32          # frozen: defines z-path + exit -> defines the outcome
ENTRY_THRESHOLD = 2.0       # frozen: a pair must be dislocated to be tradeable
EXIT_Z = 0.5                # frozen: take-profit convention for the forward outcome
COINT_CEIL = 0.20           # fixed normalisation constant (not tunable)
MAX_HALFLIFE = 60.0         # log-ceiling for halflife_to_score (WS1)
HORIZON = 40                # forward trading days for the outcome
LOOKBACK_WINDOW = 152       # frozen: component window in CAL days (matches live
                            # start_date = end_date - lookback_window); ~105 td
MAX_CORR_LONG = 100         # widest corr_long_window (bounded by LOOKBACK_WINDOW) -> returns cached

LOOKBACK_CAL = 400          # calendar days of pre-T history to FETCH (bounded to LOOKBACK_WINDOW in-code)
HORIZON_CAL = 70            # calendar days after T to cover HORIZON trading days

# Default corr windows (gate values) — used for the baseline column; WS3 re-windows.
DEF_CORR_LONG = 84
DEF_CORR_SHORT = 25


def _forward_gross(spread_fwd: np.ndarray, z_fwd: np.ndarray, z0_signed: float,
                   hedge: float, exit_z: float = EXIT_Z, cap: int = HORIZON) -> float | None:
    """Gross bps of gross notional for one dislocated pair under the frozen exit.

    ``spread_fwd`` is the log-spread from T forward; ``z_fwd`` its |z| path; ``z0_signed``
    the signed entry z (its sign fixes the trade direction — look-ahead-safe, entry
    info only).  Mirrors the Phase-1 notebook's ``sim_exit`` at exit_z=0.5.
    """
    n = min(cap, len(spread_fwd) - 1)
    if n < 1:
        return None
    sgn = np.sign(z0_signed)
    scale = (1 + abs(hedge))
    for t in range(1, n + 1):
        if z_fwd[t] <= exit_z:
            return float(-sgn * (spread_fwd[t] - spread_fwd[0]) / scale * 1e4)
    return float(-sgn * (spread_fwd[n] - spread_fwd[0]) / scale * 1e4)


def _score_pair(px: pd.DataFrame, lead: str, lag: str, T: pd.Timestamp,
                evaluator: StockEvaluator) -> dict | None:
    """Full observation for one candidate at scoring date T, or None if unusable.

    Returns discovery-time components (look-ahead-safe) + the forward gross outcome +
    trailing log-returns (for WS3 corr re-windowing).  Only dislocated pairs
    (|z_entry| >= ENTRY_THRESHOLD) get a finite forward outcome — undislocated pairs
    are not tradeable and carry NaN outcome (excluded from the WS3 metric).
    """
    if lead not in px.columns or lag not in px.columns:
        return None
    both = px[[lead, lag]].dropna()
    if both.empty:
        return None
    # Align to the last trading day <= T (index is stamped at 05:00, not midnight),
    # and bound the component window to lookback_window (matches live scoring, which
    # feeds coint/half-life/corr/z a series of exactly lookback_window days).
    pos = int(both.index.searchsorted(T, side='right')) - 1
    if pos < 0:
        return None
    lo = int(both.index.searchsorted(T - timedelta(days=LOOKBACK_WINDOW), side='left'))
    pre = both.iloc[lo:pos + 1]
    if len(pre) < ZSCORE_WINDOW + 15:
        return None
    lead_pre, lag_pre = pre[lead], pre[lag]

    # --- Dislocation gate FIRST (cheap): only tradeable pairs are kept/scored ---
    # A non-dislocated pair (|z| < entry) has no round-trip, so it never enters the
    # WS3 metric — skipping it here also skips its ADF/coint cost (~10x speedup).
    z_depth, z_raw = evaluator.compute_z_depth(
        lead_pre, lag_pre, ZSCORE_WINDOW, ENTRY_THRESHOLD, EXIT_Z)
    z_entry = abs(z_raw) if z_raw is not None and np.isfinite(z_raw) else np.nan
    if not (np.isfinite(z_entry) and z_entry >= ENTRY_THRESHOLD):
        return None

    # --- Components (prices up to and including T) — real StockEvaluator ---
    corr_long, corr_short = evaluator.get_correlation_dual(
        lead_pre, lag_pre, DEF_CORR_LONG, DEF_CORR_SHORT)
    ss = evaluator.compute_spread_scores(
        lead_pre, lag_pre, coint_pvalue_ceiling=COINT_CEIL, max_halflife_days=MAX_HALFLIFE)

    # --- Trailing aligned log-returns (both legs) for WS3 corr re-windowing ---
    ll = np.log(lead_pre.astype(float).clip(lower=1e-9)).diff().dropna()
    lg = np.log(lag_pre.astype(float).clip(lower=1e-9)).diff().dropna()
    common = ll.index.intersection(lg.index)[-MAX_CORR_LONG:]
    ret_lead = ll.loc[common].to_numpy()
    ret_lag = lg.loc[common].to_numpy()

    # --- Forward outcome (only (T, T+HORIZON]), hedge frozen on pre-T ---
    forward_gross = np.nan
    try:
        # DB get_prices can return NUMERIC as decimal.Decimal; np.log on Decimal raises
        # TypeError. Cast to float first (matching StockEvaluator's price handling), and
        # keep TypeError in the guard below as a belt-and-braces net.
        hedge = float(np.polyfit(
            np.log(pre[lead].astype(float).clip(lower=1e-9)).to_numpy(),
            np.log(pre[lag].astype(float).clip(lower=1e-9)).to_numpy(), 1)[0])
        spread = (np.log(both[lag].astype(float).clip(lower=1e-9))
                  - hedge * np.log(both[lead].astype(float).clip(lower=1e-9)))
        zc = (spread - spread.rolling(ZSCORE_WINDOW).mean()) / spread.rolling(ZSCORE_WINDOW).std()
        z0_signed = zc.iloc[pos]
        spread_fwd = spread.iloc[pos:pos + HORIZON + 1].to_numpy()
        z_fwd = zc.iloc[pos:pos + HORIZON + 1].abs().to_numpy()
        if len(spread_fwd) >= 2 and np.isfinite(z_fwd[0]) and np.isfinite(z0_signed):
            forward_gross = _forward_gross(spread_fwd, z_fwd, z0_signed, hedge)
    except (np.linalg.LinAlgError, ValueError, KeyError, TypeError):
        forward_gross = np.nan

    return dict(
        lead=lead, lag=lag,
        corr_long=corr_long, corr_short=corr_short, z_depth=z_depth,
        coint_pvalue=ss.coint_pvalue, halflife_days=ss.halflife_days,
        score_coint=ss.coint_score, score_halflife=ss.halflife_score,
        z_entry=z_entry, forward_gross=forward_gross,
        ret_lead=ret_lead, ret_lag=ret_lag,
    )


def build_fold(fold_name: str, scoring_dates: list[str], db: DatabaseClient,
               clusterer: TickerClusterer, evaluator: StockEvaluator,
               tickers: list[str], metadata: dict, max_pairs_per_date: int = 1500,
               verbose: bool = True) -> pd.DataFrame:
    """Build the observation table for one fold across the given scoring dates."""
    rows = []
    for ds in scoring_dates:
        T = pd.Timestamp(ds)
        clusters = clusterer.get_clusters(
            tickers, as_of=T.to_pydatetime(), recompute_days=0, ticker_metadata=metadata)
        cand: list[tuple] = []
        for cl in clusters:
            cand += clusterer.get_top_pairs_by_corr(cl, n=len(cl) * (len(cl) - 1) // 2)
        cand = sorted(cand, key=lambda x: -x[2])[:max_pairs_per_date]
        syms = sorted({s for p in cand for s in p[:2]})
        px = db.get_prices(syms, (T - timedelta(days=LOOKBACK_CAL)).to_pydatetime(),
                           (T + timedelta(days=HORIZON_CAL)).to_pydatetime())
        if px.empty:
            if verbose:
                print(f'  {ds}: no prices')
            continue
        px.index = pd.to_datetime(px.index).normalize()
        px = px.astype(float)   # DB returns NUMERIC as Decimal; np.log needs float
        n_disloc = 0
        for s1, s2, _cc in cand:
            obs = _score_pair(px, s1, s2, T, evaluator)
            if obs is None:
                continue
            if np.isfinite(obs['forward_gross']):
                n_disloc += 1
            rows.append({'fold': fold_name, 'date': T, **obs})
        if verbose:
            print(f'  {ds}: {len(cand)} candidates, {n_disloc} dislocated (tradeable)')
    return pd.DataFrame(rows)


# Fold windows (active scoring windows, matching the gate runs).
FOLDS = {
    'sideways_2022': ['2022-01-14', '2022-02-15', '2022-03-15', '2022-04-14'],
    'bull_2023':     ['2023-02-15', '2023-03-15', '2023-04-14', '2023-05-15'],
    'mixed_2023_q4': ['2023-07-17', '2023-08-15', '2023-09-15', '2023-10-16'],
}
# Cluster params from the gate-run settings (fixed; pool-defining).
CLUSTER_KW = dict(lookback_days=126, min_cluster_size=5, pca_variance=0.95,
                  min_coverage=0.5, hdbscan_min_samples=2, hdbscan_metric='precomputed',
                  hdbscan_selection_method='eom', hdbscan_cluster_selection_epsilon=0.0)
CACHE_DIR = os.path.join(os.path.dirname(__file__), '_scoring_cache')


def _make_clusterer(db: DatabaseClient) -> TickerClusterer:
    kw = dict(CLUSTER_KW)
    try:
        return TickerClusterer(db=db, get_prices=db.get_prices, min_intra_cluster_corr=0.3, **kw)
    except TypeError:
        return TickerClusterer(db=db, get_prices=db.get_prices, **kw)


def main(folds: list[str] | None = None, max_pairs_per_date: int = 1500) -> None:
    db = DatabaseClient(os.environ['DB_URL'])
    evaluator = StockEvaluator()
    clusterer = _make_clusterer(db)
    tickers = db.get_tickers()
    md_df = db.get_ticker_metadata(tickers)
    metadata = {r['symbol']: {'sector': r['sector'] if pd.notna(r['sector']) else None,
                              'is_etf': bool(r['is_etf'])}
                for _, r in md_df.iterrows()} if not md_df.empty else {}
    print(f'universe: {len(tickers)} tickers, {len(metadata)} with metadata')
    os.makedirs(CACHE_DIR, exist_ok=True)
    for fold in (folds or list(FOLDS)):
        print(f'\n=== {fold} ===')
        df = build_fold(fold, FOLDS[fold], db, clusterer, evaluator, tickers, metadata,
                        max_pairs_per_date=max_pairs_per_date)
        out = os.path.join(CACHE_DIR, f'{fold}.parquet')
        df.to_parquet(out)
        disloc = df.forward_gross.notna().sum() if len(df) else 0
        print(f'  -> {len(df)} obs ({disloc} tradeable) saved to {out}')
    db.close()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='*', default=None)
    ap.add_argument('--max-pairs', type=int, default=1500)
    a = ap.parse_args()
    main(a.folds, a.max_pairs)
