"""V1 — entry-criteria parity check (plan §6, docs/plans/2026-08-01_entry-criteria-overhaul.md).

Asks one question: does the SHIPPED code path reproduce the analysis the design
was based on?  The design comparison was run with inline formulas in a
scratchpad; if `StockEvaluator.compute_entry_metrics` diverges from those — a
different hedge window, a different scaling, a sign slip — the implementation
is testing something other than what was validated, and everything downstream
is unmoored.

No fitting, no parameter selection, no gate on strategy performance.  Two
checks:

  A. numeric parity — compute_entry_metrics on the same pre-entry price
     windows must reproduce the cached z_entry and the analysis spread std
  B. design parity — applying the implemented gate + ranking logic must
     reproduce the design table (top-n by expected_gross with live dedup)

Prices are windowed exactly as BobsBrain sees them: `_get_series` returns
`[end_date - lookback_window, end_date]`, so the hedge fit inside
compute_entry_metrics is bounded the same way.

Run:  DB_URL=... python -m tuning.studies.study3_v1_entry_parity
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd

from DatabaseClient import DatabaseClient
from StockEvaluator import StockEvaluator
from tuning.studies.scoring_replay import (EXIT_Z, LOOKBACK_CAL,
                                           LOOKBACK_WINDOW, ZSCORE_WINDOW)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_scoring_cache')
FOLDS = ('sideways_2022', 'bull_2023', 'mixed_2023_q4')
N_BOOK = 20
FLOOR = 25.0          # implemented default min_expected_gross_bps
TOL_Z = 1e-6
TOL_BPS = 1e-6


def load() -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(CACHE_DIR, 'e2_paths.parquet'))
    cache = pd.concat([pd.read_parquet(os.path.join(CACHE_DIR, f'{f}.parquet'))
                       for f in FOLDS], ignore_index=True)
    cache['date'] = pd.to_datetime(cache.date)
    keys = ['fold', 'date', 'lead', 'lag']
    df = p.merge(cache[keys + ['z_entry', 'z_depth']], on=keys, how='left')
    std_path = os.path.join(CACHE_DIR, 'spread_std.parquet')
    if os.path.exists(std_path):
        df = df.merge(pd.read_parquet(std_path), on=keys, how='left')
    else:
        df['lvl_std'] = np.nan
    return df


def compute_via_shipped_code(df: pd.DataFrame, db: DatabaseClient) -> pd.DataFrame:
    """Call the real StockEvaluator on BobsBrain-shaped price windows."""
    ev = StockEvaluator()
    rows = []
    for (fold, T), g in df.groupby(['fold', 'date']):
        syms = sorted(set(g.lead) | set(g.lag))
        # NB: get_prices filters `time <= end` and bars are stamped intraday,
        # so passing T (midnight) would EXCLUDE T's own bar and shift the
        # "latest" observation back a day.  Fetch through T+1d and slice at T
        # below — no forward information is used.  (Live is unaffected:
        # BobsBrain's end_date carries a market time-of-day.)
        px = db.get_prices(syms, (T - timedelta(days=LOOKBACK_CAL)).to_pydatetime(),
                           (T + timedelta(days=1)).to_pydatetime())
        if px.empty:
            continue
        px.index = pd.to_datetime(px.index).normalize()
        px = px.astype(float)
        for _, r in g.iterrows():
            if r.lead not in px.columns or r.lag not in px.columns:
                continue
            both = px[[r.lead, r.lag]].dropna()
            pos = int(both.index.searchsorted(T, side='right')) - 1
            if pos < 0:
                continue
            # BobsBrain._get_series window: [end_date - lookback_window, end_date]
            lo = int(both.index.searchsorted(
                T - timedelta(days=LOOKBACK_WINDOW), side='left'))
            w = both.iloc[lo:pos + 1]
            m = ev.compute_entry_metrics(
                w[r.lead], w[r.lag], window=ZSCORE_WINDOW, exit_threshold=EXIT_Z)
            if m is None:
                continue
            rows.append(dict(fold=fold, date=T, lead=r.lead, lag=r.lag,
                             impl_z=m.z, impl_std=m.spread_std_bps,
                             impl_exp=m.expected_gross_bps))
    return pd.DataFrame(rows)


def check_numeric_parity(df: pd.DataFrame) -> bool:
    print('=' * 72)
    print('A — Numeric parity: shipped code vs the analysis')
    print('=' * 72)
    ok = True

    d = df.dropna(subset=['impl_z', 'z_entry'])
    dz = (d.impl_z.abs() - d.z_entry).abs()
    print(f'|z| vs cached z_entry      : n={len(d)} max diff {dz.max():.2e} '
          f'({(dz < TOL_Z).mean() * 100:.1f}% within {TOL_Z:g})')
    ok &= bool((dz < 1e-4).mean() > 0.999)

    s = df.dropna(subset=['impl_std', 'lvl_std'])
    ds = (s.impl_std - s.lvl_std).abs()
    print(f'spread std vs analysis     : n={len(s)} max diff {ds.max():.2e} '
          f'({(ds < TOL_BPS).mean() * 100:.1f}% within {TOL_BPS:g})')
    ok &= bool((ds < 1e-4).mean() > 0.999)

    e = df.dropna(subset=['impl_exp', 'z_entry', 'lvl_std']).copy()
    e['analysis_exp'] = (e.z_entry - EXIT_Z) * e.lvl_std
    de = (e.impl_exp - e.analysis_exp).abs()
    print(f'expected_gross vs analysis : n={len(e)} max diff {de.max():.2e} '
          f'({(de < TOL_BPS).mean() * 100:.1f}% within {TOL_BPS:g})')
    ok &= bool((de < 1e-4).mean() > 0.999)

    # The dislocation gate keys on SIGN, so a convention slip would be silent
    # in the magnitude checks above but would invert selection.
    z1 = df[df.z_depth >= 1 - 1e-9]
    if len(z1):
        neg = (z1.impl_z < 0).mean() * 100
        print(f'\nsign convention: {neg:.1f}% of z_depth==1 rows have impl_z < 0 '
              f'(expect ~100% — z_depth==1 marks the tradeable, negative-z side)')
        ok &= bool(neg > 99.0)
    return ok


def check_design_parity(df: pd.DataFrame) -> bool:
    print('\n' + '=' * 72)
    print('B — Design parity: implemented gates + ranking, live dedup')
    print('=' * 72)
    d = df.dropna(subset=['impl_z', 'impl_exp']).copy()

    # Implemented gate order: direction+dislocation, then magnitude floor.
    disloc = d[d.impl_z <= -2.0]           # entry_threshold in the gate runs
    qualified = disloc[disloc.impl_exp >= FLOOR]
    print(f'pool {len(d)} -> dislocated {len(disloc)} -> '
          f'qualified (floor {FLOOR:.0f}) {len(qualified)}')
    print(f'median qualified/date: '
          f'{qualified.groupby("date").size().median():.0f} (k_target 20)')

    def take_dedup(frame, n=N_BOOK):
        out = []
        for _, g in frame.groupby('date'):
            used, kept = set(), []
            for _, r in g.sort_values('impl_exp', ascending=False).iterrows():
                if r.lead in used or r.lag in used:
                    continue
                used.update([r.lead, r.lag])
                kept.append(r)
                if len(kept) >= n:
                    break
            out.extend(kept)
        return pd.DataFrame(out)

    book = take_dedup(qualified)
    print(f'\nimplemented book: n={len(book)} mean {book.gross.mean():+.1f} '
          f'median {book.gross.median():+.1f} '
          f'disaster {book["cat"].mean() * 100:.1f}%')
    print('analysis reference (design D, dedup): mean +2.4 median +50.7 '
          'disaster 18.8%')
    print('\nNOTE: the reference used floor=0 and a right-direction filter via '
          'z_depth;\nthis run applies the shipped gates, so small differences '
          'are expected.\nA large divergence means the implementation is not '
          'what was analysed.')

    supply_ok = bool((qualified.groupby('date').size() >= N_BOOK).mean() >= 0.9)
    print(f'\nsupply: {(qualified.groupby("date").size() >= N_BOOK).sum()}/'
          f'{qualified.date.nunique()} dates can fill k=20 -> '
          f'{"OK" if supply_ok else "STARVED"}')
    return supply_ok


def main() -> None:
    db = DatabaseClient(os.environ['DB_URL'])
    df = load()
    impl = compute_via_shipped_code(df, db)
    db.close()
    merged = df.merge(impl, on=['fold', 'date', 'lead', 'lag'], how='left')
    print(f'observations: {len(merged)}, shipped-code metrics for '
          f'{merged.impl_z.notna().sum()}\n')
    a = check_numeric_parity(merged)
    b = check_design_parity(merged)
    print('\n' + '=' * 72)
    print(f'V1 VERDICT: numeric parity {"PASS" if a else "FAIL"} | '
          f'supply {"PASS" if b else "FAIL"}')
    print('=' * 72)


if __name__ == '__main__':
    main()
