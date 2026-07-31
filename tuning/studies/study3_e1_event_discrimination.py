"""Study E1 — do item-coded filing events separate catastrophic round trips?

Full-pool version of the entered-pairs dry run
(study2_event_retro/edgar_event_analysis.py), per the plan
(docs/plans/2026-07-31_composite-score-overhaul.md §5b).

Data: the Pass A v4 replay cache (tuning/studies/_scoring_cache/<fold>.parquet)
restricted to tradeable observations (|z_entry| >= 2 by construction), joined
to filing_events over each observation's holding window.

The cache stores the forward outcome but not the exit day, so the holding
window is recomputed here from stock_prices with the same frozen-hedge z0.5
exit as scoring_replay._forward_gross; each recomputed gross is checked
against the cached value (>1 bps disagreement = row dropped and counted, a
consistency alarm if frequent).

Preregistered (no fitting, no free parameters):
  - item groups: results (2.02), deals (1.01/2.01), exec_change (5.02),
    restatement (4.02), guidance (7.01), foreign (any 6-K)
  - window: (entry, exit] plus the 7 calendar days before entry
  - catastrophic threshold: gross < -100 bps
  - readout: per-fold and pooled rates with Fisher tests and bootstrap CIs
    on the rate difference; duration-control cut by holding length

Run:  DB_URL=... python -m tuning.studies.study3_e1_event_discrimination
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from DatabaseClient import DatabaseClient
# Exact outcome-contract constants — the exit replay below must mirror
# scoring_replay._score_pair to the row, or the consistency check fails.
from tuning.studies.scoring_replay import (
    EXIT_Z,
    HORIZON,
    HORIZON_CAL,
    LOOKBACK_CAL,
    LOOKBACK_WINDOW,
    ZSCORE_WINDOW,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_scoring_cache')
OUT = os.path.join(CACHE_DIR, 'e1_events.parquet')

FOLDS = ('sideways_2022', 'bull_2023', 'mixed_2023_q4')
CATASTROPHIC_BPS = -100
PRE_ENTRY_DAYS = 7
TOL_BPS = 1.0

# Single-stock leveraged ETFs -> underlying (same map as study2).
UNDERLYING = {'AAPB': 'AAPL', 'AAPU': 'AAPL', 'FBL': 'META'}

ITEM_GROUPS = {
    'results': lambda form, items: form.startswith('8-K') and '2.02' in items,
    'deals': lambda form, items: form.startswith('8-K') and ('1.01' in items or '2.01' in items),
    'exec_change': lambda form, items: form.startswith('8-K') and '5.02' in items,
    'restatement': lambda form, items: form.startswith('8-K') and '4.02' in items,
    'guidance': lambda form, items: form.startswith('8-K') and '7.01' in items,
    'foreign': lambda form, items: form == '6-K',
}


def load_cache() -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        path = os.path.join(CACHE_DIR, f'{fold}.parquet')
        if not os.path.exists(path):
            raise SystemExit(f'missing cache fold: {path} — build via scoring_replay first')
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df.forward_gross)].copy()
    df['date'] = pd.to_datetime(df.date)
    return df


def add_exit_dates(df: pd.DataFrame, db: DatabaseClient) -> pd.DataFrame:
    """Recompute each observation's exit day (z0.5 frozen exit) and verify the
    cached forward_gross. Returns rows that verify within TOL_BPS."""
    rows = []
    dropped = 0
    for (fold, T), grp in df.groupby(['fold', 'date']):
        syms = sorted(set(grp.lead) | set(grp.lag))
        px = db.get_prices(syms, (T - timedelta(days=LOOKBACK_CAL)).to_pydatetime(),
                           (T + timedelta(days=HORIZON_CAL)).to_pydatetime())
        px.index = pd.to_datetime(px.index).normalize()
        px = px.astype(float)
        for _, r in grp.iterrows():
            res = _exit_for(px, r.lead, r.lag, T)
            if res is None or abs(res['gross'] - r.forward_gross) > TOL_BPS:
                dropped += 1
                continue
            rows.append({**r.to_dict(), **res})
    out = pd.DataFrame(rows)
    print(f'exit-date replay: {len(out)} verified, {dropped} dropped '
          f'({dropped / max(len(df), 1) * 100:.1f}% — investigate if > a few %)')
    return out


def _exit_for(px: pd.DataFrame, lead: str, lag: str, T: pd.Timestamp) -> dict | None:
    """Mirror scoring_replay._score_pair's forward-outcome block exactly,
    additionally returning the exit day the frozen z0.5 exit lands on."""
    if lead not in px.columns or lag not in px.columns:
        return None
    both = px[[lead, lag]].dropna()
    if both.empty:
        return None
    pos = int(both.index.searchsorted(T, side='right')) - 1
    if pos < 0:
        return None
    lo = int(both.index.searchsorted(T - timedelta(days=LOOKBACK_WINDOW), side='left'))
    pre = both.iloc[lo:pos + 1]
    if len(pre) < ZSCORE_WINDOW + 15:
        return None
    try:
        hedge = float(np.polyfit(
            np.log(pre[lead].astype(float).clip(lower=1e-9)).to_numpy(),
            np.log(pre[lag].astype(float).clip(lower=1e-9)).to_numpy(), 1)[0])
    except Exception:
        return None
    spread = (np.log(both[lag].astype(float).clip(lower=1e-9))
              - hedge * np.log(both[lead].astype(float).clip(lower=1e-9)))
    z = (spread - spread.rolling(ZSCORE_WINDOW).mean()) / spread.rolling(ZSCORE_WINDOW).std()
    z0 = z.iloc[pos]
    if not np.isfinite(z0):
        return None
    sgn = np.sign(z0)
    scale = 1 + abs(hedge)
    zf = z.iloc[pos:pos + HORIZON + 1].abs()
    sf = spread.iloc[pos:pos + HORIZON + 1]
    n = min(HORIZON, len(sf) - 1)
    if n < 1 or not np.isfinite(zf.iloc[0]):
        return None
    exit_t = n
    for t in range(1, n + 1):
        if zf.iloc[t] <= EXIT_Z:
            exit_t = t
            break
    gross = float(-sgn * (sf.iloc[exit_t] - sf.iloc[0]) / scale * 1e4)
    return dict(gross=gross, hold_days=exit_t, exit_date=sf.index[exit_t])


def join_events(df: pd.DataFrame, db: DatabaseClient) -> pd.DataFrame:
    syms = sorted({UNDERLYING.get(s, s) for s in set(df.lead) | set(df.lag)})
    ev = db.get_filing_events(syms, df.date.min() - pd.Timedelta(days=PRE_ENTRY_DAYS),
                              df.exit_date.max() + pd.Timedelta(days=1))
    ev['items'] = ev['items'].fillna('')
    ev['day'] = pd.to_datetime(ev.filed_at.dt.date)
    by_sym = {s: g for s, g in ev.groupby('symbol')}
    print(f'{len(ev)} filings joined for {ev.symbol.nunique()} of {len(syms)} symbols')

    def hit(r, pred) -> bool:
        lo = r.date - pd.Timedelta(days=PRE_ENTRY_DAYS)
        for sym in (r.lead, r.lag):
            g = by_sym.get(UNDERLYING.get(sym, sym))
            if g is None:
                continue
            w = g[(g.day > lo) & (g.day <= r.exit_date)]
            if any(pred(f, i) for f, i in zip(w.form, w['items'])):
                return True
        return False

    for name, pred in ITEM_GROUPS.items():
        df[name] = df.apply(hit, axis=1, pred=pred)
    return df


def _boot_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 7):
    """Bootstrap CI for rate(a) - rate(b)."""
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(df: pd.DataFrame) -> None:
    df['cat'] = df.gross < CATASTROPHIC_BPS
    print(f'\ntradeable observations: {len(df)} | catastrophic {df.cat.sum()} '
          f'({df.cat.mean() * 100:.1f}%)')
    print(f'\n=== Pooled discrimination per item group ===')
    print(f'{"group":12s} {"cat":>6s} {"rest":>6s} {"OR":>7s} {"p":>9s} {"CI(diff)":>18s} {"per-fold direction"}')
    for name in ITEM_GROUPS:
        cat, rest = df[df.cat][name].values, df[~df.cat][name].values
        tab = pd.crosstab(df.cat, df[name])
        orr, p = fisher_exact(tab) if tab.shape == (2, 2) else (np.nan, np.nan)
        lo, hi = _boot_ci(cat.astype(float), rest.astype(float))
        folds = []
        for fold, g in df.groupby('fold'):
            diff = g[g.cat][name].mean() - g[~g.cat][name].mean()
            folds.append(f'{fold[:4]}:{"+" if diff > 0 else "-" if diff < 0 else "0"}')
        print(f'{name:12s} {cat.mean() * 100:5.1f}% {rest.mean() * 100:5.1f}% '
              f'{orr:7.2f} {p:9.2g} [{lo * 100:+6.1f},{hi * 100:+6.1f}]pp  {" ".join(folds)}')

    print('\n=== Duration control: results-rate among non-catastrophic, by hold ===')
    nc = df[~df.cat]
    for lo_d, hi_d in [(1, 15), (16, 30), (31, 40)]:
        g = nc[(nc.hold_days >= lo_d) & (nc.hold_days <= hi_d)]
        if len(g):
            print(f'hold {lo_d:2d}-{hi_d:2d}d: n={len(g):4d} | results {g.results.mean() * 100:4.1f}%')
    g = df[df.cat]
    print(f'catastrophic: median hold {g.hold_days.median():.0f}d | results {g.results.mean() * 100:.1f}%')


def main() -> None:
    db = DatabaseClient(os.environ['DB_URL'])
    df = load_cache()
    print(f'cache: {len(df)} tradeable observations across {df.date.nunique()} dates')
    df = add_exit_dates(df, db)
    df = join_events(df, db)
    df.drop(columns=['ret_lead', 'ret_lag'], errors='ignore').to_parquet(OUT)
    report(df)
    db.close()


if __name__ == '__main__':
    main()
