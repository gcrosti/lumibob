"""Disaster anatomy — what do catastrophic round trips share, and can any of
it be seen at entry?

Backs `docs/deepdives/2026-08-01_disasters-surviving-the-event-gate.md`.
Descriptive only: no fitting, no parameter selection, no gate. Every statistic
is a characterization of the replay pool, and the pool's observations are NOT
independent (see the leg-sharing section), so pooled intervals are reported
alongside date-clustered ones.

Sections:
  1  event-timing decomposition — why E2's veto missed what E1's flag caught
  2  surviving disasters vs normal: discovery features and path shape
  3  instrument composition (preferred / ETF / common)
  4  repeat-offender symbols and date concentration
  5  early-path drawdown as a (falsified) early warning
  6  ETF metadata coverage check
  7  leg sharing, unit independence, and E1's date-clustered CI

Prereqs: the replay cache (`_scoring_cache/<fold>.parquet`), E1's output
(`e1_events.parquet`), and the path cache (`e2_paths.parquet`, built here if
absent). Requires DB_URL and the SSH tunnel.

Run:  DB_URL=... python -m tuning.studies.study3_disaster_anatomy
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from DatabaseClient import DatabaseClient

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_scoring_cache')
PATHS = os.path.join(CACHE_DIR, 'e2_paths.parquet')
E1 = os.path.join(CACHE_DIR, 'e1_events.parquet')
KEYS = ['fold', 'date', 'lead', 'lag']
CATASTROPHIC_BPS = -100


def load(db: DatabaseClient) -> pd.DataFrame:
    if not os.path.exists(PATHS):
        from tuning.studies.study3_e2_event_exclusion import build_paths
        p = build_paths(db)
        p['gross'] = [r.bps[r.exit_t] for _, r in p.iterrows()]
        p['cat'] = p.gross < CATASTROPHIC_BPS
        p['max_dd'] = [float(r.bps[:r.exit_t + 1].min()) for _, r in p.iterrows()]
        p['bps_list'] = [list(map(float, r.bps)) for _, r in p.iterrows()]
        p.drop(columns=['bps']).to_parquet(PATHS)
    p = pd.read_parquet(PATHS)
    e1 = pd.read_parquet(E1)
    cols = KEYS + ['results', 'deals', 'exec_change', 'guidance', 'foreign',
                   'hold_days', 'corr_long', 'corr_short', 'z_depth', 'z_entry',
                   'coint_pvalue', 'halflife_days']
    return p.merge(e1[cols], on=KEYS, how='left')


def section_event_timing(df: pd.DataFrame) -> None:
    print('=' * 72)
    print('1 — Event timing: why the E2 veto missed what the E1 flag caught')
    print('=' * 72)
    base = df.gross.mean()
    print(f'pool n={len(df)} mean {base:+.1f} | catastrophic {df.cat.sum()}\n')

    def win(lo, hi):
        t = df.next_results_t
        return t.notna() & (t > lo) & (t <= hi)

    rows = [('pre-entry only', df.results.fillna(False) & df.next_results_t.isna()),
            ('t 1-10  (H=10 sees)', win(0, 10)),
            ('t 11-25 (H=25 sees)', win(10, 25)),
            ('t 26-40 (no H sees)', win(25, 40)),
            ('no results event', ~df.results.fillna(False) & df.next_results_t.isna())]
    print(f'{"timing":22s} {"n":>5s} {"mean":>9s} {"median":>8s} {"cat%":>6s}')
    for name, m in rows:
        g = df[m]
        if len(g):
            print(f'{name:22s} {len(g):5d} {g.gross.mean():+9.1f} '
                  f'{g.gross.median():+8.1f} {g.cat.mean() * 100:5.1f}%')
    print('\ncounterfactual pool mean after excluding each set:')
    for name, m in rows[:4]:
        k = df[~m]
        print(f'  {name:22s} ({m.sum():4d}) -> {k.gross.mean():+7.1f} '
              f'({k.gross.mean() - base:+.1f})')


def section_profile(df: pd.DataFrame) -> None:
    print('\n' + '=' * 72)
    print('2 — Surviving disasters vs normal trades')
    print('=' * 72)
    exposed = df.results.fillna(False) | (
        df.next_results_t.notna() & (df.next_results_t <= 40))
    cat = df[df.cat]
    surv, caught = cat[~exposed[cat.index]], cat[exposed[cat.index]]
    normal = df[~df.cat]
    tot = cat.gross.sum()
    print(f'catastrophic {len(cat)} | caught {len(caught)} '
          f'({caught.gross.sum() / tot * 100:.0f}% of loss) | '
          f'surviving {len(surv)} ({surv.gross.sum() / tot * 100:.0f}% of loss)')

    print(f'\n{"feature":15s} {"surviving":>12s} {"caught":>10s} {"normal":>10s}')
    for f in ['corr_long', 'corr_short', 'z_entry', 'coint_pvalue', 'halflife_days']:
        print(f'{f:15s} {surv[f].median():12.3f} {caught[f].median():10.3f} '
              f'{normal[f].median():10.3f}')
    print(f'\n{"group":12s} {"hold":>6s} {"max dd":>10s} {"cap-exit":>10s}')
    for name, g in [('surviving', surv), ('caught', caught), ('normal', normal)]:
        print(f'{name:12s} {g.hold_days.median():5.0f}d {g.max_dd.median():+10.0f} '
              f'{(g.hold_days >= 40).mean() * 100:9.0f}%')
    other = surv[['deals', 'exec_change', 'guidance', 'foreign']].any(axis=1)
    none_at_all = surv[~other]
    print(f'\ndisasters with NO filing of any kind: {len(none_at_all)} '
          f'({len(none_at_all) / len(cat) * 100:.0f}% of disasters, '
          f'{none_at_all.gross.sum() / tot * 100:.0f}% of loss)')


def section_instruments(df: pd.DataFrame, is_etf: dict) -> None:
    print('\n' + '=' * 72)
    print('3 — Instrument composition (knowable at entry)')
    print('=' * 72)

    def klass(s: str) -> str:
        if re.search(r'\.PR[A-Z]?$|\-P[A-Z]?$', s):
            return 'preferred'
        return 'etf' if is_etf.get(s, False) else 'common'

    k = df.assign(pk=[('+'.join(sorted([klass(a), klass(b)])))
                      for a, b in zip(df.lead, df.lag)])
    t = k.groupby('pk').agg(n=('gross', 'size'), mean=('gross', 'mean'),
                            median=('gross', 'median'),
                            cat_pct=('cat', lambda s: s.mean() * 100)).sort_values('mean')
    print(t.round(1).to_string())
    print('\nNOTE: ETF rows are unreliable while is_etf coverage is broken '
          '(see section 6).')


def section_concentration(df: pd.DataFrame) -> None:
    print('\n' + '=' * 72)
    print('4 — Repeat offenders and date concentration')
    print('=' * 72)
    legs = pd.concat([
        df[['lead', 'date', 'gross', 'cat']].rename(columns={'lead': 'sym'}),
        df[['lag', 'date', 'gross', 'cat']].rename(columns={'lag': 'sym'})])
    agg = legs.groupby('sym').agg(n=('gross', 'size'), mean=('gross', 'mean'),
                                  cat=('cat', 'sum')).query('n >= 8')
    print('worst 8 symbols by mean (>= 8 appearances):')
    print(agg.sort_values('mean').head(8).round(1).to_string())
    print('\nper-date breakdown of the worst symbol:')
    worst = agg.sort_values('mean').index[0]
    print(legs[legs.sym == worst].groupby('date').agg(
        n=('gross', 'size'), cat=('cat', 'sum'), mean=('gross', 'mean')).round(0).to_string())

    tot = df[df.cat].gross.sum()
    d = df.groupby('date').agg(n=('gross', 'size'), mean=('gross', 'mean'),
                               cat=('cat', 'sum'))
    d['loss_share_%'] = df[df.cat].groupby(df.date).gross.sum().reindex(
        d.index).fillna(0) / tot * 100
    print('\nper-date:')
    print(d.round(1).to_string())
    w3 = d.nlargest(3, 'loss_share_%')
    print(f'worst 3 dates carry {w3["loss_share_%"].sum():.0f}% of all disaster loss')


def section_early_path(df: pd.DataFrame) -> None:
    print('\n' + '=' * 72)
    print('5 — Early drawdown (characterization only; stops falsified 2026-07-18)')
    print('=' * 72)
    for day in (5, 10):
        df[f'dd{day}'] = [min(b[:day + 1]) if len(b) > day else np.nan
                          for b in df.bps_list]
    for day in (5, 10):
        for thr in (-100, -300):
            s = df[df[f'dd{day}'] <= thr]
            if len(s) > 5:
                print(f'  dd{day:<2d} <= {thr:5d}: n={len(s):4d} '
                      f'P(disaster)={s.cat.mean() * 100:5.1f}% '
                      f'(base {df.cat.mean() * 100:.1f}%) mean {s.gross.mean():+8.1f}')


def section_metadata(db: DatabaseClient, df: pd.DataFrame) -> None:
    print('\n' + '=' * 72)
    print('6 — ETF metadata coverage (gates the live clustering partition)')
    print('=' * 72)
    syms = sorted(set(df.lead) | set(df.lag))
    md = db.get_ticker_metadata(syms)
    n_etf = int(md.is_etf.astype(bool).sum()) if len(md) else 0
    print(f'symbols {len(syms)} | metadata rows {len(md)} | is_etf True {n_etf}')
    known = ['QQQ', 'TQQQ', 'PTNQ', 'ROBT', 'SDVY', 'MILN', 'PHO', 'KBWY']
    if len(md):
        print('\nspot-check of symbols that ARE ETFs:')
        print(md[md.symbol.isin(known)][['symbol', 'is_etf', 'sector']].to_string(index=False))
    if n_etf < 0.1 * len(syms):
        print('\n*** is_etf coverage is broken -> TickerClusterer ETF partition '
              'is largely non-functional ***')


def section_independence(df: pd.DataFrame, seed: int = 11) -> None:
    print('\n' + '=' * 72)
    print('7 — Leg sharing, unit independence, and E1 re-tested by date')
    print('=' * 72)
    rows = []
    for d, g in df.groupby('date'):
        legs = pd.concat([g.lead, g.lag]).value_counts()
        rows.append(dict(date=d.date(), pairs=len(g), symbols=legs.size,
                         max_reuse=int(legs.max())))
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nBobsBrain skips candidates whose either leg is already held, so a '
          'LIVE book cannot repeat a symbol this way; the replay pool has no '
          'such dedup -> these observations are not independent units.')

    rng = np.random.default_rng(seed)
    dates = df.date.unique()

    def diff(sub):
        c, r = sub[sub.cat], sub[~sub.cat]
        if len(c) < 5 or len(r) < 5:
            return np.nan
        return c.results.mean() - r.results.mean()

    boot = [diff(pd.concat([df[df.date == d] for d in rng.choice(dates, len(dates))]))
            for _ in range(2000)]
    boot = [b for b in boot if np.isfinite(b)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f'\nE1 results-rate difference: {diff(df) * 100:+.1f}pp')
    print(f'  date-clustered 95% CI: {lo * 100:+.1f} .. {hi * 100:+.1f} pp '
          f'({"excludes 0" if lo > 0 else "INCLUDES 0"})')
    print('  (E1 reported +15.6 .. +27.0 pp using the pair as the unit)')


def main() -> None:
    db = DatabaseClient(os.environ['DB_URL'])
    df = load(db)
    syms = sorted(set(df.lead) | set(df.lag))
    md = db.get_ticker_metadata(syms)
    is_etf = dict(zip(md.symbol, md.is_etf.astype(bool))) if len(md) else {}

    section_event_timing(df)
    section_profile(df)
    section_instruments(df, is_etf)
    section_concentration(df)
    section_early_path(df)
    section_metadata(db, df)
    section_independence(df)
    db.close()


if __name__ == '__main__':
    main()
