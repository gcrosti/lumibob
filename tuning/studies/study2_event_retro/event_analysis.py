"""Study 2 (event retro), step 3 of 3: earnings exposure vs catastrophic losses.

Joins the round trips from build_outcomes.py with the earnings dates from
fetch_earnings.py and reports:
  1. pair composition of the catastrophic class (stock-legged vs fund-only),
  2. earnings-in-window exposure rates, catastrophic vs rest, with Fisher tests,
  3. a duration-control cut (longer holds mechanically catch more earnings),
  4. the veto-counterfactual ledger: losses avoided vs wins missed, per fold.

Catastrophic = gross < -100 bps. An event counts if either leg (or a mapped
single-stock-ETF underlying) announces inside (entry, exit] or in the 7 calendar
days before entry. Symbols with gapped earnings history are treated as unknown
coverage, not as event-free.

Findings write-up: docs/deepdives/2026-07-31_earnings-events-catastrophic-losses.md
"""
import os

import pandas as pd
from scipy.stats import fisher_exact

from fetch_earnings import UNDERLYING

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_event_cache')
OUT = os.path.join(CACHE_DIR, 'pair_outcomes_events.parquet')

CATASTROPHIC_BPS = -100
PRE_ENTRY_DAYS = 7
# yfinance history for these starts after the study window — coverage unknown.
GAPPED = {'LBTYK', 'BATRA'}


def load() -> tuple[pd.DataFrame, dict[str, list[pd.Timestamp]]]:
    out = pd.read_parquet(os.path.join(CACHE_DIR, 'pair_outcomes.parquet'))
    earn = pd.read_csv(os.path.join(CACHE_DIR, 'earnings_dates.csv'))
    earn = earn[earn.earn_date.notna()].copy()
    earn['earn_date'] = pd.to_datetime(earn.earn_date)
    return out, earn.groupby('symbol')['earn_date'].apply(list).to_dict()


def classify(r, edates) -> pd.Series:
    hold_hits, pre_hits, n_stock = [], [], 0
    for sym in (r.lead, r.lag):
        sym = UNDERLYING.get(sym, sym)
        if sym in GAPPED or sym not in edates:
            continue
        n_stock += 1
        for d in edates[sym]:
            if r.entry_date < d <= r.exit_date:
                hold_hits.append(f'{sym}@{d.date()}')
            elif r.entry_date - pd.Timedelta(days=PRE_ENTRY_DAYS) <= d <= r.entry_date:
                pre_hits.append(f'{sym}@{d.date()}')
    return pd.Series(dict(n_stock_legs=n_stock,
                          ev_hold=len(hold_hits) > 0, ev_pre=len(pre_hits) > 0,
                          hits='; '.join(hold_hits + [p + '(pre)' for p in pre_hits])))


def main() -> None:
    out, edates = load()
    out = pd.concat([out, out.apply(classify, axis=1, edates=edates)], axis=1)
    out['ev_any'] = out.ev_hold | out.ev_pre
    out['cat'] = out.gross < CATASTROPHIC_BPS
    out['has_stock'] = out.n_stock_legs > 0
    out.to_parquet(OUT)

    print(f'total {len(out)} | catastrophic {out.cat.sum()} | '
          f'with >=1 stock leg {out.has_stock.sum()}')

    print('\n=== Pair composition: catastrophic vs rest ===')
    for is_cat, g in out.groupby('cat'):
        lbl = 'CATASTROPHIC' if is_cat else 'rest        '
        print(f'{lbl}: n={len(g):3d} | >=1 stock leg {g.has_stock.mean() * 100:4.0f}% | '
              f'both stock {(g.n_stock_legs == 2).mean() * 100:4.0f}%')

    print('\n=== Earnings exposure (all pairs; fund-only pairs count as no-event) ===')
    for is_cat, g in out.groupby('cat'):
        lbl = 'CATASTROPHIC' if is_cat else 'rest        '
        print(f'{lbl}: ev_hold {g.ev_hold.mean() * 100:4.0f}% | ev_pre {g.ev_pre.mean() * 100:4.0f}% | '
              f'ev_any {g.ev_any.mean() * 100:4.0f}%')
    tab = pd.crosstab(out.cat, out.ev_any)
    print(f'Fisher exact (ev_any x catastrophic): OR={fisher_exact(tab)[0]:.2f} '
          f'p={fisher_exact(tab)[1]:.4f}')

    print('\n=== Conditional on >=1 stock leg (the fair comparison) ===')
    st = out[out.has_stock]
    for is_cat, g in st.groupby('cat'):
        lbl = 'CATASTROPHIC' if is_cat else 'rest        '
        print(f'{lbl}: n={len(g):3d} | ev_hold {g.ev_hold.mean() * 100:4.0f}% | '
              f'ev_any {g.ev_any.mean() * 100:4.0f}% | median hold {g.hold_days.median():.0f}d')
    tab2 = pd.crosstab(st.cat, st.ev_any)
    if tab2.shape == (2, 2):
        print(f'Fisher exact (stock-legged only): OR={fisher_exact(tab2)[0]:.2f} '
              f'p={fisher_exact(tab2)[1]:.4f}')

    print('\n=== Duration control: ev_hold rate by hold length (stock-legged, non-cat) ===')
    nc = st[~st.cat]
    for lo, hi in [(1, 15), (16, 30), (31, 40)]:
        g = nc[(nc.hold_days >= lo) & (nc.hold_days <= hi)]
        if len(g):
            print(f'hold {lo:2d}-{hi:2d}d: n={len(g):3d} | ev_hold {g.ev_hold.mean() * 100:4.0f}%')

    print('\n=== The catastrophic pairs, annotated ===')
    for _, r in out[out.cat].sort_values('gross').iterrows():
        note = ('EVENT: ' + r.hits if r.hits
                else ('no earnings in window' if r.n_stock_legs else 'fund-only pair'))
        print(f'{r.fold:14s} {r.lead:>6s}/{r.lag:<6s} {r.gross:+7.0f}  '
              f'{r.n_stock_legs}stk  {note}')

    print('\n=== Veto counterfactual: losses avoided vs wins missed ===')
    ev = out[out.ev_any]
    neg, pos = ev[ev.gross < 0], ev[ev.gross >= 0]
    print(f'losses avoided : {neg.gross.sum():+8.1f} bps across {len(neg)} pairs')
    print(f'wins missed    : {pos.gross.sum():+8.1f} bps across {len(pos)} pairs')
    print(f'NET            : {-ev.gross.sum():+8.1f} bps '
          f'({-ev.gross.sum() / len(out):+.1f} bps per trade over {len(out)} trades)')
    for fold, g in out.groupby('fold'):
        e = g[g.ev_any]
        print(f'{fold:14s}: mean before {g.gross.mean():+6.1f} | '
              f'after veto {g[~g.ev_any].gross.mean():+6.1f} | dropped {len(e)}')


if __name__ == '__main__':
    main()
