"""Study 2 (event retro), EDGAR upgrade: item-coded filings vs catastrophic losses.

Re-runs the event join of event_analysis.py with filing_events (EDGAR 8-K item
codes + 6-K) in place of the yfinance earnings calendar, using the item groups
preregistered in plan Study E1:

    results     -- 8-K item 2.02 (results of operations, incl. interim updates)
    deals       -- 8-K items 1.01 / 2.01 (material agreements / M&A)
    exec_change -- 8-K item 5.02
    restatement -- 8-K item 4.02
    guidance    -- 8-K item 7.01 (Reg FD)
    foreign     -- any 6-K (no item codes; noisy — includes routine filings)

This is the *entered-pairs* dry run (n = 175, 14 disasters); the full Study E1
runs the same groups over the replay pool. Interpretation caveat: 8-K/7.01 and
6-K are high-frequency forms, so their in-window base rates are elevated for
everyone — separation, not raw rate, is the readout.
"""
import os

import pandas as pd
import psycopg2
from scipy.stats import fisher_exact

from fetch_earnings import UNDERLYING

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_event_cache')
DB_URL = os.getenv('DB_URL', 'postgresql://lumibob:lumibob@localhost:5433/lumibob')

CATASTROPHIC_BPS = -100
PRE_ENTRY_DAYS = 7

ITEM_GROUPS = {
    'results': lambda form, items: form.startswith('8-K') and '2.02' in items,
    'deals': lambda form, items: form.startswith('8-K') and ('1.01' in items or '2.01' in items),
    'exec_change': lambda form, items: form.startswith('8-K') and '5.02' in items,
    'restatement': lambda form, items: form.startswith('8-K') and '4.02' in items,
    'guidance': lambda form, items: form.startswith('8-K') and '7.01' in items,
    'foreign': lambda form, items: form == '6-K',
}


def main() -> None:
    out = pd.read_parquet(os.path.join(CACHE_DIR, 'pair_outcomes.parquet'))
    out['cat'] = out.gross < CATASTROPHIC_BPS

    syms = sorted({UNDERLYING.get(s, s) for s in set(out.lead) | set(out.lag)})
    conn = psycopg2.connect(DB_URL)
    ev = pd.read_sql(
        'SELECT symbol, form, items, filed_at FROM filing_events '
        'WHERE symbol = ANY(%(s)s)', conn, params=dict(s=syms))
    ev['items'] = ev['items'].fillna('')
    ev['day'] = pd.to_datetime(ev.filed_at.dt.date)
    by_sym = {s: g for s, g in ev.groupby('symbol')}
    print(f'{len(ev)} filings for {ev.symbol.nunique()} of {len(syms)} symbols')

    def hit(r, pred) -> bool:
        for sym in (r.lead, r.lag):
            g = by_sym.get(UNDERLYING.get(sym, sym))
            if g is None:
                continue
            lo = r.entry_date - pd.Timedelta(days=PRE_ENTRY_DAYS)
            w = g[(g.day > lo) & (g.day <= r.exit_date)]
            if any(pred(f, i) for f, i in zip(w.form, w['items'])):
                return True
        return False

    print(f'\n{"group":12s} {"cat rate":>9s} {"rest rate":>10s} {"odds ratio":>11s} {"p":>8s}')
    for name, pred in ITEM_GROUPS.items():
        out[name] = out.apply(hit, axis=1, pred=pred)
        tab = pd.crosstab(out.cat, out[name])
        if tab.shape == (2, 2):
            orr, p = fisher_exact(tab)
            print(f'{name:12s} {out[out.cat][name].mean()*100:8.0f}% '
                  f'{out[~out.cat][name].mean()*100:9.0f}% {orr:11.2f} {p:8.4f}')
        else:
            print(f'{name:12s} {out[out.cat][name].mean()*100:8.0f}% '
                  f'{out[~out.cat][name].mean()*100:9.0f}%       (no variation)')

    print('\n=== Catastrophic pairs: which groups fired ===')
    for _, r in out[out.cat].sort_values('gross').iterrows():
        fired = [g for g in ITEM_GROUPS if r[g]]
        print(f'{r.fold:14s} {r.lead:>6s}/{r.lag:<6s} {r.gross:+7.0f}  {fired or "none"}')

    out.to_parquet(os.path.join(CACHE_DIR, 'pair_outcomes_edgar_events.parquet'))


if __name__ == '__main__':
    main()
