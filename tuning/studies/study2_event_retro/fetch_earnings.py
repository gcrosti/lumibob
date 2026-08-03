"""Study 2 (event retro), step 2 of 3: fetch earnings announcement dates.

Pulls historical earnings dates via yfinance for every symbol appearing in the
round trips from build_outcomes.py, plus the underlyings of known single-stock
leveraged ETFs. Results are cached to _event_cache/earnings_dates.csv and the
fetch is resumable (already-fetched symbols are skipped).

Note: this is the quick-prototype source. Most of the universe is ETFs and
closed-end funds with no earnings — a symbol yielding no dates is recorded with
a null earn_date so coverage is explicit. yfinance also misses interim results
disclosures (see the RGLD case in the 2026-07-31 deepdive); the production
source is the EDGAR 8-K feed (plan WS2a).
"""
import os
import time

import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_event_cache')
OUTCOMES = os.path.join(CACHE_DIR, 'pair_outcomes.parquet')
CACHE = os.path.join(CACHE_DIR, 'earnings_dates.csv')

# Single-stock leveraged ETFs in the universe: the fund has no earnings, but its
# underlying's announcement moves it exactly like a stock leg.
UNDERLYING = {'AAPB': 'AAPL', 'AAPU': 'AAPL', 'FBL': 'META'}
LIMIT = 28          # quarterly reports -> ~7 years of history
SLEEP_S = 0.4       # be polite to Yahoo


def main() -> None:
    out = pd.read_parquet(OUTCOMES)
    syms = sorted(set(out.lead) | set(out.lag) | set(UNDERLYING.values()))
    done: set[str] = set()
    if os.path.exists(CACHE):
        done = set(pd.read_csv(CACHE).symbol.unique())
    todo = [s for s in syms if s not in done]
    print(f'{len(syms)} symbols total, {len(todo)} to fetch')

    rows = []
    for i, s in enumerate(todo):
        try:
            ed = yf.Ticker(s).get_earnings_dates(limit=LIMIT)
        except Exception:
            ed = None
        if ed is None or len(ed) == 0:
            rows.append(dict(symbol=s, earn_date=None))
        else:
            rows.extend(dict(symbol=s, earn_date=pd.Timestamp(d).date()) for d in ed.index)
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(todo)}')
        time.sleep(SLEEP_S)

    new = pd.DataFrame(rows)
    if os.path.exists(CACHE):
        new = pd.concat([pd.read_csv(CACHE), new], ignore_index=True)
    new.to_csv(CACHE, index=False)

    ok = new[new.earn_date.notna()]
    print(f'saved {len(ok)} dates for {ok.symbol.nunique()} symbols '
          f'({new.symbol.nunique() - ok.symbol.nunique()} symbols with no earnings data)')
    # A symbol whose earliest fetched date is after the study start has a silent
    # gap (yfinance history simply stops) — downstream must treat it as unknown,
    # not as event-free.
    first = ok.assign(d=pd.to_datetime(ok.earn_date)).groupby('symbol')['d'].min()
    gaps = first[first > pd.Timestamp('2021-10-01')]
    print(f'symbols with possible history gaps (earliest date after 2021-10-01): '
          f'{sorted(gaps.index.tolist())}')


if __name__ == '__main__':
    main()
