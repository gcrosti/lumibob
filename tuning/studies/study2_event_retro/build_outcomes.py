"""Study 2 (event retro), step 1 of 3: rebuild per-pair round-trip outcomes.

Replays the pass-a-v3 notebook's per-pair methodology (see
notebooks/pass_a_v3_score_signal_retroactive.ipynb) for the three gate runs,
keeping calendar entry/exit dates so downstream steps can join event data:
hedge frozen on pre-entry data, |z| >= 1 entry-fidelity filter, and the
baseline z <= 0.5 take-profit exit capped at 40 trading days.

Output: _event_cache/pair_outcomes.parquet — one row per analyzable round trip
with fold, legs, entry/exit dates, holding days, and gross bps.

Requires the SSH tunnel to the cloud DB (see CLAUDE.md) and DB_URL.
"""
import os

import numpy as np
import pandas as pd
import psycopg2

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_event_cache')
OUT = os.path.join(CACHE_DIR, 'pair_outcomes.parquet')
DB_URL = os.getenv('DB_URL', 'postgresql://lumibob:lumibob@localhost:5433/lumibob')

RUNS = {
    '4f419e': dict(fold='sideways_2022', start='2021-12-01', end='2022-04-30'),
    '4b26c6': dict(fold='bull_2023',     start='2023-02-01', end='2023-06-30'),
    'bcb308': dict(fold='mixed_2023_q4', start='2023-07-01', end='2023-11-30'),
}
LOOKBACK_CAL = 152   # calendar days of pre-entry history (hedge + z warm-up)
ZSCORE_WINDOW = 32
EXIT_Z = 0.5         # baseline take-profit exit; matches the notebook's 1g tail analysis
CAP = 40             # max trading days held


def load_entered_pairs(conn) -> pd.DataFrame:
    pairs = pd.read_sql('''
        SELECT p.run_id, p.id AS pair_id, p.lead_symbol, p.lag_symbol,
               p.composite_score, p.coint_pvalue, p.halflife_days,
               e.entry_date
        FROM pairs p
        JOIN LATERAL (
            SELECT MIN(t.filled_at)::date AS entry_date
            FROM trades t
            WHERE t.pair_id = p.id AND t.side = 'buy' AND t.leg = 'long'
        ) e ON e.entry_date IS NOT NULL
        WHERE p.run_id IN %(runs)s
          AND p.composite_score IS NOT NULL
    ''', conn, params=dict(runs=tuple(RUNS)))
    pairs['fold'] = pairs['run_id'].map({k: v['fold'] for k, v in RUNS.items()})
    return pairs


def load_prices(conn, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prices = {}
    for rid, meta in RUNS.items():
        sub = pairs[pairs.run_id == rid]
        syms = sorted(set(sub.lead_symbol) | set(sub.lag_symbol))
        lo = (pd.Timestamp(meta['start']) - pd.Timedelta(days=LOOKBACK_CAL + 30)).date()
        hi = (pd.Timestamp(meta['end']) + pd.Timedelta(days=90)).date()
        df = pd.read_sql(
            'SELECT time::date AS d, symbol, close FROM stock_prices '
            'WHERE symbol = ANY(%(syms)s) AND time >= %(lo)s AND time <= %(hi)s',
            conn, params=dict(syms=syms, lo=lo, hi=hi))
        px = df.pivot_table(index='d', columns='symbol', values='close', aggfunc='last')
        px.index = pd.to_datetime(px.index)
        prices[rid] = px.astype(float)
    return prices


def round_trip(px: pd.DataFrame, lead: str, lag: str, entry_date) -> dict | None:
    """One pair's simulated round trip (z0.5 take-profit, 40-day cap)."""
    if lead not in px.columns or lag not in px.columns:
        return None
    both = px[[lead, lag]].dropna()
    if len(both) < ZSCORE_WINDOW + 10:
        return None
    entry_ts = pd.Timestamp(entry_date)
    pre = both.loc[entry_ts - pd.Timedelta(days=LOOKBACK_CAL):entry_ts]
    if len(pre) < ZSCORE_WINDOW + 5 or entry_ts not in both.index:
        return None
    ll = np.log(pre[lead].clip(lower=1e-9))
    lg = np.log(pre[lag].clip(lower=1e-9))
    try:
        hedge = float(np.polyfit(ll.values, lg.values, 1)[0])
    except Exception:
        return None
    seg = both.loc[entry_ts - pd.Timedelta(days=LOOKBACK_CAL):]
    spread = np.log(seg[lag].clip(lower=1e-9)) - hedge * np.log(seg[lead].clip(lower=1e-9))
    z = (spread - spread.rolling(ZSCORE_WINDOW).mean()) / spread.rolling(ZSCORE_WINDOW).std()
    i0 = z.index.get_loc(entry_ts)
    z0 = z.iloc[i0]
    if not np.isfinite(z0) or abs(z0) < 1.0:   # fidelity filter: strategy entered at |z|>=2
        return None
    zf = z.iloc[i0:i0 + CAP + 1].abs()
    fav = -np.sign(z0) * (spread.iloc[i0:i0 + CAP + 1] - spread.iloc[i0])
    n = min(CAP, len(fav) - 1)
    if n < 1:
        return None
    exit_t, reason = n, 'cap'
    for t in range(1, n + 1):
        if zf.iloc[t] <= EXIT_Z:
            exit_t, reason = t, 'take_profit'
            break
    gross = fav.iloc[exit_t] / (1 + abs(hedge)) * 1e4
    return dict(entry_date=entry_ts, exit_date=fav.index[exit_t], hold_days=exit_t,
                reason=reason, z0=float(z0), gross=float(gross))


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    conn = psycopg2.connect(DB_URL)
    pairs = load_entered_pairs(conn)
    prices = load_prices(conn, pairs)
    rows = []
    for _, p in pairs.iterrows():
        m = round_trip(prices[p.run_id], p.lead_symbol, p.lag_symbol, p.entry_date)
        if m is not None:
            rows.append(dict(run_id=p.run_id, fold=p.fold, pair_id=p.pair_id,
                             lead=p.lead_symbol, lag=p.lag_symbol,
                             composite_score=p.composite_score, **m))
    out = pd.DataFrame(rows)
    out.to_parquet(OUT)
    print(f'analyzable round trips: {len(out)} of {len(pairs)} entered -> {OUT}')
    for fold, g in out.groupby('fold'):
        cat = g.gross < -100
        print(f'{fold:14s}: mean {g.gross.mean():+6.1f} | median {g.gross.median():+5.1f} | '
              f'worst {g.gross.min():+7.0f} | catastrophic(<-100) {cat.mean() * 100:4.1f}% ({cat.sum()})')


if __name__ == '__main__':
    main()
