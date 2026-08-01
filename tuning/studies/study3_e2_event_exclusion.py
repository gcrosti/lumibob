"""Study E2 — right-size the event-exclusion package (plan §5c).

RESULT: NO-GO (v1 here, v2 in _v2.py). Retained as the record of the runs and
as reusable path/policy machinery for follow-up analysis; do not treat the
policy below as a recommended design.

SUPERSEDED MECHANISM: the pre-event exit (parameter `L`) was dropped
2026-08-01 — it created a catastrophic trade on the final test and never
improved any grid cell. It remains implemented here only so the recorded runs
stay reproducible. New work should use H (entry veto) and R (reactive exit)
only, and should add the pre-entry limb of the veto that these runs lacked.

The decision to exclude event-exposed candidates is fixed by the plan; this
study selects the timing parameters and reports the honest net effect:

  * entry veto horizon H:  exclude a candidate whose either leg has a results
    event (8-K 2.02) within the next H trading days   — grid {10, 15, 20, 25}
  * pre-event exit lead L: a held pair exits L trading days before a results
    event that lands inside its natural hold          — grid {2, 5}
  * reactive exit R:       exit the day after a surprise filing (deal 1.01/2.01
    or restatement 4.02) on a held leg                — {off, on}

Proxy note: actual announcement dates stand in for the scheduled calendar
(companies pre-announce dates weeks out; live blind spots are exactly why the
pre-event exit exists). Only results events drive the veto — deals and
restatements are surprises and can only be reacted to.

Rigor (optuna-study skill):
  * unit = the tradeable pair-observation (2,043 across 12 scoring dates);
    grid of 16 configs over ~1,900 optimization units — complexity green.
  * E1 (gate passed) is the unit-level-signal prerequisite.
  * The LAST scoring date (2023-10-16) is the reserved FINAL TEST window:
    selection uses only the 11 earlier dates, and every selection date
    precedes final-test entry — the 40-td outcome horizon of the latest
    selection date (2023-09-15) does overlap the final date's window, so the
    embargo is imperfect; flagged in the report rather than hidden (the
    alternative — dropping 2023-09-15 from selection — is run as a check).
  * Preregistered selection rule: among configs whose optimization-set
    catastrophic loss magnitude is cut by >= 50%, pick the highest pooled
    mean; ties to the least-excluding config.
  * Guard panel: null baseline (per-symbol event-calendar shifts, 200 draws —
    the chosen config's mean improvement must beat the 90th percentile of
    null improvements), and fragility (re-select after dropping each fold —
    does the winner flip?).

Run:  DB_URL=... python -m tuning.studies.study3_e2_event_exclusion
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd

from DatabaseClient import DatabaseClient
from tuning.studies.scoring_replay import (
    EXIT_Z,
    HORIZON,
    HORIZON_CAL,
    LOOKBACK_CAL,
    LOOKBACK_WINDOW,
    ZSCORE_WINDOW,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_scoring_cache')
FOLDS = ('sideways_2022', 'bull_2023', 'mixed_2023_q4')
FINAL_TEST_DATE = pd.Timestamp('2023-10-16')
CATASTROPHIC_BPS = -100
TOL_BPS = 1.0
UNDERLYING = {'AAPB': 'AAPL', 'AAPU': 'AAPL', 'FBL': 'META'}

H_GRID = (10, 15, 20, 25)
L_GRID = (2, 5)
R_GRID = (False, True)

RESULTS = lambda form, items: form.startswith('8-K') and '2.02' in items
SURPRISE = lambda form, items: form.startswith('8-K') and (
    '1.01' in items or '2.01' in items or '4.02' in items)


# --------------------------------------------------------------------- paths

def build_paths(db: DatabaseClient) -> pd.DataFrame:
    """One row per tradeable observation with its full forward bps path and
    the trading-day indices of leg events, verified against the cache."""
    frames = []
    for fold in FOLDS:
        frames.append(pd.read_parquet(os.path.join(CACHE_DIR, f'{fold}.parquet')))
    cache = pd.concat(frames, ignore_index=True)
    cache = cache[np.isfinite(cache.forward_gross)].copy()
    cache['date'] = pd.to_datetime(cache.date)

    ev = db.get_filing_events(
        sorted({UNDERLYING.get(s, s) for s in set(cache.lead) | set(cache.lag)}),
        cache.date.min() - pd.Timedelta(days=5),
        cache.date.max() + pd.Timedelta(days=HORIZON_CAL))
    ev['items'] = ev['items'].fillna('')
    ev['day'] = pd.to_datetime(ev.filed_at.dt.date)
    by_sym = {s: g for s, g in ev.groupby('symbol')}

    rows, dropped = [], 0
    for (fold, T), grp in cache.groupby(['fold', 'date']):
        syms = sorted(set(grp.lead) | set(grp.lag))
        px = db.get_prices(syms, (T - timedelta(days=LOOKBACK_CAL)).to_pydatetime(),
                           (T + timedelta(days=HORIZON_CAL)).to_pydatetime())
        px.index = pd.to_datetime(px.index).normalize()
        px = px.astype(float)
        for _, r in grp.iterrows():
            p = _path_for(px, r.lead, r.lag, T)
            if p is None or abs(p['bps'][p['exit_t']] - r.forward_gross) > TOL_BPS:
                dropped += 1
                continue
            fwd_dates = p['dates']          # trading days 0..n of this pair
            res_days, sur_days = [], []
            for sym in (r.lead, r.lag):
                g = by_sym.get(UNDERLYING.get(sym, sym))
                if g is None:
                    continue
                w = g[(g.day > T) & (g.day <= fwd_dates[-1])]
                for form, items, day in zip(w.form, w['items'], w.day):
                    t = int(np.searchsorted(fwd_dates, day, side='left'))
                    if RESULTS(form, items):
                        res_days.append(t)
                    if SURPRISE(form, items):
                        sur_days.append(t)
            rows.append(dict(
                fold=fold, date=T, lead=r.lead, lag=r.lag,
                bps=p['bps'], exit_t=p['exit_t'],
                next_results_t=min(res_days) if res_days else None,
                surprise_t=min(sur_days) if sur_days else None))
    out = pd.DataFrame(rows)
    print(f'paths: {len(out)} verified, {dropped} dropped '
          f'({dropped / (len(out) + dropped) * 100:.1f}%)')
    return out


def _path_for(px, lead, lag, T):
    """scoring_replay's forward-outcome block, returning the full bps path."""
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
            np.log(pre[lead].clip(lower=1e-9)).to_numpy(),
            np.log(pre[lag].clip(lower=1e-9)).to_numpy(), 1)[0])
    except Exception:
        return None
    spread = (np.log(both[lag].clip(lower=1e-9))
              - hedge * np.log(both[lead].clip(lower=1e-9)))
    z = (spread - spread.rolling(ZSCORE_WINDOW).mean()) / spread.rolling(ZSCORE_WINDOW).std()
    z0 = z.iloc[pos]
    if not np.isfinite(z0):
        return None
    sgn, scale = np.sign(z0), 1 + abs(hedge)
    zf = z.iloc[pos:pos + HORIZON + 1].abs()
    sf = spread.iloc[pos:pos + HORIZON + 1]
    n = min(HORIZON, len(sf) - 1)
    if n < 1 or not np.isfinite(zf.iloc[0]):
        return None
    bps = (-sgn * (sf - sf.iloc[0]) / scale * 1e4).to_numpy()
    exit_t = n
    for t in range(1, n + 1):
        if zf.iloc[t] <= EXIT_Z:
            exit_t = t
            break
    return dict(bps=bps, exit_t=exit_t, dates=sf.index.to_numpy())


# -------------------------------------------------------------------- policy

def apply_policy(df: pd.DataFrame, H: int, L: int, R: bool) -> pd.DataFrame:
    """Returns df with columns: traded (bool), pol_gross (bps or NaN)."""
    traded, gross = [], []
    for _, r in df.iterrows():
        if r.next_results_t is not None and r.next_results_t <= H:
            traded.append(False)
            gross.append(np.nan)
            continue
        exit_t = r.exit_t
        if r.next_results_t is not None and r.next_results_t <= r.exit_t:
            exit_t = min(exit_t, max(1, int(r.next_results_t) - L))
        if R and r.surprise_t is not None and r.surprise_t <= exit_t:
            exit_t = min(len(r.bps) - 1, int(r.surprise_t) + 1)
        traded.append(True)
        gross.append(float(r.bps[min(exit_t, len(r.bps) - 1)]))
    out = df.copy()
    out['traded'] = traded
    out['pol_gross'] = gross
    return out


def metrics(pol: pd.DataFrame) -> dict:
    tr = pol[pol.traded]
    base = pol
    cat_base = base[base.apply(lambda r: r.bps[r.exit_t] < CATASTROPHIC_BPS, axis=1)]
    cat_loss_base = sum(r.bps[r.exit_t] for _, r in cat_base.iterrows())
    cat_tr = tr[tr.pol_gross < CATASTROPHIC_BPS]
    return dict(
        mean=tr.pol_gross.mean(), worst=tr.pol_gross.min() if len(tr) else np.nan,
        n_cat=len(cat_tr), cat_loss=cat_tr.pol_gross.sum(),
        cat_loss_cut=1 - (cat_tr.pol_gross.sum() / cat_loss_base if cat_loss_base else 0),
        excluded=1 - tr.shape[0] / len(pol),
        base_mean=np.mean([r.bps[r.exit_t] for _, r in base.iterrows()]))


def select(df: pd.DataFrame) -> tuple:
    """Preregistered rule: among configs cutting catastrophic loss magnitude
    >= 50%, max pooled mean; ties to least exclusion."""
    rows = []
    for H in H_GRID:
        for L in L_GRID:
            for R in R_GRID:
                m = metrics(apply_policy(df, H, L, R))
                rows.append(dict(H=H, L=L, R=R, **m))
    t = pd.DataFrame(rows)
    ok = t[t.cat_loss_cut >= 0.5]
    pick = (ok if len(ok) else t).sort_values(
        ['mean', 'excluded'], ascending=[False, True]).iloc[0]
    return (int(pick.H), int(pick.L), bool(pick.R)), t


# --------------------------------------------------------------------- main

def main() -> None:
    rng = np.random.default_rng(7)
    db = DatabaseClient(os.environ['DB_URL'])
    df = build_paths(db)
    db.close()

    opt = df[df.date != FINAL_TEST_DATE].copy()
    fin = df[df.date == FINAL_TEST_DATE].copy()
    print(f'optimization set: {len(opt)} obs on {opt.date.nunique()} dates | '
          f'final test: {len(fin)} obs on {FINAL_TEST_DATE.date()}')

    (H, L, R), table = select(opt)
    print('\n=== Config grid (optimization set) ===')
    print(table.round(3).to_string(index=False))
    print(f'\nSELECTED: H={H} L={L} reactive={R}')

    base = metrics(apply_policy(opt, 0, 99, False))  # H=0 => nothing vetoed
    chosen = metrics(apply_policy(opt, H, L, R))
    print(f'optimization set: baseline mean {base["base_mean"]:+.1f} -> '
          f'{chosen["mean"]:+.1f} | worst {base["worst"]:+.0f} -> {chosen["worst"]:+.0f} | '
          f'cat-loss cut {chosen["cat_loss_cut"] * 100:.0f}% | excluded {chosen["excluded"] * 100:.1f}%')

    # --- Guard: null baseline — shift each symbol's event calendar ---
    real_delta = chosen['mean'] - base['base_mean']
    null_deltas = []
    for _ in range(200):
        shifted = opt.copy()
        offs = {s: int(rng.integers(-120, 121)) for s in set(opt.lead) | set(opt.lag)}
        # shifting per-symbol event *day indices* by a trading-day offset;
        # events shifted out of the 40-day window vanish, as they would in
        # a genuinely shifted calendar
        def shift_row(r):
            o = min(offs[r.lead], offs[r.lag])
            nr = None if r.next_results_t is None else r.next_results_t + o
            ns = None if r.surprise_t is None else r.surprise_t + o
            return pd.Series(dict(
                next_results_t=nr if nr is not None and 0 < nr <= HORIZON else None,
                surprise_t=ns if ns is not None and 0 < ns <= HORIZON else None))
        shifted[['next_results_t', 'surprise_t']] = shifted.apply(shift_row, axis=1)
        null_deltas.append(metrics(apply_policy(shifted, H, L, R))['mean'] - base['base_mean'])
    p90 = float(np.percentile(null_deltas, 90))
    print(f'\nnull baseline: real mean-delta {real_delta:+.2f} vs null 90th pct {p90:+.2f} '
          f'-> {"PASS" if real_delta > p90 else "FAIL"}')

    # --- Guard: fragility — drop each fold, re-select ---
    flips = []
    for fold in FOLDS:
        (h2, l2, r2), _ = select(opt[opt.fold != fold])
        flips.append(f'-{fold}: H={h2} L={l2} R={r2}')
    print('fragility (drop-one-fold re-selection): ' + ' | '.join(flips))

    # --- Embargo check: re-select without the 2023-09-15 date ---
    (h3, l3, r3), _ = select(opt[opt.date != pd.Timestamp('2023-09-15')])
    print(f'embargo check (drop 2023-09-15): H={h3} L={l3} R={r3}')

    # --- FINAL TEST (scored once) ---
    fb = metrics(apply_policy(fin, 0, 99, False))
    fc = metrics(apply_policy(fin, H, L, R))
    print(f'\n=== FINAL TEST {FINAL_TEST_DATE.date()} (scored once) ===')
    print(f'baseline mean {fb["base_mean"]:+.1f} -> {fc["mean"]:+.1f} | '
          f'worst {fb["worst"]:+.0f} -> {fc["worst"]:+.0f} | '
          f'catastrophic {fb["n_cat"]} -> {fc["n_cat"]} | '
          f'cat-loss cut {fc["cat_loss_cut"] * 100:.0f}% | excluded {fc["excluded"] * 100:.1f}%')


if __name__ == '__main__':
    main()
