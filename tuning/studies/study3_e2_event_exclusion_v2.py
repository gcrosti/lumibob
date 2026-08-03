"""Study E2 v2 — event exclusion on the score-selected book (plan §5c).

Version bump per the optuna-study skill: v1 (study3_e2_event_exclusion.py)
ran on the FULL dislocated-candidate pool and FAILED its gates — no config
cut catastrophic loss magnitude >= 50% (best 29.7%), the null baseline failed
(mean effect indistinguishable from random pruning), and selection was
fold-fragile. Diagnosis: the full pool is the wrong universe — the strategy
holds a top-K composite-ranked book, and the event concentration of disasters
was established on *entered* pairs.

v2 preregistration (before running):
  * universe: top-K per scoring date by the live 3-component composite
    (0.3*corr_long + 0.5*corr_short + 0.2*z_depth, correlations clamped to
    [0,1]), K = 20 (~ live max_k);
  * everything else unchanged from v1: same grid, same selection rule
    (>= 50% catastrophic-loss cut, then max mean, ties to least exclusion),
    same guards (null calendar-shift baseline, drop-one-fold fragility),
    same reserved final-test date (2023-10-16), scored once.

Run:  DB_URL=... python -m tuning.studies.study3_e2_event_exclusion_v2
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from DatabaseClient import DatabaseClient
from tuning.studies.scoring_replay import HORIZON
from tuning.studies.study3_e2_event_exclusion import (
    CACHE_DIR,
    FINAL_TEST_DATE,
    FOLDS,
    apply_policy,
    build_paths,
    metrics,
    select,
)

K = 20


def composite(row) -> float:
    cl = min(max(row.corr_long, 0.0), 1.0) if np.isfinite(row.corr_long) else 0.0
    cs = min(max(row.corr_short, 0.0), 1.0) if np.isfinite(row.corr_short) else 0.0
    return 0.3 * cl + 0.5 * cs + 0.2 * row.z_depth


def main() -> None:
    rng = np.random.default_rng(7)
    db = DatabaseClient(os.environ['DB_URL'])
    paths = build_paths(db)
    db.close()

    # Score every observation with the live composite and keep the top-K book
    # per scoring date (the pairs the strategy would actually consider holding).
    frames = []
    for fold in FOLDS:
        frames.append(pd.read_parquet(os.path.join(CACHE_DIR, f'{fold}.parquet')))
    cache = pd.concat(frames, ignore_index=True)
    cache['date'] = pd.to_datetime(cache.date)
    cache['score'] = cache.apply(composite, axis=1)
    df = paths.merge(cache[['fold', 'date', 'lead', 'lag', 'score']],
                     on=['fold', 'date', 'lead', 'lag'], how='left')
    df = (df.sort_values('score', ascending=False)
            .groupby('date', group_keys=False).head(K).reset_index(drop=True))
    n_cat = sum(r.bps[r.exit_t] < -100 for _, r in df.iterrows())
    print(f'top-{K} book: {len(df)} obs across {df.date.nunique()} dates | '
          f'catastrophic {n_cat}')

    opt = df[df.date != FINAL_TEST_DATE].copy()
    fin = df[df.date == FINAL_TEST_DATE].copy()

    (H, L, R), table = select(opt)
    print('\n=== Config grid (optimization set, top-K book) ===')
    print(table.round(3).to_string(index=False))
    print(f'\nSELECTED: H={H} L={L} reactive={R}')

    base = metrics(apply_policy(opt, 0, 99, False))
    chosen = metrics(apply_policy(opt, H, L, R))
    print(f'optimization set: baseline mean {base["base_mean"]:+.1f} -> '
          f'{chosen["mean"]:+.1f} | worst {base["worst"]:+.0f} -> {chosen["worst"]:+.0f} | '
          f'cat-loss cut {chosen["cat_loss_cut"] * 100:.0f}% | '
          f'excluded {chosen["excluded"] * 100:.1f}%')

    # Guard: null baseline (same construction as v1)
    real_delta = chosen['mean'] - base['base_mean']
    null_deltas = []
    for _ in range(200):
        shifted = opt.copy()
        offs = {s: int(rng.integers(-120, 121)) for s in set(opt.lead) | set(opt.lag)}

        def shift_row(r):
            o = min(offs[r.lead], offs[r.lag])
            nr = None if r.next_results_t is None else r.next_results_t + o
            ns = None if r.surprise_t is None else r.surprise_t + o
            return pd.Series(dict(
                next_results_t=nr if nr is not None and 0 < nr <= HORIZON else None,
                surprise_t=ns if ns is not None and 0 < ns <= HORIZON else None))
        shifted[['next_results_t', 'surprise_t']] = shifted.apply(shift_row, axis=1)
        null_deltas.append(
            metrics(apply_policy(shifted, H, L, R))['mean'] - base['base_mean'])
    p90 = float(np.percentile(null_deltas, 90))
    print(f'\nnull baseline: real mean-delta {real_delta:+.2f} vs null 90th pct '
          f'{p90:+.2f} -> {"PASS" if real_delta > p90 else "FAIL"}')

    flips = []
    for fold in FOLDS:
        (h2, l2, r2), _ = select(opt[opt.fold != fold])
        flips.append(f'-{fold}: H={h2} L={l2} R={r2}')
    print('fragility (drop-one-fold re-selection): ' + ' | '.join(flips))

    fb = metrics(apply_policy(fin, 0, 99, False))
    fc = metrics(apply_policy(fin, H, L, R))
    print(f'\n=== FINAL TEST {FINAL_TEST_DATE.date()} (scored once) ===')
    print(f'baseline mean {fb["base_mean"]:+.1f} -> {fc["mean"]:+.1f} | '
          f'worst {fb["worst"]:+.0f} -> {fc["worst"]:+.0f} | '
          f'catastrophic {fb["n_cat"]} -> {fc["n_cat"]} | '
          f'cat-loss cut {fc["cat_loss_cut"] * 100:.0f}% | '
          f'excluded {fc["excluded"] * 100:.1f}%')


if __name__ == '__main__':
    main()
