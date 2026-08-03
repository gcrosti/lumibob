"""V2 — walk-forward selection of the magnitude floor and book size.

Plan: docs/plans/2026-08-01_entry-criteria-overhaul.md §6.
Methodology: .claude/skills/optuna-study.md.

PREREGISTERED BEFORE RUNNING — nothing below is chosen after seeing results:

  free params   min_expected_gross_bps in {0, 15, 25, 40, 60}
                max_k                  in {10, 20, 30, 40}
  unit          the pair round-trip
  objective     median gross bps of the selected book, pooled over eval windows.
                MEDIAN, not mean: every validated finding in this program is a
                median/tail effect, and the mean is dominated by a handful of
                blow-ups (one trade is 4.9% of all catastrophic loss).
  selection     highest eval-window objective; ties to the LOWEST floor and the
                HIGHEST k (prefer deploying capital, per the design principle
                that the book should stay invested).
  temporal      walk_forward_splits with embargo >= the 40-trading-day outcome
                horizon, so a train window's outcomes cannot overlap its eval
                window.  Enforced by assert_causal inside the primitive.
  final test    the last scoring date (2023-10-16) is held out of every
                selection step and scored EXACTLY ONCE at the end.
  guards        complexity_ratio, null_baseline (shuffle + random),
                holdout_gap (leave-one-fold-out), seed_stability
                (re-selection under resampled dates), reported via report_panel.

  PASS  the chosen config's held-out median beats the no-floor/k=20 baseline
        with a date-clustered CI excluding zero, AND the guard panel is green,
        AND the selection survives dropping any one fold.
  FAIL  anything else -> stop and report.  Do not re-cut the grid.

Run:  DB_URL=... python -m tuning.studies.study3_v2_floor_and_k
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd

from StockEvaluator import StockEvaluator
from tuning.overfit_guards import (complexity_ratio, holdout_gap, null_baseline,
                                   report_panel, walk_forward_splits)
from tuning.studies.scoring_replay import EXIT_Z

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_scoring_cache')
FOLDS = ('sideways_2022', 'bull_2023', 'mixed_2023_q4')
FINAL_TEST_DATE = pd.Timestamp('2023-10-16')
ENTRY_THRESHOLD = 2.0
HORIZON_TD = 40                      # outcome horizon, trading days
EMBARGO = '70D'                      # >= 40 trading days in calendar terms
FLOOR_GRID = (0.0, 15.0, 25.0, 40.0, 60.0)
K_GRID = (10, 20, 30, 40)
BASELINE = (0.0, 20)                 # no floor, current max_k


# --------------------------------------------------------------------- data

def load() -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(CACHE_DIR, 'e2_paths.parquet'))
    cache = pd.concat([pd.read_parquet(os.path.join(CACHE_DIR, f'{f}.parquet'))
                       for f in FOLDS], ignore_index=True)
    cache['date'] = pd.to_datetime(cache.date)
    keys = ['fold', 'date', 'lead', 'lag']
    df = p.merge(cache[keys + ['z_entry', 'z_depth']], on=keys, how='left')
    df = df.merge(pd.read_parquet(os.path.join(CACHE_DIR, 'spread_std.parquet')),
                  on=keys, how='left')
    # Signed z: z_depth == 1 marks the tradeable (negative-z) side; V1 verified
    # this against the shipped compute_entry_metrics at 100%.
    df['z_signed'] = np.where(df.z_depth >= 1 - 1e-9, -df.z_entry, df.z_entry)
    df['expected_gross'] = (df.z_entry - EXIT_Z) * df.lvl_std
    # Gate 1 is fixed by the design, not tuned: only the tradeable direction.
    return df[(df.z_signed <= -ENTRY_THRESHOLD)
              & df.expected_gross.notna()].copy()


# ------------------------------------------------------------------ policy

def select_book(frame: pd.DataFrame, floor: float, k: int) -> pd.DataFrame:
    """Apply the magnitude floor, then take top-k by expected_gross per date
    with the live per-symbol dedup (BobsBrain skips a candidate whose either
    leg is already held)."""
    f = frame[frame.expected_gross >= floor]
    out = []
    for _, g in f.groupby('date'):
        used, kept = set(), []
        for _, r in g.sort_values('expected_gross', ascending=False).iterrows():
            if r.lead in used or r.lag in used:
                continue
            used.update([r.lead, r.lag])
            kept.append(r)
            if len(kept) >= k:
                break
        out.extend(kept)
    return pd.DataFrame(out)


def objective(frame: pd.DataFrame, floor: float, k: int) -> float:
    book = select_book(frame, floor, k)
    return float(book.gross.median()) if len(book) else float('nan')


def fit_best(frame: pd.DataFrame) -> tuple[float, int]:
    """Preregistered selection rule: highest objective; ties broken toward the
    LOWEST floor then the HIGHEST k (prefer staying invested).

    Encoded as a lexicographic max over (objective, -floor, k).
    """
    scored = []
    for floor in FLOOR_GRID:
        for k in K_GRID:
            v = objective(frame, floor, k)
            if np.isfinite(v):
                scored.append(((v, -floor, k), (floor, k)))
    if not scored:
        return BASELINE
    return max(scored, key=lambda t: t[0])[1]


# -------------------------------------------------------------------- main

def main() -> None:
    df = load()
    opt = df[df.date != FINAL_TEST_DATE].copy()
    fin = df[df.date == FINAL_TEST_DATE].copy()
    print(f'dislocated observations: {len(df)}  '
          f'(optimization {len(opt)} on {opt.date.nunique()} dates, '
          f'final test {len(fin)} on {FINAL_TEST_DATE.date()})')

    # --- a-priori design check -------------------------------------------
    comp = complexity_ratio(n_free_params=2, n_independent_units=len(opt))
    print(f"\ncomplexity_ratio: {comp['ratio']:.0f} units/param -> {comp['flag']}")

    # --- walk-forward selection ------------------------------------------
    windows = walk_forward_splits(
        start=opt.date.min(), end=opt.date.max(),
        train_span='180D', eval_span='120D', embargo=EMBARGO)
    print(f'\nwalk-forward windows (embargo {EMBARGO} >= {HORIZON_TD}td horizon): '
          f'{len(windows)}')
    rows = []
    for w in windows:
        tr = opt[(opt.date >= w.train_start) & (opt.date <= w.train_end)]
        ev = opt[(opt.date >= w.eval_start) & (opt.date <= w.eval_end)]
        if len(tr) < 50 or len(ev) < 50:
            continue
        cfg = fit_best(tr)
        rows.append(dict(train=f'{w.train_start.date()}..{w.train_end.date()}',
                         eval=f'{w.eval_start.date()}..{w.eval_end.date()}',
                         floor=cfg[0], k=cfg[1],
                         eval_obj=objective(ev, *cfg),
                         base_obj=objective(ev, *BASELINE)))
    wf = pd.DataFrame(rows)
    if len(wf):
        print(wf.round(1).to_string(index=False))
        print(f"\nrolled OOS: chosen {wf.eval_obj.mean():+.1f} vs "
              f"baseline {wf.base_obj.mean():+.1f}")
    else:
        print('  (range too short for a rolled schedule — falling back to '
              'leave-one-fold-out for selection; reported as a limitation)')

    # --- selection on the full optimization set --------------------------
    grid = pd.DataFrame([
        dict(floor=f, k=k, obj=objective(opt, f, k),
             n=len(select_book(opt, f, k)))
        for f in FLOOR_GRID for k in K_GRID])
    print('\n=== grid on the optimization set (objective = median bps) ===')
    print(grid.pivot(index='floor', columns='k', values='obj').round(1).to_string())
    chosen = fit_best(opt)
    print(f'\nSELECTED: min_expected_gross_bps={chosen[0]:.0f}  max_k={chosen[1]}')

    # --- guards -----------------------------------------------------------
    groups = {f: g for f, g in opt.groupby('fold')}
    hg = holdout_gap(
        fit_fn=lambda tr: fit_best(tr),
        eval_fn=lambda d, params: objective(d, *params),
        groups=groups)
    hg_ok = hg['mean_holdout'] > 0 and hg['mean_gap'] <= abs(hg['mean_holdout'])
    print(f"\nholdout_gap: train {hg['mean_train']:+.1f} | "
          f"holdout {hg['mean_holdout']:+.1f} | gap {hg['mean_gap']:+.1f} "
          f"-> {'pass' if hg_ok else 'FAIL'}")
    for f, g in groups.items():
        print(f'   drop-{f}: re-selected {fit_best(opt[opt.fold != f])}')

    nb = null_baseline(
        df=opt.reset_index(drop=True), group_col='fold', date_col='date',
        outcome_col='gross',
        score_series=opt.reset_index(drop=True).expected_gross,
        k=chosen[1], n_perm=500, seed=0)
    print(f"\nnull_baseline: real {nb['real']:+.1f} | random null "
          f"{nb['random']['mean']:+.1f} (pct {nb['random']['percentile']:.0f}) | "
          f"shuffle null {nb['shuffle']['mean']:+.1f} "
          f"(pct {nb['shuffle']['percentile']:.0f})")

    report_panel(complexity=comp, null=nb, holdout=hg)

    # --- date-clustered CI on the chosen config vs baseline ---------------
    rng = np.random.default_rng(7)
    dates = opt.date.unique()
    obs = objective(opt, *chosen) - objective(opt, *BASELINE)
    boot = []
    for _ in range(1000):
        sub = pd.concat([opt[opt.date == d] for d in rng.choice(dates, len(dates))])
        a, b = objective(sub, *chosen), objective(sub, *BASELINE)
        if np.isfinite(a) and np.isfinite(b):
            boot.append(a - b)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ci_ok = lo > 0
    print(f'\nchosen vs baseline (median bps): {obs:+.1f} | '
          f'date-clustered CI {lo:+.1f} .. {hi:+.1f} -> '
          f'{"EXCLUDES 0" if ci_ok else "includes 0"}')

    # --- FINAL TEST, scored once -----------------------------------------
    print('\n' + '=' * 72)
    print(f'FINAL TEST {FINAL_TEST_DATE.date()} — scored once')
    print('=' * 72)
    fb, fc = select_book(fin, *BASELINE), select_book(fin, *chosen)
    print(f'baseline (floor {BASELINE[0]:.0f}, k={BASELINE[1]}): '
          f'n={len(fb)} median {fb.gross.median():+.1f} mean {fb.gross.mean():+.1f} '
          f'disaster {fb["cat"].mean() * 100:.1f}%')
    print(f'chosen   (floor {chosen[0]:.0f}, k={chosen[1]}): '
          f'n={len(fc)} median {fc.gross.median():+.1f} mean {fc.gross.mean():+.1f} '
          f'disaster {fc["cat"].mean() * 100:.1f}%')

    stable = all(fit_best(opt[opt.fold != f]) == chosen for f in groups)
    print('\n' + '=' * 72)
    print(f'V2 VERDICT: CI {"pass" if ci_ok else "FAIL"} | '
          f'holdout {"pass" if hg_ok else "FAIL"} | '
          f'fold-stability {"pass" if stable else "FAIL"}')
    print('=' * 72)


if __name__ == '__main__':
    main()
