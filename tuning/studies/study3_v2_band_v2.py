"""V2 v2 — the band hypothesis for expected_gross.

Plan: docs/plans/2026-08-01_entry-criteria-overhaul.md §6 (V2 v2).
Methodology: .claude/skills/optuna-study.md.

Version bump from study3_v2_floor_and_k (v1), which FAILED: its objective was
the median while null_baseline scores the mean, and k was unidentifiable.

HYPOTHESIS (mechanism, fixed before running): a very large expected gross stops
indicating a big opportunity and starts indicating the pair's RELATIONSHIP IS
BREAKING — a structural repricing rather than a dislocation. If true, capping
expected gross removes non-converging pairs and improves the MEAN.

PREREGISTERED:
  free params   floor   in {0, 25, 50}
                ceiling in {150, 250, 400, inf}
                k FIXED at 20 (v1 showed k is not identifiable here)
  unit          the pair round-trip
  objective     MEAN gross bps of the selected book.  A book of equal-weighted
                positions earns the mean; median is a secondary diagnostic.
                Using the mean also makes the objective and null_baseline
                coherent — the incoherence that sank v1 — and the band
                hypothesis predicts a MEAN improvement, so this is its
                sharpest test.
  selection     highest mean; ties to the WIDEST band (lowest floor, highest
                ceiling), i.e. prefer intervening less.
  held-out      leave-one-fold-out over 3 genuinely different regimes.  Thin
                (3 units) and reported as such.
  guards        complexity_ratio, null_baseline (mean-based -> coherent),
                holdout_gap, fold-drop stability.
  mechanism     high expected_gross must show the BREAKDOWN signature:
                higher cap-exit (non-convergence) rate, longer holds, deeper
                drawdowns.  Performance without mechanism = curve-fitting.

CONFIRMATORY STATUS: the band hypothesis was generated from these same 12
dates and v1's final-test date is spent.  No clean confirmatory test exists in
this dataset.  This study is HYPOTHESIS-GENERATING — it decides whether a real
test (fresh folds, or the V4 comparative backtest) is warranted.  It cannot
validate the band, and a pass here does not authorise shipping.

Run:  DB_URL=... python -m tuning.studies.study3_v2_band_v2
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from tuning.overfit_guards import complexity_ratio, holdout_gap, null_baseline
from tuning.studies.scoring_replay import EXIT_Z, HORIZON

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_scoring_cache')
FOLDS = ('sideways_2022', 'bull_2023', 'mixed_2023_q4')
ENTRY_THRESHOLD = 2.0
K_FIXED = 20
FLOOR_GRID = (0.0, 25.0, 50.0)
CEIL_GRID = (150.0, 250.0, 400.0, float('inf'))
BASELINE = (0.0, float('inf'))       # no floor, no ceiling


def load() -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(CACHE_DIR, 'e2_paths.parquet'))
    cache = pd.concat([pd.read_parquet(os.path.join(CACHE_DIR, f'{f}.parquet'))
                       for f in FOLDS], ignore_index=True)
    cache['date'] = pd.to_datetime(cache.date)
    keys = ['fold', 'date', 'lead', 'lag']
    df = p.merge(cache[keys + ['z_entry', 'z_depth']], on=keys, how='left')
    df = df.merge(pd.read_parquet(os.path.join(CACHE_DIR, 'spread_std.parquet')),
                  on=keys, how='left')
    df['z_signed'] = np.where(df.z_depth >= 1 - 1e-9, -df.z_entry, df.z_entry)
    df['expected_gross'] = (df.z_entry - EXIT_Z) * df.lvl_std
    df['capped'] = df.exit_t >= HORIZON        # never converged
    return df[(df.z_signed <= -ENTRY_THRESHOLD)
              & df.expected_gross.notna()].copy()


def select_book(frame: pd.DataFrame, floor: float, ceil: float,
                k: int = K_FIXED) -> pd.DataFrame:
    f = frame[(frame.expected_gross >= floor) & (frame.expected_gross <= ceil)]
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


def objective(frame: pd.DataFrame, floor: float, ceil: float) -> float:
    b = select_book(frame, floor, ceil)
    return float(b.gross.mean()) if len(b) else float('nan')


def fit_best(frame: pd.DataFrame) -> tuple[float, float]:
    """Highest mean; ties to the widest band (lowest floor, highest ceiling)."""
    scored = []
    for floor in FLOOR_GRID:
        for ceil in CEIL_GRID:
            v = objective(frame, floor, ceil)
            if np.isfinite(v):
                scored.append(((v, -floor, ceil), (floor, ceil)))
    return max(scored, key=lambda t: t[0])[1] if scored else BASELINE


def _fmt(c: float) -> str:
    return 'inf' if not np.isfinite(c) else f'{c:.0f}'


def main() -> None:
    df = load()
    print(f'dislocated observations: {len(df)} across {df.date.nunique()} dates')

    # --- MECHANISM CHECK FIRST -------------------------------------------
    # If this fails the performance numbers are not worth reading.
    print('\n' + '=' * 72)
    print('MECHANISM — do high-expected-gross pairs look like broken relationships?')
    print('=' * 72)
    df['q'] = pd.qcut(df.expected_gross, 5, labels=False, duplicates='drop')
    mech = df.groupby('q').agg(
        n=('gross', 'size'), exp_gross=('expected_gross', 'median'),
        cap_exit_pct=('capped', lambda s: s.mean() * 100),
        hold_days=('exit_t', 'median'),
        max_dd=('max_dd', 'median'),
        disaster_pct=('cat', lambda s: s.mean() * 100),
        mean=('gross', 'mean'), median=('gross', 'median'))
    print(mech.round(1).to_string())
    lo_q, hi_q = mech.index.min(), mech.index.max()
    mech_ok = (mech.loc[hi_q, 'cap_exit_pct'] > mech.loc[lo_q, 'cap_exit_pct']
               and mech.loc[hi_q, 'disaster_pct'] > mech.loc[lo_q, 'disaster_pct'])
    print(f'\nbreakdown signature (cap-exit and disaster rate both rise with '
          f'expected gross): {"PRESENT" if mech_ok else "ABSENT"}')

    # --- grid --------------------------------------------------------------
    comp = complexity_ratio(n_free_params=2, n_independent_units=len(df))
    print(f"\ncomplexity_ratio: {comp['ratio']:.0f} units/param -> {comp['flag']}")
    print('\n=== grid (objective = MEAN gross bps, k=20) ===')
    grid = pd.DataFrame(
        [[objective(df, f, c) for c in CEIL_GRID] for f in FLOOR_GRID],
        index=[f'floor {f:.0f}' for f in FLOOR_GRID],
        columns=[f'ceil {_fmt(c)}' for c in CEIL_GRID])
    print(grid.round(1).to_string())
    print('\n(median, secondary diagnostic)')
    gmed = pd.DataFrame(
        [[float(select_book(df, f, c).gross.median()) for c in CEIL_GRID]
         for f in FLOOR_GRID],
        index=[f'floor {f:.0f}' for f in FLOOR_GRID],
        columns=[f'ceil {_fmt(c)}' for c in CEIL_GRID])
    print(gmed.round(1).to_string())

    chosen = fit_best(df)
    print(f'\nSELECTED: floor={chosen[0]:.0f} ceiling={_fmt(chosen[1])}  (k=20)')
    base_book, ch_book = select_book(df, *BASELINE), select_book(df, *chosen)
    print(f'  baseline n={len(base_book)} mean {base_book.gross.mean():+.1f} '
          f'median {base_book.gross.median():+.1f} '
          f'disaster {base_book["cat"].mean() * 100:.1f}%')
    print(f'  chosen   n={len(ch_book)} mean {ch_book.gross.mean():+.1f} '
          f'median {ch_book.gross.median():+.1f} '
          f'disaster {ch_book["cat"].mean() * 100:.1f}%')

    # --- guards ------------------------------------------------------------
    groups = {f: g for f, g in df.groupby('fold')}
    hg = holdout_gap(fit_fn=fit_best,
                     eval_fn=lambda d, params: objective(d, *params),
                     groups=groups)
    hg_ok = hg['mean_holdout'] > 0 and hg['mean_gap'] <= abs(hg['mean_holdout'])
    print(f"\nholdout_gap (leave-one-fold-out, 3 thin units): train "
          f"{hg['mean_train']:+.1f} | holdout {hg['mean_holdout']:+.1f} | "
          f"gap {hg['mean_gap']:+.1f} -> {'pass' if hg_ok else 'FAIL'}")
    drops = {f: fit_best(df[df.fold != f]) for f in groups}
    for f, cfg in drops.items():
        print(f'   drop-{f}: re-selected floor={cfg[0]:.0f} ceil={_fmt(cfg[1])}')
    ceil_stable = len({c for _, c in drops.values()}) == 1
    print(f'   ceiling stable across fold-drops: '
          f'{"yes" if ceil_stable else "NO"}')

    # Null baseline: score = expected_gross clipped to the band (out-of-band
    # candidates rank last), so the null tests the BAND, not raw magnitude.
    work = df.reset_index(drop=True)
    in_band = ((work.expected_gross >= chosen[0])
               & (work.expected_gross <= chosen[1]))
    score = work.expected_gross.where(in_band, -np.inf)
    nb = null_baseline(df=work, group_col='fold', date_col='date',
                       outcome_col='gross', score_series=score,
                       k=K_FIXED, n_perm=500, seed=0)
    print(f"\nnull_baseline (mean metric): real {nb['real']:+.1f} | random "
          f"{nb['random']['mean']:+.1f} (pct {nb['random']['percentile']:.0f}) | "
          f"shuffle {nb['shuffle']['mean']:+.1f} "
          f"(pct {nb['shuffle']['percentile']:.0f})")
    null_ok = nb['random']['percentile'] >= 90

    # --- date-clustered CI: chosen vs no-ceiling --------------------------
    rng = np.random.default_rng(13)
    dates = df.date.unique()
    obs = objective(df, *chosen) - objective(df, *BASELINE)
    boot = []
    for _ in range(1000):
        sub = pd.concat([df[df.date == d] for d in rng.choice(dates, len(dates))])
        a, b = objective(sub, *chosen), objective(sub, *BASELINE)
        if np.isfinite(a) and np.isfinite(b):
            boot.append(a - b)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ci_ok = lo > 0
    print(f'\nchosen vs no-ceiling (MEAN bps): {obs:+.1f} | date-clustered CI '
          f'{lo:+.1f} .. {hi:+.1f} -> {"EXCLUDES 0" if ci_ok else "includes 0"}')

    print('\n' + '=' * 72)
    print(f'V2 v2 VERDICT: mechanism {"PRESENT" if mech_ok else "ABSENT"} | '
          f'CI {"pass" if ci_ok else "FAIL"} | '
          f'null {"pass" if null_ok else "FAIL"} | '
          f'holdout {"pass" if hg_ok else "FAIL"} | '
          f'ceiling-stable {"pass" if ceil_stable else "FAIL"}')
    print('HYPOTHESIS-GENERATING ONLY — the band was derived from these dates, '
          'so a\npass here justifies a confirmatory test (fresh folds or V4), '
          'not shipping.')
    print('=' * 72)


if __name__ == '__main__':
    main()
