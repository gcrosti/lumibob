"""Overfitting guards for tuning studies — deterministic, seeded, no DB/network.

Motivation (Pass A v4 post-mortem): a study that optimized 6 params to maximize a
"top-K mean forward P&L" objective aggregated over only **3 regime folds** reported a
great in-sample number (+52) that was pure overfitting.  The tell-tale symptoms were:

  * leave-one-fold-out held-out scores were negative;
  * top-K-by-score was indistinguishable from random selection;
  * the raw components had ~0 rank-correlation with the outcome at the *pair* level;
  * the best weights swung wildly across random seeds.

The root cause: evaluation happened at the level with few independent units (3 folds)
instead of where the independent units actually are (hundreds of pairs).  These guards
make each of those symptoms impossible to miss on a future study.

Data contract (matches ``tuning/scoring_study.py`` / ``scoring_replay.py``): a pandas
DataFrame with a group column (e.g. ``fold``), a set of component columns (e.g.
``COMPONENTS``), a ``forward_gross`` outcome column, and a ``date`` column.

Everything here is pure: no I/O, no global state, and every stochastic routine takes a
``seed`` (or a callable the caller seeds).  Spearman is computed via average-rank +
Pearson so there is no scipy dependency.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation of two 1-D arrays (NaNs dropped pairwise).

    Returns ``nan`` when there is no variance (constant input) or < 3 usable pairs.
    Implemented as Pearson correlation of average ranks to avoid a scipy dependency.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    xr = pd.Series(x[mask]).rank(method='average').to_numpy()
    yr = pd.Series(y[mask]).rank(method='average').to_numpy()
    if xr.std() == 0 or yr.std() == 0:
        return float('nan')
    return float(np.corrcoef(xr, yr)[0, 1])


def _percentile_of(value: float, dist: np.ndarray) -> float:
    """Percentile (0-100) of ``value`` within the empirical ``dist`` (<= convention)."""
    dist = np.asarray(dist, dtype=float)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0 or not np.isfinite(value):
        return float('nan')
    return float((dist <= value).mean() * 100.0)


# --------------------------------------------------------------------------- #
# Guard 1 — unit-level signal (the PRIMARY gate)
# --------------------------------------------------------------------------- #
def unit_level_signal(df: pd.DataFrame, component_cols: Sequence[str], outcome_col: str,
                      group_col: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """Per-group Spearman rho of each component vs the outcome, with a bootstrap CI.

    This is the FIRST gate a scoring study must clear.  The independent units are the
    rows (pairs), not the groups: within each group we ask, across all its rows, whether
    each raw component ranks the outcome at all.  If *every* component's CI includes 0 in
    *every* group, then no linear weighting of those components can rank the outcome —
    stop before tuning.

    Args:
        df: observation table (one row per pair-observation).
        component_cols: the raw component columns to test individually.
        outcome_col: the forward-outcome column (e.g. ``forward_gross``).
        group_col: the grouping column (e.g. ``fold``); rho is computed within each group.
        n_boot: bootstrap resamples for the CI.
        seed: RNG seed (deterministic).

    Returns:
        ``{group_value: {component: (rho, ci_lo, ci_hi)}}``.  A degenerate component
        (constant / too few rows) yields ``(nan, nan, nan)``.
    """
    rng = np.random.default_rng(seed)
    result: dict = {}
    for gval, g in df.groupby(group_col):
        per_comp: dict = {}
        y = g[outcome_col].to_numpy(dtype=float)
        for comp in component_cols:
            x = g[comp].to_numpy(dtype=float)
            rho = _spearman(x, y)
            if not np.isfinite(rho):
                per_comp[comp] = (float('nan'), float('nan'), float('nan'))
                continue
            n = len(x)
            boot = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot[b] = _spearman(x[idx], y[idx])
            boot = boot[np.isfinite(boot)]
            if boot.size == 0:
                per_comp[comp] = (rho, float('nan'), float('nan'))
            else:
                lo, hi = np.percentile(boot, [2.5, 97.5])
                per_comp[comp] = (rho, float(lo), float(hi))
        result[gval] = per_comp
    return result


def all_null(result: Mapping) -> bool:
    """True iff every component's CI includes 0 in every group (no unit-level signal).

    A degenerate ``(nan, nan, nan)`` entry counts as null.  If any component in any
    group has a CI strictly excluding 0, the panel has signal and this returns False.
    """
    for per_comp in result.values():
        for (_rho, lo, hi) in per_comp.values():
            if np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0):
                return False
    return True


# --------------------------------------------------------------------------- #
# Guard 2 — held-out gap (leave-one-group-out)
# --------------------------------------------------------------------------- #
def holdout_gap(fit_fn: Callable[[pd.DataFrame], object],
                eval_fn: Callable[[pd.DataFrame, object], float],
                groups: Mapping[object, pd.DataFrame]) -> dict:
    """Leave-one-group-out train/held-out comparison.

    For each group held out, fit params on the *other* groups and score both the
    training pool and the held-out group.  A large positive ``gap`` (train score far
    above held-out score) — especially with a negative held-out score — is the
    signature of overfitting.

    Args:
        fit_fn: ``train_df -> params`` (whatever object ``eval_fn`` consumes).
        eval_fn: ``(df, params) -> float`` scoring metric.
        groups: ``{group_label: df}``; must contain >= 2 groups.

    Returns:
        ``{'per_group': {g: {'train_score', 'holdout_score', 'gap'}},
           'mean_gap': float, 'mean_holdout': float, 'mean_train': float}``
        where ``gap = train_score - holdout_score``.
    """
    labels = list(groups)
    if len(labels) < 2:
        raise ValueError('holdout_gap needs at least 2 groups for leave-one-out')
    per_group: dict = {}
    for held in labels:
        train_df = pd.concat([groups[g] for g in labels if g != held], ignore_index=True)
        params = fit_fn(train_df)
        train_score = float(eval_fn(train_df, params))
        holdout_score = float(eval_fn(groups[held], params))
        per_group[held] = {
            'train_score': train_score,
            'holdout_score': holdout_score,
            'gap': train_score - holdout_score,
        }
    gaps = [v['gap'] for v in per_group.values()]
    holdouts = [v['holdout_score'] for v in per_group.values()]
    trains = [v['train_score'] for v in per_group.values()]
    return {
        'per_group': per_group,
        'mean_gap': float(np.mean(gaps)),
        'mean_holdout': float(np.mean(holdouts)),
        'mean_train': float(np.mean(trains)),
    }


# --------------------------------------------------------------------------- #
# Guard 3 — null baseline (real selection vs random-k and outcome-shuffled)
# --------------------------------------------------------------------------- #
def null_baseline(df: pd.DataFrame, group_col: str, date_col: str, outcome_col: str,
                  score_series: pd.Series, k: int, n_perm: int = 500, seed: int = 0) -> dict:
    """Compare real top-k-by-score selection against random and shuffled nulls.

    The selection metric is the mean outcome of the top-k pairs by score within each
    ``(group, date)`` cell, pooled (averaged) across cells — mirroring the study's
    top-K objective.  Two nulls are built:

      * **random**: pick a random k per cell (destroys the *selection*);
      * **shuffle**: permute the outcome across all rows, then re-select top-k by score
        (destroys the *score->outcome link* while keeping the selection structure).

    A genuinely predictive score lands in a high percentile of both nulls; a random
    score lands near the 50th percentile.

    Args:
        df: observation table.
        group_col, date_col: define the per-cell selection unit.
        outcome_col: forward-outcome column.
        score_series: composite score per row, index-aligned to ``df``.
        k: top-k selected per cell.
        n_perm, seed: permutation count and RNG seed.

    Returns:
        ``{'real', 'random': {mean, sd, z, percentile}, 'shuffle': {...},
           'null_mean', 'null_sd', 'z', 'percentile'}``.  The flat keys mirror the
        ``random`` null (the headline "better than random selection?").
    """
    rng = np.random.default_rng(seed)
    work = df.copy()
    work['_score'] = np.asarray(score_series, dtype=float)
    outcome = work[outcome_col].to_numpy(dtype=float)

    # Precompute, per cell: row positions and the score order (top-k indices by score).
    cells = []
    for _, g in work.groupby([group_col, date_col], sort=False):
        pos = g.index.to_numpy()
        loc = work.index.get_indexer(pos)            # integer positions into `outcome`
        order = np.argsort(-g['_score'].to_numpy())  # local top-k order (descending score)
        cells.append((loc, order))

    def _pooled(sel_outcome_per_cell) -> float:
        vals = [v for v in sel_outcome_per_cell if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float('nan')

    # Real metric: top-k by score in each cell.
    real = _pooled([
        np.nanmean(outcome[loc[order[:min(k, len(loc))]]]) for loc, order in cells
    ])

    # Random-k null.
    rand_samples = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        per_cell = []
        for loc, _order in cells:
            m = min(k, len(loc))
            pick = rng.choice(len(loc), size=m, replace=False)
            per_cell.append(np.nanmean(outcome[loc[pick]]))
        rand_samples[p] = _pooled(per_cell)

    # Outcome-shuffled null: permute outcomes globally, re-select top-k by (fixed) score.
    shuf_samples = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        perm = rng.permutation(outcome)
        per_cell = [
            np.nanmean(perm[loc[order[:min(k, len(loc))]]]) for loc, order in cells
        ]
        shuf_samples[p] = _pooled(per_cell)

    def _summ(samples: np.ndarray) -> dict:
        s = samples[np.isfinite(samples)]
        mean = float(np.mean(s)) if s.size else float('nan')
        sd = float(np.std(s, ddof=1)) if s.size > 1 else float('nan')
        z = (real - mean) / sd if (np.isfinite(sd) and sd > 0) else float('nan')
        return {'mean': mean, 'sd': sd, 'z': float(z),
                'percentile': _percentile_of(real, s)}

    rand = _summ(rand_samples)
    shuf = _summ(shuf_samples)
    return {
        'real': real,
        'random': rand,
        'shuffle': shuf,
        'null_mean': rand['mean'],
        'null_sd': rand['sd'],
        'z': rand['z'],
        'percentile': rand['percentile'],
    }


# --------------------------------------------------------------------------- #
# Guard 4 — seed stability
# --------------------------------------------------------------------------- #
def seed_stability(study_fn: Callable[[int], Mapping[str, float]], seeds: Sequence[int],
                   param_keys: Sequence[str], threshold: float | Mapping[str, float] | None = None,
                   objective_key: str = 'objective') -> dict:
    """Run a study under several seeds and measure how much its winner moves.

    A robust optimum barely moves across seeds; a noise-dominated one swings wildly.

    Args:
        study_fn: ``seed -> best_params`` mapping.  If it also returns ``objective_key``,
            its spread across seeds is reported.
        seeds: seeds to run.
        param_keys: keys to summarize.
        threshold: scalar or per-key mapping; a key is flagged when its range exceeds
            the threshold.  ``None`` disables flagging (all ``flagged=False``).
        objective_key: key under which each result carries its objective value (optional).

    Returns:
        ``{'per_key': {key: {min, median, max, range, flagged}},
           'objective_spread': float|nan, 'flagged': [keys], 'seeds': [...], 'runs': [...]}``
    """
    runs = [dict(study_fn(s)) for s in seeds]
    per_key: dict = {}
    flagged: list = []
    for key in param_keys:
        vals = np.array([float(r[key]) for r in runs], dtype=float)
        rng_span = float(vals.max() - vals.min())
        if threshold is None:
            thr = None
        elif isinstance(threshold, Mapping):
            thr = threshold.get(key)
        else:
            thr = float(threshold)
        is_flagged = bool(thr is not None and rng_span > thr)
        per_key[key] = {
            'min': float(vals.min()),
            'median': float(np.median(vals)),
            'max': float(vals.max()),
            'range': rng_span,
            'flagged': is_flagged,
        }
        if is_flagged:
            flagged.append(key)

    obj_spread = float('nan')
    if all(objective_key in r for r in runs):
        ov = np.array([float(r[objective_key]) for r in runs], dtype=float)
        obj_spread = float(ov.max() - ov.min())

    return {
        'per_key': per_key,
        'objective_spread': obj_spread,
        'flagged': flagged,
        'seeds': list(seeds),
        'runs': runs,
    }


# --------------------------------------------------------------------------- #
# Guard 5 — complexity ratio (a-priori design check)
# --------------------------------------------------------------------------- #
def complexity_ratio(n_free_params: int, n_independent_units: int) -> dict:
    """Design-time sanity check: independent units per free parameter.

    ``ratio = n_independent_units / n_free_params``.

      * **red**   — free params >= units (``ratio <= 1``): the fit can memorize; any
        in-sample optimum is meaningless.
      * **amber** — within 3x (``1 < ratio < 3``): thin; expect fragile optima.
      * **green** — ``ratio >= 3``.

    The Pass A v4 failure was a textbook red: 6 free params over 3 fold-level units.
    """
    if n_free_params <= 0:
        raise ValueError('n_free_params must be positive')
    ratio = n_independent_units / n_free_params
    if n_free_params >= n_independent_units:
        flag = 'red'
    elif n_independent_units < 3 * n_free_params:
        flag = 'amber'
    else:
        flag = 'green'
    return {
        'ratio': float(ratio),
        'flag': flag,
        'n_free_params': int(n_free_params),
        'n_independent_units': int(n_independent_units),
    }


# --------------------------------------------------------------------------- #
# Report panel
# --------------------------------------------------------------------------- #
def report_panel(*, complexity: dict | None = None, unit_signal: Mapping | None = None,
                 null: dict | None = None, holdout: dict | None = None,
                 seed: dict | None = None,
                 signal_percentile_floor: float = 90.0,
                 print_fn: Callable[[str], object] | None = print) -> str:
    """Compact red/green summary of the guard outputs.

    Pass whichever guard results are available; each renders one line with a PASS/FAIL
    (or GREEN/AMBER/RED) verdict.  Returns the full report string and, unless
    ``print_fn`` is None, prints it.
    """
    lines: list[str] = ['=== Overfitting guard panel ===']

    if complexity is not None:
        verdict = complexity['flag'].upper()
        lines.append(
            f'[{verdict:5s}] complexity   ratio={complexity["ratio"]:.2f} '
            f'({complexity["n_independent_units"]} units / '
            f'{complexity["n_free_params"]} free params)')

    if unit_signal is not None:
        null_signal = all_null(unit_signal)
        verdict = 'FAIL' if null_signal else 'PASS'
        # Count component-CIs that exclude 0 (have signal).
        n_sig = sum(
            1 for per in unit_signal.values() for (_r, lo, hi) in per.values()
            if np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
        lines.append(
            f'[{verdict:5s}] unit-signal  {n_sig} component-CIs exclude 0'
            + ('  (no component ranks the outcome — STOP)' if null_signal else ''))

    if null is not None:
        pct = null.get('percentile', float('nan'))
        verdict = 'PASS' if (np.isfinite(pct) and pct >= signal_percentile_floor) else 'FAIL'
        lines.append(
            f'[{verdict:5s}] null-base    real={null["real"]:+.3f} '
            f'pctile={pct:.0f} (random null z={null.get("z", float("nan")):+.2f})')

    if holdout is not None:
        mg = holdout['mean_gap']
        mh = holdout['mean_holdout']
        verdict = 'PASS' if mh > 0 and mg <= abs(mh) else 'FAIL'
        lines.append(
            f'[{verdict:5s}] holdout-gap  mean_holdout={mh:+.3f} '
            f'mean_gap={mg:+.3f}'
            + ('  (held-out negative / gap dominates)' if verdict == 'FAIL' else ''))

    if seed is not None:
        flagged = seed.get('flagged', [])
        verdict = 'FAIL' if flagged else 'PASS'
        lines.append(
            f'[{verdict:5s}] seed-stab    obj_spread={seed.get("objective_spread", float("nan")):.3f}'
            + (f'  unstable: {", ".join(map(str, flagged))}' if flagged else ''))

    report = '\n'.join(lines)
    if print_fn is not None:
        print_fn(report)
    return report
