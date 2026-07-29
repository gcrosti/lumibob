"""Scoring-quality study — Pass A v4, WS3.

Optuna optimizes the composite **weights** and the two **predictor windows**
(`corr_long_window`, `corr_short_window`) over the cached WS2 observation matrix, to
maximize a **tail-sensitive, edge-aligned** ranking metric:

    objective = 0.5 * mean_over_folds(fold_metric) + 0.5 * min_over_folds(fold_metric)
    fold_metric = mean over the fold's scoring dates of
                  [ mean forward_gross of the top-K pairs by composite score ]

Top-K mean (not Spearman) so the metric is tail-sensitive — a catastrophic pair that
sneaks into the top-K drags it down.  The 0.5*min term is the per-fold floor (LORO
discipline).  The outcome contract (zscore_window, exit rule, coint/halflife params)
is frozen in WS2; here we only reweight and re-window.

No lumibot, no portfolio, no costs — pure scoring quality.  See
docs/plans/2026-07-22_pass-a-v4-scoring-study.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tuning.studies.scoring_replay import CACHE_DIR, FOLDS, DEF_CORR_LONG, DEF_CORR_SHORT

# z_depth is excluded: every tradeable pair is dislocated past the entry threshold, so
# score_z_depth == 1.0 for all of them — a constant that cannot affect the ranking.
# (Operationally z_depth still gates dislocated vs non-dislocated pairs; that role is
# already applied here by keeping only tradeable pairs.  Its LIVE weight stays as-is.)
COMPONENTS = ['score_corr_long', 'score_corr_short', 'score_coint', 'score_halflife']
TOP_K = 20


def load_cache() -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        p = os.path.join(CACHE_DIR, f'{fold}.parquet')
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit(f'no cache in {CACHE_DIR}; run scoring_replay first')
    df = pd.concat(frames, ignore_index=True)
    return df[df.forward_gross.notna()].reset_index(drop=True)


def _corr_at(ret_lead, ret_lag, window) -> float:
    """corr over the trailing `window` returns (score = clip to [0,1] happens later)."""
    n = min(len(ret_lead), len(ret_lag))
    if n < window:
        return np.nan
    a, b = ret_lead[-window:], ret_lag[-window:]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def add_windowed_corr(df: pd.DataFrame, corr_long_w: int, corr_short_w: int) -> pd.DataFrame:
    """Recompute score_corr_long / score_corr_short at the given windows (WS3 tunes these)."""
    cl = np.array([_corr_at(rl, rg, corr_long_w) for rl, rg in zip(df.ret_lead, df.ret_lag)])
    cs = np.array([_corr_at(rl, rg, corr_short_w) for rl, rg in zip(df.ret_lead, df.ret_lag)])
    out = df.copy()
    out['score_corr_long'] = np.clip(np.nan_to_num(cl, nan=0.0), 0.0, 1.0)
    out['score_corr_short'] = np.clip(np.nan_to_num(cs, nan=0.0), 0.0, 1.0)
    return out


def fold_metrics(df: pd.DataFrame, weights: dict, k: int = TOP_K) -> dict:
    """Top-K mean forward_gross per fold (mean over the fold's scoring dates)."""
    w = np.array([weights[c] for c in COMPONENTS], dtype=float)
    s = df[COMPONENTS].to_numpy() @ w          # composite (rank-invariant to |w|)
    df = df.assign(_score=s)
    out = {}
    for fold, fg in df.groupby('fold'):
        date_means = []
        for _, g in fg.groupby('date'):
            topk = g.nlargest(min(k, len(g)), '_score')
            if len(topk):
                date_means.append(topk.forward_gross.mean())
        out[fold] = float(np.mean(date_means)) if date_means else np.nan
    return out


def objective_value(fm: dict) -> float:
    vals = [v for v in fm.values() if np.isfinite(v)]
    if not vals:
        return -1e9
    return 0.5 * float(np.mean(vals)) + 0.5 * float(np.min(vals))


# Weight search names (mirror COMPONENTS; z_depth excluded — inert here).
_WEIGHTS = ['w_corr_long', 'w_corr_short', 'w_coint', 'w_halflife']


def _weights_from(trial_params: dict) -> dict:
    raw = np.array([trial_params[w] for w in _WEIGHTS], dtype=float)
    raw = raw / raw.sum() if raw.sum() > 0 else np.ones(len(_WEIGHTS)) / len(_WEIGHTS)
    return dict(zip(COMPONENTS, raw))


def run_study(df: pd.DataFrame, n_trials: int = 200, seed: int = 42):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Pre-window cache so repeated (long,short) combos are cheap-ish.
    _wcache: dict[tuple, pd.DataFrame] = {}

    def windowed(cl_w, cs_w):
        key = (cl_w, cs_w)
        if key not in _wcache:
            _wcache[key] = add_windowed_corr(df, cl_w, cs_w)
        return _wcache[key]

    def objective(trial):
        # corr_long capped at ~100: lookback_window=152 cal (~105 td) bounds the
        # available returns, matching live scoring (scoring_replay.MAX_CORR_LONG).
        cl_w = trial.suggest_int('corr_long_window', 45, 100)
        cs_w = trial.suggest_int('corr_short_window', 10, min(60, cl_w))
        params = {w: trial.suggest_float(w, 0.0, 1.0) for w in _WEIGHTS}
        d = windowed(cl_w, cs_w)
        fm = fold_metrics(d, _weights_from(params))
        for f, v in fm.items():
            trial.set_user_attr(f, v)
        return objective_value(fm)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def _fmt(fm: dict) -> str:
    return ' '.join(f'{k.split("_")[0]}:{v:+.0f}' for k, v in fm.items())


def main():
    df = load_cache()
    print(f'loaded {len(df)} tradeable observations '
          f'({df.groupby("fold").size().to_dict()})')

    # --- Baseline: the noise-selected gate weights + gate windows ---
    gate_w = {'w_corr_long': 0.180, 'w_corr_short': 0.002,
              'w_coint': 0.386, 'w_halflife': 0.407}
    base_df = add_windowed_corr(df, DEF_CORR_LONG, DEF_CORR_SHORT)
    base_fm = fold_metrics(base_df, _weights_from(gate_w))
    print(f'\nBASELINE (gate weights, windows {DEF_CORR_LONG}/{DEF_CORR_SHORT}): '
          f'obj {objective_value(base_fm):+.1f} | folds {_fmt(base_fm)}')

    # --- Study ---
    study = run_study(df, n_trials=int(os.getenv('N_TRIALS', '200')))
    bt = study.best_trial
    bw = _weights_from({w: bt.params[w] for w in _WEIGHTS})
    print(f'\nBEST TRIAL: obj {bt.value:+.1f}')
    print(f'  windows: corr_long={bt.params["corr_long_window"]} '
          f'corr_short={bt.params["corr_short_window"]}')
    print('  normalized weights:')
    for c, v in bw.items():
        print(f'    {c:16s} {v:.3f}')
    print(f'  per-fold: {_fmt({f: bt.user_attrs.get(f, float("nan")) for f in FOLDS})}')

    # --- w_corr_short across the top trials (is the lever robust?) ---
    top = sorted(study.trials, key=lambda t: -(t.value or -1e9))[:10]
    csw = [_weights_from({w: t.params[w] for w in _WEIGHTS})['score_corr_short'] for t in top]
    print(f'\nw_corr_short in top-10 trials: median {np.median(csw):.3f} '
          f'range [{min(csw):.3f}, {max(csw):.3f}]  (gate value 0.002)')


if __name__ == '__main__':
    main()
