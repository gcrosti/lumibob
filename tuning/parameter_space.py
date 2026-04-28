"""
parameter_space — single source of truth for every tunable parameter.

Each entry in PARAMETER_SPACE carries:
  - tier         : 1 = constant, 2 = slow-adaptive, 3 = fast-adaptive
  - default      : value used when no tuner is active
  - low / high   : inclusive search bounds (None for categorical)
  - dtype        : 'float' | 'int' | 'categorical'
  - log          : sample in log-space (for float / int only)
  - choices      : list of allowed values (categorical only)

The composite score weights (w_corr_long, w_corr_short, w_z_depth) are
sampled freely and normalised inside suggest() so they always sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import optuna


@dataclass
class ParamSpec:
    name: str
    tier: int
    default: Any
    low: Any = None
    high: Any = None
    dtype: Literal['float', 'int', 'categorical'] = 'float'
    log: bool = False
    choices: list | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Complete parameter inventory
# ---------------------------------------------------------------------------

PARAMETER_SPACE: dict[str, ParamSpec] = {

    # =========================================================================
    # Tier 1 — Constant (set once, audit annually)
    # =========================================================================

    'lookback_window': ParamSpec(
        'lookback_window', tier=1, default=130, low=60, high=252, dtype='int',
    ),
    'cooldown_days': ParamSpec(
        'cooldown_days', tier=1, default=7, low=3, high=21, dtype='int',
    ),
    'pca_variance': ParamSpec(
        'pca_variance', tier=1, default=0.95, low=0.80, high=0.99,
    ),
    'min_coverage': ParamSpec(
        'min_coverage', tier=1, default=0.5, low=0.3, high=0.8,
    ),
    'penny_threshold': ParamSpec(
        'penny_threshold', tier=1, default=5.0, low=1.0, high=10.0,
    ),
    'hdbscan_metric': ParamSpec(
        'hdbscan_metric', tier=1, default='precomputed',
        dtype='categorical', choices=['precomputed', 'euclidean'],
    ),
    'hdbscan_selection_method': ParamSpec(
        'hdbscan_selection_method', tier=1, default='eom',
        dtype='categorical', choices=['eom', 'leaf'],
    ),

    # =========================================================================
    # Tier 2 — Slow-adaptive (re-tune monthly to quarterly)
    # =========================================================================

    'max_k': ParamSpec(
        'max_k', tier=2, default=20, low=5, high=50, dtype='int',
    ),
    'target_deployed_pct': ParamSpec(
        'target_deployed_pct', tier=2, default=0.60, low=0.40, high=0.90,
    ),
    'max_daily_candidates': ParamSpec(
        'max_daily_candidates', tier=2, default=200, low=50, high=300, dtype='int',
    ),
    'corr_long_window': ParamSpec(
        'corr_long_window', tier=2, default=90, low=45, high=252, dtype='int',
    ),
    'corr_short_window': ParamSpec(
        'corr_short_window', tier=2, default=20, low=10, high=60, dtype='int',
    ),
    # Weights are sampled freely and normalised in suggest() so they sum to 1.
    'w_corr_long': ParamSpec(
        'w_corr_long', tier=2, default=0.3, low=0.0, high=1.0,
    ),
    'w_corr_short': ParamSpec(
        'w_corr_short', tier=2, default=0.5, low=0.0, high=1.0,
    ),
    'w_z_depth': ParamSpec(
        'w_z_depth', tier=2, default=0.2, low=0.0, high=1.0,
    ),
    'hdbscan_min_cluster_size': ParamSpec(
        'hdbscan_min_cluster_size', tier=2, default=5, low=3, high=15, dtype='int',
    ),
    'hdbscan_min_samples': ParamSpec(
        'hdbscan_min_samples', tier=2, default=2, low=1, high=5, dtype='int',
    ),
    'hdbscan_cluster_selection_epsilon': ParamSpec(
        'hdbscan_cluster_selection_epsilon', tier=2, default=0.0, low=0.0, high=0.5,
    ),
    'min_intra_cluster_corr': ParamSpec(
        'min_intra_cluster_corr', tier=2, default=0.3, low=0.1, high=0.6,
    ),
    'cluster_recompute_days': ParamSpec(
        'cluster_recompute_days', tier=2, default=30, low=14, high=90, dtype='int',
    ),

    # =========================================================================
    # Tier 3 — Fast-adaptive (re-tune weekly, regime-conditioned)
    # =========================================================================

    'entry_threshold': ParamSpec(
        'entry_threshold', tier=3, default=2.0, low=1.0, high=3.5,
    ),
    'exit_threshold': ParamSpec(
        'exit_threshold', tier=3, default=0.5, low=0.1, high=1.5,
    ),
    'zscore_window': ParamSpec(
        'zscore_window', tier=3, default=20, low=10, high=40, dtype='int',
    ),
    'min_position_pct': ParamSpec(
        'min_position_pct', tier=3, default=0.03, low=0.01, high=0.10,
    ),
    'max_position_pct': ParamSpec(
        'max_position_pct', tier=3, default=0.20, low=0.05, high=0.35,
    ),
    'quality_scale_pivot': ParamSpec(
        'quality_scale_pivot', tier=3, default=0.7, low=0.4, high=1.0,
    ),
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

_WEIGHT_NAMES = frozenset({'w_corr_long', 'w_corr_short', 'w_z_depth'})


def defaults() -> dict[str, Any]:
    """Return the canonical default parameter set (all tiers)."""
    return {name: spec.default for name, spec in PARAMETER_SPACE.items()}


def defaults_for_tiers(*tiers: int) -> dict[str, Any]:
    """Return defaults for the given tier(s) only."""
    return {
        name: spec.default
        for name, spec in PARAMETER_SPACE.items()
        if spec.tier in tiers
    }


def normalize_weights(params: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of *params* with the composite score weights normalised to
    sum to 1.0.

    Optuna stores the raw (pre-normalisation) weight values in ``best_params``
    and ``trial.params``.  Call this before applying params to BobsBrain or
    writing them to ``active_parameters``.
    """
    params = dict(params)
    present = _WEIGHT_NAMES & set(params)
    if not present:
        return params
    all_weights = {w: params.get(w, PARAMETER_SPACE[w].default) for w in _WEIGHT_NAMES}
    total = sum(all_weights.values())
    if total > 0:
        for w in _WEIGHT_NAMES:
            if w in params:
                params[w] = all_weights[w] / total
    return params


def suggest(trial: optuna.Trial, tiers: tuple[int, ...]) -> dict[str, Any]:
    """
    Ask an Optuna trial to suggest values for parameters in *tiers*.

    Composite score weights (w_corr_long, w_corr_short, w_z_depth) are
    suggested freely and then normalised so they sum to 1.0.  If only a
    subset of the three weights is being tuned, the others take their
    defaults before normalisation.
    """
    params: dict[str, Any] = {}

    for name, spec in PARAMETER_SPACE.items():
        if spec.tier not in tiers:
            continue
        if spec.dtype == 'int':
            params[name] = trial.suggest_int(name, spec.low, spec.high, log=spec.log)
        elif spec.dtype == 'float':
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        elif spec.dtype == 'categorical':
            params[name] = trial.suggest_categorical(name, spec.choices)

    # Normalise composite score weights when any of them was suggested.
    tuned_weights = _WEIGHT_NAMES & set(params)
    if tuned_weights:
        all_weights = {
            w: params.get(w, PARAMETER_SPACE[w].default)
            for w in _WEIGHT_NAMES
        }
        total = sum(all_weights.values())
        if total > 0:
            for w in _WEIGHT_NAMES:
                if w in params:
                    params[w] = all_weights[w] / total

    return params
