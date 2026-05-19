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
        'lookback_window', tier=2, default=130, low=60, high=252, dtype='int',
    ),
    'cooldown_days': ParamSpec(
        'cooldown_days', tier=2, default=7, low=3, high=21, dtype='int',
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
    # H5: cointegration quality and mean-reversion speed weights.
    # low=0.10 on w_coint prevents Optuna from zeroing out cointegration signal.
    'w_coint': ParamSpec(
        'w_coint', tier=2, default=0.25, low=0.10, high=1.0,
    ),
    'w_halflife': ParamSpec(
        'w_halflife', tier=2, default=0.15, low=0.0, high=1.0,
    ),
    # Half-life normalisation ceiling (tunable so Optuna can widen or narrow the
    # scoring window without changing the formula structure).
    'max_halflife_days': ParamSpec(
        'max_halflife_days', tier=2, default=60, low=20, high=120, dtype='int',
    ),
    # Dynamic: TickerClusterer._compute() overwrites this with max(_MCS_FLOOR,
    # round(universe_size * _MCS_FRACTION)) on every cluster rebuild, so Optuna
    # suggestions have no effect on clustering.  Tier 1 — structural, not tunable.
    'hdbscan_min_cluster_size': ParamSpec(
        'hdbscan_min_cluster_size', tier=1, default=5, low=3, high=15, dtype='int',
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

    # Fraction of long notional to short the lead stock.
    # 0.0 = long-only (default); 1.0 = full dollar-neutral hedge.
    # Replaces the deprecated boolean enable_short_leg.
    # Regime-conditioned: expected to be higher in bear/stress regimes.
    'short_leg_fraction': ParamSpec(
        'short_leg_fraction', tier=3, default=0.0, low=0.0, high=1.0,
    ),
    'entry_threshold': ParamSpec(
        'entry_threshold', tier=3, default=2.0, low=1.0, high=3.5,
    ),
    'exit_threshold': ParamSpec(
        'exit_threshold', tier=3, default=0.5, low=0.1, high=1.5,
    ),
    'zscore_window': ParamSpec(
        'zscore_window', tier=2, default=20, low=10, high=40, dtype='int',
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

_WEIGHT_NAMES = frozenset({'w_corr_long', 'w_corr_short', 'w_z_depth', 'w_coint', 'w_halflife'})


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

    Composite score weights (w_corr_long, w_corr_short, w_z_depth, w_coint,
    w_halflife) are suggested freely and then normalised so they sum to 1.0.
    If only a subset of the five weights is being tuned, the others take their
    defaults before normalisation.

    Joint timescale constraints (Study 1 Pass A):
    When the following pairs are both being tuned in the same call, the
    dependent parameter is suggested with a derived bound so the combination
    is always coherent:

        zscore_window  ≤ lookback_window // 3
            → zscore_window sampled from [10, min(40, lookback_window // 3)]

        cooldown_days  ≥ zscore_window // 2
            → cooldown_days sampled from [max(3, zscore_window // 2), 21]

        corr_short_window ≤ corr_long_window
            → corr_short_window sampled from [5, min(40, corr_long_window)]

    These constraints are enforced by skipping the three dependent params in
    the main loop and suggesting them afterwards in dependency order.
    """
    # Params whose valid range depends on another param suggested in the same
    # call.  They are excluded from the main loop and handled below.
    _DEFER = frozenset({'zscore_window', 'cooldown_days', 'corr_short_window'})

    params: dict[str, Any] = {}

    for name, spec in PARAMETER_SPACE.items():
        if spec.tier not in tiers or name in _DEFER:
            continue
        if spec.dtype == 'int':
            params[name] = trial.suggest_int(name, spec.low, spec.high, log=spec.log)
        elif spec.dtype == 'float':
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        elif spec.dtype == 'categorical':
            params[name] = trial.suggest_categorical(name, spec.choices)

    # --- Joint constraints: suggest deferred params in dependency order ---

    # 1. zscore_window depends on lookback_window.
    zw_spec = PARAMETER_SPACE['zscore_window']
    if zw_spec.tier in tiers:
        if 'lookback_window' in params:
            zw_high = min(zw_spec.high, params['lookback_window'] // 3)
            zw_high = max(zw_high, zw_spec.low)  # guard against degenerate range
        else:
            zw_high = zw_spec.high
        params['zscore_window'] = trial.suggest_int(
            'zscore_window', zw_spec.low, zw_high,
        )

    # 2. cooldown_days depends on zscore_window (which may have just been set).
    cd_spec = PARAMETER_SPACE['cooldown_days']
    if cd_spec.tier in tiers:
        if 'zscore_window' in params:
            cd_low = max(cd_spec.low, params['zscore_window'] // 2)
            cd_low = min(cd_low, cd_spec.high)  # guard against degenerate range
        else:
            cd_low = cd_spec.low
        params['cooldown_days'] = trial.suggest_int(
            'cooldown_days', cd_low, cd_spec.high,
        )

    # 3. corr_short_window depends on corr_long_window.
    csw_spec = PARAMETER_SPACE['corr_short_window']
    if csw_spec.tier in tiers:
        if 'corr_long_window' in params:
            csw_high = min(csw_spec.high, params['corr_long_window'])
            csw_high = max(csw_high, csw_spec.low)  # guard against degenerate range
        else:
            csw_high = csw_spec.high
        params['corr_short_window'] = trial.suggest_int(
            'corr_short_window', csw_spec.low, csw_high,
        )

    # --- Normalise composite score weights when any of them was suggested ---
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
