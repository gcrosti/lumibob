# Tier 3 Parameters + Regime Detector — Implementation Plan

> Created: 2026-05-18. Supersedes the Tier 3 and Phase 4 sections of `TUNING_ENGINE_PLAN.md`.

---

## Context

Phases 0–Pre-4 of the tuning engine are complete. The cointegration gate (H5) is
validated. The short leg (H1) has been implemented as a boolean (`enable_short_leg`)
and empirically tested: the binary switch destroys value in bull markets and in sideways
markets alike, because regime drives whether the short leg earns or costs. A continuous
`short_leg_fraction` parameter, conditioned on regime, is the correct replacement.

The coint cache write bug is fixed; `zscore_window` is now Tier 1 (jointly optimized
with `lookback_window`). The strategy is ready for regime-conditioned Tier 3 tuning.

---

## Step 1 — Implement `short_leg_fraction`

**What it is:** A float in [0.0, 1.0] that scales the short position size relative to
the long leg. `0.0` = long-only (current baseline). `1.0` = full hedge (current H1).
Values between allow partial exposure.

**Code changes:**

- `BobsBrain.initialize()`: read `short_leg_fraction = float(self.parameters.get('short_leg_fraction', 0.0))`. Deprecate `enable_short_leg` (keep reading it for backward compatibility: if `enable_short_leg=True` and `short_leg_fraction` not set, default to `1.0`).
- `BobsBrain.on_trading_iteration()` (buy path, line ~700): replace the boolean short-leg block with `short_qty = round(lead_qty * short_leg_fraction)`. Only open the short if `short_qty > 0`.
- `BobsBrain.on_trading_iteration()` (effective cost, line ~682): `effective_cost = per_stock_budget * (1 + short_leg_fraction)` instead of `* 2`.
- `backtest_runs.settings`: include `short_leg_fraction` in the settings dict written at run start.

**Parameter space:**

```python
'short_leg_fraction': ParamSpec(
    'short_leg_fraction', tier=3, default=0.0, low=0.0, high=1.0,
    description='Fraction of long position size to short the lead stock. '
                '0=long-only, 1=full hedge. Regime-conditioned.',
),
```

**Gate:** Run a 2-window validation backtest (sideways 2022 + bull 2023) with
`short_leg_fraction` as the only free parameter, all others at current defaults.
Confirm that the optimizer finds a value strictly between 0 and 1 in at least one
regime — if it snaps to 0 in both, partial hedging adds no value and the parameter
can be dropped before building the full regime machinery.

---

## Step 2 — Regime Detector

### Feature vector

Fetched daily before market open, stored in a local cache. All features are
backward-looking (no lookahead).

| Feature | Source | Computation |
|---|---|---|
| `vix_level` | CBOE VIX (via `yfinance` or FRED `VIXCLS`) | Raw close |
| `vix_1m_change` | Same | (today − 21 days ago) / 21d ago |
| `spy_20d_vol` | SPY daily returns | Rolling 20d realized vol (annualized) |
| `spy_trend` | SPY close | Sign of (50d EMA − 200d EMA) slope |
| `spy_20d_return` | SPY close | 20d return |
| `yield_curve` | FRED `T10Y2Y` | Raw value (negative = inverted) |
| `hy_spread` | FRED `BAMLH0A0HYM2` | Raw value |
| `xsec_dispersion` | SPY constituent returns | Std dev of 20d returns across S&P 500 |

### Regime labels

Start with a **rule-based** detector (no ML, no fitting risk) with 4 labels.
Fit a learned detector (k-means or HMM on the feature vector) only if the
rule-based labels fail the stability gate.

| Label | Name | Conditions |
|---|---|---|
| 0 | **Sideways / mean-reverting** | `spy_trend ≤ 0` AND `vix_level < 25` |
| 1 | **Bull trending** | `spy_trend > 0` AND `spy_20d_return > 0` AND `vix_level < 25` |
| 2 | **High vol / stress** | `vix_level ≥ 25` OR `hy_spread > 4.5` |
| 3 | **Bear trending** | `spy_trend ≤ 0` AND `spy_20d_return < -0.05` |

**Stability gate:** Label transitions should not flip more than once per week on
average across 2020–2024 history. Flipping more often than that means the thresholds
are too sensitive to noise; adjust or smooth with a 3-day minimum-hold rule.

### Implementation

New module: `tuning/regime_detector.py`

```python
class RegimeDetector:
    def __init__(self, fred_api_key: str): ...
    def get_label(self, as_of: date) -> int:
        """Return regime label for the given date. Cached per date."""
    def get_feature_vector(self, as_of: date) -> dict[str, float]:
        """Raw features for a date — used for diagnostics and learned detector."""
    def label_history(self, start: date, end: date) -> pd.Series:
        """Time series of regime labels — used to build Tier 3 lookup tables."""
```

`BobsBrain.initialize()` instantiates `RegimeDetector` and stores `self._regime_label`
for the run. In a backtest, the label is set once at `initialize()` time (the regime
the strategy was tuned for); the detector runs live in paper/live mode.

`active_parameters.regime_label` is populated from this label when params are written.

---

## Step 3 — Tier 3 Parameter Set

Final Tier 3 inventory — all regime-conditioned, re-tuned weekly in production.

| Parameter | Default | Range | Rationale |
|---|---|---|---|
| `short_leg_fraction` | 0.0 | [0.0, 1.0] | Core new param: partial hedge scaled by regime |
| `entry_threshold` | 2.0 | [1.2, 3.0] | Wider in high-vol regimes; tighter in sideways |
| `exit_threshold` | 0.5 | [0.1, 1.5] | Interacts with entry; must be tuned jointly |
| `min_position_pct` | 0.03 | [0.01, 0.08] | More concentrated in high-conviction regimes |
| `max_position_pct` | 0.20 | [0.05, 0.30] | Hard ceiling per position |
| `quality_scale_pivot` | 0.7 | [0.4, 1.0] | Affects how quickly K scales with pool quality |

**Not Tier 3:** `zscore_window` was promoted to Tier 1 (must be jointly optimized
with `lookback_window`, not conditioned per-regime independently).

**Joint constraint:** `entry_threshold` and `exit_threshold` must always satisfy
`exit_threshold < entry_threshold`. Enforce in the Optuna `suggest_*` calls:
suggest `exit_threshold` as a fraction of `entry_threshold`.

---

## Step 4 — Optuna Study Plan

### Study 0: `short_leg_fraction` validation (run first, gate for Study 1)

**Purpose:** Confirm that a fractional short leg adds value before building full
regime machinery around it.

| Setting | Value |
|---|---|
| Free params | `short_leg_fraction` only |
| Fixed params | Current best-known defaults |
| Folds | sideways_2022 (2022-02 → 2022-04), bull_2023 (2023-04 → 2023-06) |
| Trials | 30 per fold (60 total) |
| Objective | Sharpe (same formula as existing `objective.py`) |
| Wall-clock | ~2 hrs (warm cache) |

**Gate:** If best `short_leg_fraction > 0.05` in at least one fold → proceed to
Study 0.5. If it snaps to 0 in all folds → the short leg hypothesis is dead; do not
build regime machinery around it.

---

### Study 1: Tier 2 two-pass joint optimization

`lookback_window`, `zscore_window`, `cooldown_days`, and all five composite score
weights are all Tier 2. They are tightly coupled — a weight is only meaningful
relative to the signal generated at the timescale that produces it. Tuning them
in separate studies would find a locally consistent but globally suboptimal set.
This study has no regime conditioning: Tier 2 params are held constant across
regimes and re-tuned monthly as market structure slowly shifts.

Run across a wide multi-regime window (sideways, bull, mixed) so the optimizer
finds timescales and weights that are robust rather than overfit to one condition.

#### Pass A — Signal construction

All signal construction params free simultaneously. Constrain the search space
to enforce coherent timescale relationships.

| Setting | Value |
|---|---|
| Free params | `lookback_window` [60, 252], `zscore_window` [10, 40], `cooldown_days` [3, 21], `w_corr_long` [0, 1], `w_corr_short` [0, 1], `w_z_depth` [0, 1], `w_coint` [0.1, 1], `w_halflife` [0, 1], `corr_long_window` [60, 180], `corr_short_window` [5, 40] |
| Fixed params | All Tier 1 and Tier 3 at current defaults |
| Window | 2022-01 → 2024-06 (covers sideways, bull, and mixed regimes) |
| Trials | 200 |
| Objective | Sharpe on a 3-month rolling OOS holdout (walk-forward, 3 folds) |
| Wall-clock | ~6 hrs (warm cache, 10 free params × 200 trials ÷ parallelism) |

**Joint constraints:**
- `zscore_window ≤ lookback_window / 3` — z-score window must be shorter than a
  third of the cointegration lookback; suggest as `[10, min(40, lookback_window // 3)]`
- `cooldown_days ≥ zscore_window / 2` — cooldown must allow the spread time to
  reset; suggest as `[max(3, zscore_window // 2), 21]`
- `corr_short_window ≤ corr_long_window` — suggest as `[5, min(40, corr_long_window)]`
- Weights normalised to sum to 1.0 via `normalize_weights()` after suggestion

#### Pass B — Position and discovery params

Signal params fixed at Pass A best-trial output. Remaining Tier 2 params free.

| Setting | Value |
|---|---|
| Free params | `hdbscan_min_cluster_size` [3, 15], `hdbscan_min_samples` [1, 5], `hdbscan_cluster_selection_epsilon` [0, 0.5], `cluster_recompute_days`, `max_daily_candidates` [50, 300], `target_deployed_pct` [0.3, 0.9], `max_k` |
| Fixed params | Pass A outputs + all Tier 1 and Tier 3 defaults |
| Window | Same 2022-01 → 2024-06 |
| Trials | 150 |
| Wall-clock | ~4 hrs |

**Gate:** Pass A best-trial OOS Sharpe must beat the default param baseline on the
same 3-fold walk-forward. If not — widen to 300 trials before proceeding to Pass B.
Pass B has no independent gate; its output feeds directly into Study 2.

---

### Study 2: Per-regime Tier 3 joint optimization (Phase 4 coarse equivalent)

**Purpose:** For each regime label, find the jointly optimal Tier 3 parameter set.

**Setup:**

- Label 2020–2024 history with the regime detector.
- Split into 12 non-overlapping 3-month folds, each tagged with its dominant regime label.
- For each fold: run Optuna TPE with all 6 Tier 3 params free, Tier 1 and Tier 2 params fixed at Study 1 outputs.
- Separate Optuna studies per regime label so the optimizer builds independent param distributions per regime.

| Setting | Value |
|---|---|
| Folds | 12 × 3-month windows (2022-01 → 2024-12) |
| Trials per fold | 50 |
| Total trials | 600 |
| Parallelism | `n_jobs = physical_cores − 2` |
| Wall-clock estimate | ~15 hrs (all folds warm, max_daily_candidates ≤ 300) |
| Pruner | `MedianPruner(n_startup_trials=5, n_warmup_steps=1)` |
| Per-trial timeout | 20 min hard kill |

**Objective:**

```
score = annualized_sharpe  −  0.5 * max_drawdown_pct  −  0.1 * trade_count_penalty
```

Same formula as existing `objective.py`. No change needed.

**Gate (Phase 4 coarse):** Regime-conditioned params beat a static param set
(Phase 3 best-trial, held constant across all 12 folds) in ≥ 8 of 12 folds at
the fold level, and in aggregate Sharpe at p < 0.10 (permutation test).
If not met: regime detector labels are not discriminative enough — revisit
feature thresholds or try a 2-label simplification (risk-on / risk-off only).

---

### Study 3: Dense walk-forward (Phase 4.5 equivalent — launch only if Study 2 gate passes)

**Purpose:** Densify the per-regime search and validate out-of-sample on 2024–2025 holdout.

| Setting | Value |
|---|---|
| Folds | 36 × 3-month windows (2022-01 → 2025-12) |
| Trials per fold | 100 |
| Total trials | 3,600 |
| Holdout | 2025-01 → 2025-12 (unseen, not used in any training fold) |
| Wall-clock estimate | ~75 hrs — Friday-night launch |

**Gate:** Positive Sharpe on the 2025 holdout in ≥ 3 of the 4 quarterly sub-windows.
SPY-beating not required at this stage — that gate is reserved for paper trade validation.

---

## Implementation Order

1. `short_leg_fraction` in `BobsBrain` + `parameter_space.py` (replaces boolean)
2. Run Study 0 — `short_leg_fraction` validation gate
3. If gate passes: run Study 1 Pass A — signal construction (timescales + weights)
4. Run Study 1 Pass B — position and discovery params, signal params fixed from Pass A
5. `tuning/regime_detector.py` (rule-based, 4 labels, FRED + SPY features)
6. Stability check: label 2020–2024 history, verify < 1 flip/week average
7. Wire `RegimeDetector` into `BobsBrain.initialize()` and `active_parameters`
8. Run Study 2 (coarse per-regime Tier 3 optimization, Tier 2 fixed at Study 1 outputs)
9. If gate passes: run Study 3 (dense, Friday night)
10. Update `parameter_store.py` to serve per-regime params from `active_parameters`
