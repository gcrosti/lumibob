# Parameter Tuning Engine — Design & Rollout Plan

> Living document. Last updated: 2026-04-17. Source of truth for the parameter
> tuning engine project. Update as decisions evolve.

## Goal

Build a parameter tuning engine that optimizes LumiBob's strategy parameters
based on market conditions so that LumiBob consistently outperforms the SPY
index and ultimately maximizes returns.

## Confirmed design decisions


| #   | Decision                       | Choice                                                            |
| --- | ------------------------------ | ----------------------------------------------------------------- |
| 1   | Tuning universe (early phases) | S&P 500                                                           |
| 2   | Tuning universe (later phases) | S&P 1500, validated against full Alpaca universe                  |
| 3   | Macro data source              | FRED via `fredapi` (free API key in `.env` as `FRED_API_KEY`)     |
| 4   | Tier 3 weekly param change cap | Max ±20% week-over-week per param to avoid whiplash               |
| 5   | Phase 0 (expose magic numbers) | Approved — additive, behavior-preserving                          |
| 6   | Negative correlations          | Keep clamp (no shorting) for now                                  |
| 7   | Sector gate                    | Remove after sector pre-partition (Phase 2); becomes tautological |
| 8   | Unknown-sector tickers         | Option B — their own clustering partition                         |
| 9   | Metadata staleness             | Hard-fail metadata pre-fetch if SEC coverage < 50%                |
| 10  | Cloud migration                | Stay local through Phase 3; re-evaluate before Phase 4            |
| 11  | Wall-clock cap per study       | 24 hours on local Mac; `caffeinate -i` baked into runner CLI      |


---

## Parameter inventory

### A. Currently exposed in `STRATEGY_PARAMETERS` (15 params)


| Parameter                | Default | Family      |
| ------------------------ | ------- | ----------- |
| `lookback_window`        | 130     | data        |
| `cluster_recompute_days` | None    | clustering  |
| `min_position_pct`       | 0.03    | sizing      |
| `max_position_pct`       | 0.20    | sizing      |
| `target_deployed_pct`    | 0.60    | sizing/risk |
| `entry_threshold`        | 2.0     | signal      |
| `exit_threshold`         | 0.5     | signal      |
| `zscore_window`          | 20      | signal      |
| `corr_long_window`       | 90      | scoring     |
| `corr_short_window`      | 20      | scoring     |
| `w_corr_long`            | 0.3     | scoring     |
| `w_corr_short`           | 0.5     | scoring     |
| `w_z_depth`              | 0.2     | scoring     |
| `max_daily_candidates`   | 200     | discovery   |
| `cooldown_days`          | 7       | discovery   |


### B. Hard-coded magic numbers (Phase 0 must expose)


| Location                                   | Current value                    | Proposed param name                               |
| ------------------------------------------ | -------------------------------- | ------------------------------------------------- |
| `BobsBrain.before_market_opens` L297, L304 | `< 5`                            | `penny_threshold`                                 |
| `BobsBrain.before_market_opens` L391       | `0.7` (pivot)                    | `quality_scale_pivot`                             |
| `BobsBrain.before_market_opens` L391       | `[0.5, 1.5]`                     | `quality_scale_min`, `quality_scale_max`          |
| `TickerClusterer.__init__` L48             | `lookback_days=126`              | `cluster_lookback_days`                           |
| `TickerClusterer.__init__` L49             | `min_cluster_size=5`             | `hdbscan_min_cluster_size`                        |
| `TickerClusterer.__init__` L50             | `pca_variance=0.95`              | `pca_variance`                                    |
| `TickerClusterer._compute` L196            | `0.5`                            | `min_coverage`                                    |
| `TickerClusterer._compute` L213            | `min_samples=2`                  | `hdbscan_min_samples`                             |
| `TickerClusterer._compute` L215            | `metric='euclidean'`             | `hdbscan_metric`                                  |
| `TickerClusterer._compute` L216            | `cluster_selection_method='eom'` | `hdbscan_selection_method`                        |
| *(new)*                                    | —                                | `hdbscan_cluster_selection_epsilon` (default 0.0) |


### C. Operational parameters (run-level, not strategy-level)

`budget`, `backtesting_start`, `backtesting_end`, universe filter, slippage
model, bar resolution. Owned by `main.py` and the tuner orchestrator, not by
`BobsBrain`.

### D. Out of scope

Cointegration p-value (legacy), `PairSimulator` MA/lag grids (off live path).

---

## Parameter classification by adaptivity

### Tier 1 — Constant (set once, audit annually)

`lookback_window`, `cooldown_days`, `pca_variance`, `min_coverage`,
`penny_threshold`, `hdbscan_metric`, `hdbscan_selection_method`, composite
score's negative-correlation clamp policy.

### Tier 2 — Slow-adaptive (re-tune monthly to quarterly)

`hdbscan_min_cluster_size`, `hdbscan_min_samples`, `hdbscan_cluster_selection_epsilon`,
`cluster_recompute_days`, `corr_long_window`, `corr_short_window`,
`w_corr_long`, `w_corr_short`, `w_z_depth`, `max_daily_candidates`,
`target_deployed_pct`, `max_k`.

### Tier 3 — Fast-adaptive (re-tune weekly, regime-conditioned)

`entry_threshold`, `exit_threshold`, `zscore_window`, `min_position_pct`,
`max_position_pct`, `quality_scale_pivot`.

### Regime conditioning features (Tier 3 inputs)

- **VIX** level + 1m change (CBOE)
- **SPY** 20d realized vol
- **SPY** 50d/200d trend slope
- Cross-sectional **dispersion** of S&P 500 returns
- Average pairwise **correlation** of S&P 500
- **FRED**: `T10Y2Y` (yield curve), `BAMLH0A0HYM2` (HY credit spread), `DFF` (fed funds)

---

## Methodology

### Walk-forward harness (mandatory, all tiers)

```
[--- train 12mo ---][holdout 3mo][--- train 12mo ---][holdout 3mo]...
                      ^                                ^
              tune here, evaluate here       slide 3mo, repeat
```

### Objective function

```
score = (mean_oos_return - rf) / oos_vol  -  λ * max_drawdown  -  μ * trade_count_penalty
        ─────────────────── Sharpe ───────  ── DD penalty ──   ── overfitting guard ──
```

**Hard constraint**: out-of-sample return > SPY return for ≥ 60% of holdout
windows. Implements the "consistently outperform SPY" requirement directly.

### Tuning cadence


| Layer             | Cadence        | Method                                                  |
| ----------------- | -------------- | ------------------------------------------------------- |
| Tier 1            | Annual / on PR | Coarse grid + manual review                             |
| Tier 2            | Monthly        | Optuna TPE on 12mo rolling window                       |
| Tier 3            | Weekly         | Regime-conditioned lookup table built from Tier 2 study |
| Composite weights | Quarterly      | Constrained optimization (sum=1, all ≥ 0)               |


### Sensitivity / stability checks (catch overfit early)

- **±10% perturbation** around each chosen parameter — objective should degrade gracefully
- **HDBSCAN re-seed** with `min_cluster_size ± 1` — clusters should be stable
- **Subset bootstrap** — re-run on 5 random 80% subsets of universe

---

## HDBSCAN — making clusters meaningful

### Fixes (priority order)

1. **Distance metric: correlation distance `1 - ρ`** (currently euclidean).
  Use `metric='precomputed'` with the correlation matrix we already compute.
   Typical noise-rate reduction: 30-50%.
2. **Joint-tune `min_cluster_size`, `min_samples`, `pca_variance`** via two-objective Pareto study:
  ```
   maximize:    coverage = 1 - noise_fraction
   subject to:  median_intra_cluster_corr >= 0.4
                median_cluster_size in [8, 50]
  ```
   Search: `min_cluster_size ∈ [3, 15]`, `min_samples ∈ [1, 5]`, `pca_variance ∈ [0.80, 0.99]`.
3. **Replace Ward fallback with `cluster_selection_epsilon` relaxation** — keeps
  density-based topology instead of forcing K equal-size clusters.
4. **Per-cluster sanity gate** — drop clusters with median intra-corr < 0.3,
  send members back to noise tail.
5. **Sector pre-partition** — cluster within SIC sector first, ETFs separately,
  unknown-sector as Option B (their own partition). Hardens metadata fetch
   to fail-loud at <50% coverage.

### Recommended starting config (pre-tuning baseline)

```python
hdbscan.HDBSCAN(
    min_cluster_size=4,
    min_samples=1,
    metric='precomputed',
    cluster_selection_method='leaf',
    cluster_selection_epsilon=0.3,
    allow_single_cluster=False,
)
```

---

## Test battery (5 standard regimes)


| Regime               | Window            | Tests                               |
| -------------------- | ----------------- | ----------------------------------- |
| Calm bull            | 2017-01 → 2017-12 | Underdeployment risk                |
| Vol shock            | 2020-02 → 2020-06 | Drawdown control                    |
| Sideways high-vol    | 2022-01 → 2022-12 | Pairs strategy's natural habitat    |
| Trend-following bull | 2023-04 → 2023-12 | Cluster gate / sector gate behavior |
| Mixed recent         | 2024-01 → 2024-09 | OOS vs. anything trained on ≤2023   |


**Pass criterion**: beats SPY in **≥ 4 of 5 regimes**, never has worse DD than
SPY by more than 2pp.

---

## External data sources


| Source                                     | Use                     | Cadence            |
| ------------------------------------------ | ----------------------- | ------------------ |
| **CBOE VIX**                               | Vol regime              | Daily              |
| **FRED** (`T10Y2Y`, `BAMLH0A0HYM2`, `DFF`) | Macro regime            | Daily              |
| **CBOE SKEW**                              | Tail risk               | Daily              |
| **SPY/QQQ/IWM**                            | Trend, dispersion, beta | Daily              |
| **SEC EDGAR** (already have)               | Sector partition        | One-time + nightly |
| **Fama-French factors**                    | Factor regime           | Weekly             |


---

## Architecture

New top-level module: `tuning/`

```
tuning/
├── __init__.py
├── parameter_space.py      # Single source of truth: every param + bounds + tier
├── regime_detector.py      # FRED + VIX + SPY → regime feature vector & label
├── objective.py            # Backtest harness wrapper + scoring function
├── walk_forward.py         # Rolling window orchestrator
├── universe.py             # S&P 500 / 1500 constituent loaders
├── studies/
│   ├── tier1_constants.py
│   ├── tier2_slow.py
│   ├── tier3_fast.py
│   └── hdbscan_study.py
├── battery.py              # 5-regime standard test battery
├── runner.py               # CLI: caffeinate-wrapped study launcher
└── parameter_store.py      # Reads/writes active_parameters; enforces ±20%/wk cap
```

### New tables

```sql
CREATE TABLE tuning_studies (
    study_id            SERIAL PRIMARY KEY,
    tier                INT,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    train_start         DATE, train_end  DATE,
    holdout_start       DATE, holdout_end DATE,
    best_params         JSONB,
    objective           NUMERIC,
    holdout_metrics     JSONB,
    -- compute receipt
    n_trials_completed  INT,
    n_trials_pruned     INT,
    wall_clock_seconds  INT,
    parallel_jobs       INT,
    machine             VARCHAR(50),
    estimated_cost_usd  NUMERIC
);

CREATE TABLE active_parameters (
    effective_date      DATE PRIMARY KEY,
    regime_label        INT,
    params              JSONB,
    source_study_id     INT REFERENCES tuning_studies(study_id)
);
```

The strategy's `BobsBrain.initialize` adds **one line** to load
`active_parameters` for today's regime. Tuner reuses existing
`BobsBrain.backtest(...)` entry point — no strategy duplication.

---

## Compute discipline

### Three first-class mechanisms (built in from Phase 1)

1. `**budget_hours` on every study** — Optuna `study.optimize(timeout=budget_hours*3600)`.
  Always returns a result; precision scales with budget.
2. **Aggressive pruning** — `MedianPruner` with monthly intermediate reports
  from `BobsBrain` via `trial.report(interim_sharpe, step=month)`. ~2-4x
   effective speedup.
3. **Compute receipt** — every study writes wall-clock, trials, parallelism,
  and cost into `tuning_studies`. After 5-10 studies we can plot marginal
   Sharpe gain per trial → data-driven scaling decisions.

### Cloud migration triggers (only move when one fires)

- Tier 4 walk-forward exceeds **48 hours** locally
- Want to keep laptop free for normal work during tuning
- Multiple competing studies in parallel
- Live trading workload competes with tuning

Until then: stay local, pre-warm `StockDataCache`, parallelize Optuna with
`n_jobs = physical_cores - 2`.

---

## Phased rollout

Each phase has an explicit compute envelope and a gate criterion. **Always
start coarse; densify only if the gate is met.**

> **Order change (2026-04-18):** Phase 2 moved before Phase 1. The sector
> gate bottleneck makes warm-cache trial time ~5.5h, which makes any Optuna
> study impractical until Phase 2 eliminates it. Phase 2 also adds the
> `max_k` K-ballooning fix, which is required for the battery (Phase 3) and
> all subsequent backtests to produce meaningful signals.


| Phase           | Deliverable                                                                                                                                              | Active work | Backtest wait | Compute envelope | Gate to advance                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------- | ---------------- | ------------------------------------------------------------------------ |
| **0** ✓         | Expose magic numbers as parameters; verify no behavior change                                                                                            | 2-3 hrs     | 5.5 hrs       | trivial          | Verification backtest passed (e52b20 vs b94e19 baseline)                 |
| **2**           | HDBSCAN (corr distance + sector pre-partition + Option B unknown bucket) + remove sector/ETF gate + harden metadata pre-fetch + `max_k` K-ballooning fix | 1-2 days    | ~30-60 min    | ~2 hrs           | Cluster noise rate drops measurably; trial time < 1hr; K stays bounded   |
| **1**           | Tuning harness skeleton (`parameter_space`, `objective`, `walk_forward`); one Optuna proof study on Tier 2, single 12mo window                           | 1 day       | 1 hr          | 1 hr             | Tuner produces param set that beats default on holdout, even marginally  |
| **3**           | 5-regime battery + comparison harness vs. baseline                                                                                                       | half day    | 2 hrs         | 2 hrs            | Battery distinguishes a known-good vs. known-bad param set (calibration) |
| **4 (coarse)**  | Regime detector (FRED + VIX + dispersion) + Tier 3 lookup table; 12 folds × 50 trials = 600 backtests                                                    | half day    | ~24 hrs       | 24 hrs           | Holdout Sharpe distribution beats baseline at p<0.10                     |
| **4.5 (dense)** | Densify to 36 folds × 100 trials = 3,600 backtests                                                                                                       | minimal     | ~72 hrs       | 72 hrs           | Same gate at p<0.05                                                      |
| **5**           | Scheduled tuning job → `active_parameters`; strategy reads from there; ±20% week-over-week cap                                                           | 1 day       | —             | 6 hrs/month      | Live shadow run beats baseline for 4 consecutive weeks                   |


**Total minimum wall-clock**: ~10 days continuous, including 3 explicit
decision points before any large compute spend.

### Decision points (no/no-go gates)

1. **After Phase 2** — Did trial time drop to < 1hr? Does cluster noise rate
  fall measurably? Does K stay bounded with `max_k`? If no on any: fix
   before Phase 1.
2. **After Phase 1** — Did the proof study produce sensible suggestions? If
  no: fix objective/space before continuing.
3. **After Phase 3** — Do battery results validate Phase 2 changes? Compare
  to pre-Phase-2 baseline. If regression: fix before Phase 4.
4. **After Phase 4 coarse** — Does coarse run beat baseline at p<0.10? If
  no: stop, rethink objective or features. Only densify (Phase 4.5) if yes.

---

## Benchmark timing (measured 2026-04-17)

Jan 2 → Mar 25 2024, full Alpaca universe (~4,900 symbols), 58 trading days:


| Cache state      | Wall-clock       | Notes                                             |
| ---------------- | ---------------- | ------------------------------------------------- |
| Cold (first run) | ~4.7 hours       | All prices fetched from Alpaca; one-time cost     |
| **Warm**         | **~3–7 minutes** | Prices already in DB; all tuning trials land here |


**Implications for the tuning engine:**

- Pre-warm `StockDataCache` once per tuning window before Optuna starts
- 100 trials × ~5 min / 4 parallel jobs ≈ **2 hours per study**
- Phase 4 dense (3,600 trials / 4 jobs) ≈ **20–30 hours** — feasible locally overnight + weekend

## Known concerns / observations

### K-ballooning (observed 2026-04-17; fix scheduled for Phase 2)

During the Phase 0 verification backtest (Jan–Mar 2024, full Alpaca universe),
target K grew from 10 on day 1 to 1,061 by the end, with composite scores as
low as 0.28 and `z_depth=0.00` (no actual spread signal). Pair quality (median
`corr_short`) degraded from 0.898 to 0.251 by end of run.

**Root cause:** The line `k_target = max(k_target, len(existing_scored))` is a
ratchet — K can never shrink below the current number of held positions. Once
accumulated, pairs are never displaced unless their individual z-score exits.
The strategy effectively becomes an unweighted index of noisy pairs.

**Fix (added to Phase 2):** Replace the entire `k_base` cash-derived formula
with a direct, interpretable expression. `k_base` conflated two questions —
"how many positions can I afford?" (answered by the buy loop's existing cash
check) and "what is the target portfolio size?" (answered by `max_k`).
Separating them makes K interpretable and removes the ratchet:

```python
# Before — complex, contradictory, with ratchet:
target_pos_pct = (min_position_pct + max_position_pct) / 2
k_base = max(1, int(available_cash / (target_pos_pct * portfolio_value)))
quality_scale = clamp(pool_corr / quality_scale_pivot, quality_scale_min, quality_scale_max)
k_target = max(1, round(k_base * quality_scale))
k_target = max(k_target, len(existing_scored))   # ratchet — never shrinks

# After — two lines, directly interpretable:
quality_scale = clamp(pool_corr / quality_scale_pivot, quality_scale_min, quality_scale_max)
k_target = max(1, round(max_k * quality_scale))  # K floats between max_k×min and max_k
```

`quality_scale` now bounds K between `max_k × quality_scale_min` and `max_k`
based on daily pool quality. The buy loop's existing `if available_cash <
per_stock_budget: continue` already enforces affordability — no need to encode
it in K. Genuine rotation is now possible: positions ranked below `k_target`
in composite score get displaced and sold each day.

**Why Phase 2, not Phase 0:** This is a behavioral change (not just a parameter
expose), so it belongs in its own phase with a before/after comparison. Without
it, the Phase 3 battery cannot produce meaningful signals — all runs degenerate
into the same K-bloated state regardless of other parameter choices.

### Sector gate as discovery bottleneck (observed 2026-04-17, Phase 0 run)

388,924 sector-gate rejections in 17 simulated days (~~23k pair checks/day).
The inner discovery loop scans almost every within-cluster combination before
finding 200 that pass. This is the dominant cost of each simulated trading day
on the warm-cache path (~~7 hours total for 58 days vs. the expected 30–60 min
after Phase 2). Phase 2 sector pre-partition eliminates this entirely.

## Open questions still to resolve

- **Cloud spec** (only relevant if a cloud trigger fires): preferred provider
(AWS / Hetzner / DigitalOcean), preferred Postgres hosting (managed RDS /
self-hosted on same VM / Supabase).
- **Live shadow run mechanism** (Phase 5): how to run a "shadow" parameter set
in parallel with the live strategy without doubling Alpaca rate-limit cost.
Probably: same `BobsBrain` instance computes both targets, but only the
active one places orders, and the shadow's would-be PnL is logged.

---

## Glossary

- **Optuna** — Bayesian hyperparameter optimizer; uses TPE (Tree-structured
Parzen Estimator) to sample preferentially from promising regions of the
search space. Supports pruning, parallel trials, persistent storage,
multi-objective Pareto fronts. Replaces grid/random search.
- **Walk-forward** — Train on a window, evaluate on the following holdout,
slide forward, repeat. Standard time-series ML protocol; prevents future
leakage.
- **Regime** — A discrete market state (vol level, trend, dispersion) used to
condition fast-adaptive parameters.
- **Pruning** — Aborting an Optuna trial early when intermediate scores show
it's clearly underperforming the median completed trial.
- **Pareto front** — Set of solutions where no objective can be improved
without worsening another. Used for HDBSCAN coverage-vs-quality trade-off.

