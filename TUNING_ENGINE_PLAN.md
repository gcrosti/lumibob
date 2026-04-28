# Parameter Tuning Engine — Design & Rollout Plan

> Living document. Last updated: 2026-04-17. Source of truth for the parameter
> tuning engine project. Update as decisions evolve.

## Goal

Build a parameter tuning engine that optimizes LumiBob's strategy parameters
based on market conditions so that LumiBob consistently outperforms the SPY
index and ultimately maximizes returns.

> **Goal revision (2026-04-23):** Phase 3.5 deep dive identified a structural constraint —
> the strategy is long-only with beta ~0.5–0.8 vs SPY. A long-only anti-momentum book cannot
> beat SPY consistently through parameter tuning alone, regardless of regime conditioning.
> **Interim goal for Phase 4:** positive absolute returns with Sharpe > 0 across all three
> test regimes. The SPY-beating goal is reinstated once a short leg is added (H1 fix),
> making the strategy genuinely market-neutral. The short leg is being built in parallel
> with Phase 4 and is targeted for completion before Phase 4.5 (dense study) launches.

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

**Hard constraint (Phase 4 coarse/dense, pre-short-leg):** out-of-sample Sharpe > 0 across
all three test regimes. The original SPY-beating constraint is suspended until the short leg
(H1 fix) is implemented — a long-only strategy cannot meet it structurally. The SPY constraint
is reinstated for Phase 4.5 if the short leg is in place by then, or for Phase 5 otherwise.

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

## Test battery (3-regime calibration set — Phase 3)

Original plan called for 5 regimes. Trimmed to 3 after observing 9–17 h
wall-clock per best-trial run; `trend_bull_2023` and `mixed_2024` deferred to
Phase 4 (added as mandatory folds there).


| Regime            | Window            | Tests                            | Best-trial run                  | Best-trial result    |
| ----------------- | ----------------- | -------------------------------- | ------------------------------- | -------------------- |
| Calm bull         | 2017-01 → 2017-12 | Underdeployment risk             | `3f7def`                        | −37.8% vs SPY +19.0% |
| Vol shock         | 2020-02 → 2020-06 | Drawdown control                 | `feac3e` (partial, 62/104 days) | +0.3% vs SPY −10.8%  |
| Sideways high-vol | 2022-01 → 2022-12 | Pairs strategy's natural habitat | `815f18`                        | −22.1% vs SPY −13.7% |


*Deferred regimes (Phase 4 mandatory folds):*


| Regime               | Window            | Reason deferred                             |
| -------------------- | ----------------- | ------------------------------------------- |
| Trend-following bull | 2023-04 → 2023-12 | Wall-clock budget; redundant for gate check |
| Mixed recent         | 2024-01 → 2024-09 | OOS signal better used in Phase 4 holdouts  |


**Phase 3 gate criterion**: best-trial composite score beats baseline in **≥ 2 of 3**
completed regimes (lower bar reflects trimmed set; full 5-regime gate applies in Phase 4).

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


| Phase           | Deliverable                                                                                                                                                                                                             | Active work | Backtest wait  | Compute envelope | Gate to advance                                                                             |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | -------------- | ---------------- | ------------------------------------------------------------------------------------------- |
| **0** ✓         | Expose magic numbers as parameters; verify no behavior change                                                                                                                                                           | 2-3 hrs     | 5.5 hrs        | trivial          | Verification backtest passed (e52b20 vs b94e19 baseline)                                    |
| **2** ✓         | HDBSCAN (corr distance + sector pre-partition + Option B unknown bucket) + remove sector/ETF gate + harden metadata pre-fetch + `max_k` K-ballooning fix                                                                | 1-2 days    | ~30-60 min     | ~2 hrs           | Cluster noise rate drops measurably; trial time < 1hr; K stays bounded                      |
| **1** ✓         | Tuning harness skeleton (`parameter_space`, `objective`, `walk_forward`); one Optuna proof study on Tier 2, single 12mo window                                                                                          | 1 day       | 1 hr           | 1 hr             | Tuner produces param set that beats default on holdout, even marginally                     |
| **Pre-3** ✓     | SEC EDGAR metadata refresh — `scripts/refresh_ticker_metadata.py`; clears dotted-symbol artifacts, uses broader `company_tickers.json`, retries all NULL-sector rows; **result: 22% → 74.5% coverage (+4,621 tickers)** | ~20 min     | —              | trivial          | Coverage crosses 50% ✓; re-running Phase 1 study NOT required (Phase 1 gate already passed) |
| **3**           | 3-regime battery (calm_bull_2017, vol_shock_2020, sideways_2022) vs. baseline; `trend_bull_2023` / `mixed_2024` deferred to Phase 4 mandatory folds — actual wall-clock: ~36 hrs (9–17 h/run cold)                      | half day    | ~36 hrs actual | 36 hrs           | best-trial beats baseline in ≥ 2/3 completed regimes                                        |
| **3.5**  | Strategy deep dive: 6 analysis notebooks (regime P&L, signal quality, pair selection, portfolio beta, timing, tuning engine health); findings writeup with 1–3 hypotheses and recommended changes; go/no-go for Phase 4 — see `STRATEGY_DEEPDIVE_PLAN.md` | half day | — | trivial | `STRATEGY_DEEPDIVE_FINDINGS.md` written; go/no-go decision recorded in Decision Point 4.5 |
| **Pre-4** | Pre-warm `stock_prices` for all Phase 4 fold windows (2022-01 → 2026-03); cap `max_daily_candidates` upper bound to 300 in `parameter_space.py`; add 20-min per-trial wall-clock timeout | ~2 hrs | — | trivial | All fold windows warm in DB; no trial exceeds 20 min |
| **4 (coarse)**  | Regime detector (FRED + VIX + dispersion) + Tier 3 lookup table; 12 folds × 50 trials = 600 backtests; fold windows restricted to 2022+ warm data; `trend_bull_2023` / `mixed_2024` as mandatory holdout folds (deferred from Phase 3); also run Phase 3 best-trial (static) through same 12 folds as comparator (~12 extra backtests, ~1 hr) | half day | ~25 hrs | 25 hrs | Positive Sharpe across all three test regimes; regime-conditioned params beat Phase 3 best-trial (static) at p<0.10 — validates regime conditioning adds value |
| **4.5 (dense)** | Densify to 36 folds × 100 trials = 3,600 backtests — plan as Friday-night launch. **Run with short leg if implemented; SPY-beating gate reinstated if so.** | minimal | ~75 hrs | 75 hrs | If short leg in place: SPY-beating in ≥ 60% of holdout windows at p<0.05. If not: same Sharpe gate at p<0.05 vs Phase 3 best-trial (static) |
| **5**           | Scheduled tuning job → `active_parameters`; strategy reads from there; ±20% week-over-week cap                                                                                                                          | 1 day       | —              | 6 hrs/month      | Live shadow run beats baseline for 4 consecutive weeks                                      |


**Total minimum wall-clock**: ~10 days continuous, including 3 explicit
decision points before any large compute spend.

### Decision points (no/no-go gates)

1. **After Phase 2** ✓ — Trial time dropped to ~7 min (warm cache). K stays bounded. Noise rate reduced. Passed.
2. **After Phase 1** ✓ — Best trial (score -3.056) beat baseline (-4.712) by +1.656 delta. Harness validated. Passed.
  - Note: best trial returned -3.89% vs SPY +9.84% on Jan–Mar 2024 (strong bull market — structurally hostile to market-neutral pairs). Not a concern; real verdict is Phase 3.
3. **Pre-Phase-3** ✓ — SEC EDGAR metadata refresh completed. Coverage: 22% → **74.5%** (+4,621 tickers). All 8,204 universe tickers now have metadata rows. 259 dotted-symbol artifacts cleaned. Re-running Phase 1 study skipped (Jan–Mar 2024 bull market is not the right benchmark; compute better spent on Phase 3 multi-regime battery).
4. **After Phase 3** — Does best-trial beat baseline in ≥ 2/3 regimes? Observations so far: vol_shock_2020 best-trial +0.3% vs SPY −10.8% (strong signal); calm_bull_2017 and sideways_2022 both negative (expected for market-neutral strategy in trending/bear markets). Gate verdict pending baseline comparison. `trend_bull_2023` / `mixed_2024` added as mandatory folds in Phase 4.
4.5. **After Phase 3.5 deep dive** ✓ — GO, with revised goal. Phase 3 gate FAIL (1/3). Four hypotheses: (H1) structural long-only SPY beta ~0.57 — beating SPY is not achievable through tuning alone without a short leg; (H2/H3) exit/entry parameters tuned on a bull market — Phase 4 sideways folds fix this without pre-work; (H4) 75.6% displacement exits — working as intended, no intervention. **Goal revised:** Phase 4 target is positive Sharpe across regimes, not SPY-beating. SPY gate reinstated once short leg is implemented (targeted before Phase 4.5). Short leg is Priority 1 and runs as a parallel workstream to Phase 4. `max_portfolio_beta` cap added as interim H1 mitigation during Phase 4 coarse. See `STRATEGY_DEEPDIVE_FINDINGS.md`.
5. **After Phase 4 coarse** — Does regime-conditioned system beat Phase 3 best-trial (static, same 12 folds) at p<0.10? If no: regime conditioning is not adding value over tuning alone — stop, rethink detector features or objective. Only densify (Phase 4.5) if yes.

---

## Benchmark timing (measured 2026-04-17, updated 2026-04-21)

### Short window (Jan 2 → Mar 25 2024, ~4,900 symbols, 58 trading days)

| Cache state      | Wall-clock       | Notes                                             |
| ---------------- | ---------------- | ------------------------------------------------- |
| Cold (first run) | ~4.7 hours       | All prices fetched from Alpaca; one-time cost     |
| **Warm**         | **~3–7 minutes** | Prices already in DB; all tuning trials land here |

### Full-year window (Phase 3 actuals, ~250 trading days, best-trial params)

| Regime / params       | Wall-clock  | Notes                                                           |
| --------------------- | ----------- | --------------------------------------------------------------- |
| calm_bull_2017 / best | ~17 hrs     | Cold fetch + max_daily_candidates=487                           |
| vol_shock_2020 / best | ~3 hrs      | 5-month window, cold (crashed at 62/104 days)                   |
| sideways_2022 / best  | ~9 hrs      | Partially warm (stock_prices starts 2022-01-24)                 |
| calm_bull_2017 / base | ~5 hrs est. | Cold, but default max_daily_candidates=200 is ~2× faster        |

**Key Phase 3 findings for Phase 4 planning:**

- `stock_prices` only covers **2022-01-24 → present**. Any fold outside this range triggers a cold Alpaca fetch that turns a 5-min warm trial into a multi-hour stall.
- `max_daily_candidates=487` (Phase 1 best-trial) is ~2.5× slower per day than the default 200. Must cap the Optuna search space upper bound to prevent runaway trials.
- **Phase 4 timing estimate still holds** if and only if: (a) all fold windows are pre-warmed in `stock_prices` before Optuna starts, (b) `max_daily_candidates` is capped at 300, and (c) a 20-min per-trial timeout prunes anything pathological.

**Revised implications:**

- Pre-warm all fold windows in `stock_prices` (one-time ~2h cost) before Phase 4 Optuna launch
- Cap `max_daily_candidates` upper bound to 300 in `parameter_space.py`
- 600 coarse trials × ~5 min ÷ 4 parallel workers ≈ **~12–15 hours** — feasible overnight
- Phase 4.5 dense (3,600 trials ÷ 4 workers) ≈ **~75 hours** — plan as a Friday-night launch

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

`quality_scale` bounds K between `max_k × quality_scale_min` and `max_k ×
quality_scale_max`. `quality_scale_max` must be set to **≤ 1.0** (default: 1.0)
to enforce the `max_k` hard ceiling — values above 1.0 allow K to exceed
`max_k` and reproduce the ballooning behaviour. The buy loop's existing
`if available_cash < per_stock_budget: continue` already enforces affordability
— no need to encode it in K. Genuine rotation is now possible: positions ranked
below `k_target` in composite score get displaced and sold each day.

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

