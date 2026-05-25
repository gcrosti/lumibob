# Strategy Deep Dive — Analysis Plan (Phase 3.5)

> Trigger: run after Phase 3 baseline backtests complete and before committing
> to Phase 4's 25-hour compute spend. Purpose: determine whether the alpha
> signal is real and correctly plumbed, or whether the strategy needs a design
> change before more tuning is worthwhile.

---

## Goal

Produce 1–3 concrete hypotheses explaining why the strategy underperforms SPY,
backed by data. Each hypothesis must be specific enough to suggest a testable
code or parameter change.

---

## Deliverables

| Deliverable | Path | Description |
|---|---|---|
| Notebook: regime performance | `notebooks/phase35_01_regime_performance.ipynb` | P&L, SPY comparison, cash utilization broken down by regime |
| Notebook: signal quality | `notebooks/phase35_02_signal_quality.ipynb` | Z-score predictiveness; does spread mean-revert after entry? |
| Notebook: pair selection | `notebooks/phase35_03_pair_selection.ipynb` | Composite score vs realized P&L; is the scorer selecting better pairs? |
| Notebook: portfolio characteristics | `notebooks/phase35_04_portfolio_characteristics.ipynb` | Net beta, daily return correlation with SPY, drawdown decomposition |
| Notebook: timing analysis | `notebooks/phase35_05_timing_analysis.ipynb` | Holding period distribution; z-score at entry vs exit vs outcome |
| Notebook: tuning engine health | `notebooks/phase35_06_tuning_engine_health.ipynb` | Parameter bounds, search efficiency, objective calibration, fold structure |
| Findings writeup | `STRATEGY_DEEPDIVE_FINDINGS.md` | Narrative synthesis: hypotheses, evidence, recommended changes |

All notebooks must be run against real DB data and committed with output cells
intact. The findings writeup is produced last, after all notebooks are complete.

---

## Runs to include

Use all completed Phase 3 runs: the three best-trial runs and the three
baseline runs across the same regimes. Query to identify them:

```sql
SELECT run_id, started_at, completed_at,
       settings->>'backtesting_start' AS bt_start,
       settings->>'backtesting_end'   AS bt_end
FROM backtest_runs
WHERE mode = 'backtest'
  AND completed_at IS NOT NULL
ORDER BY started_at DESC
LIMIT 20;
```

Tag each run as `best_trial` or `baseline` and assign its regime label
(`calm_bull_2017`, `vol_shock_2020`, `sideways_2022`) based on the date window.
Use these tags throughout all notebooks for consistent labeling.

---

## Notebook 1 — Regime performance (`phase35_01_regime_performance.ipynb`)

**Questions to answer:**
- What is the return, max drawdown, and Sharpe ratio for each run vs SPY
  over the same window?
- Is the strategy at least reducing drawdown in bear/vol regimes, even if
  total return is negative?
- What is average cash utilization per regime? Is the strategy sitting in
  cash rather than deploying?
- How does `candidates_found` and `candidates_buy_ready` vary by regime?
  If candidates are scarce, discovery is the bottleneck. If buy-ready is low
  relative to candidates, the entry signal is too restrictive.
- How does `active_pairs` evolve over time within each regime? Does it grow,
  plateau, or collapse?

**Key tables:** `portfolio_snapshots`, `backtest_runs`

**Expected outputs:**
- Side-by-side return table: each run vs SPY, normalised to starting budget
- Cash utilization chart per regime (daily `cash_ratio` over time)
- `candidates_found` vs `candidates_buy_ready` bar chart per regime
- Drawdown curve comparison (strategy vs SPY) for each regime

---

## Notebook 2 — Signal quality (`phase35_02_signal_quality.ipynb`)

**Questions to answer:**
- For completed round-trips (buy → sell), what is the distribution of
  realized P&L? What fraction are profitable?
- Is the z-score at entry correlated with realized P&L? (A real signal
  should show: higher entry z-score → more P&L recovered on exit.)
- Of pairs that were entered, what fraction actually mean-reverted
  (i.e., the spread compressed after entry)? What fraction widened further?
- What is the distribution of holding periods? Are we being stopped out
  early, or holding winners to full reversion?
- Does slippage materially eat into returns? What is avg slippage as a
  fraction of avg trade P&L?

**Key tables:** `trades`, `pairs`

**Method for round-trip reconstruction:**

```sql
SELECT
    b.run_id,
    b.symbol,
    b.pair_id,
    b.price      AS buy_price,
    b.quantity   AS buy_qty,
    b.filled_at  AS buy_time,
    s.price      AS sell_price,
    s.quantity   AS sell_qty,
    s.filled_at  AS sell_time,
    s.slippage   AS sell_slippage,
    (s.price - b.price) * b.quantity - s.slippage AS pnl,
    EXTRACT(EPOCH FROM (s.filled_at - b.filled_at)) / 86400 AS hold_days
FROM trades b
JOIN trades s
  ON s.run_id   = b.run_id
 AND s.symbol   = b.symbol
 AND s.pair_id  = b.pair_id
 AND s.side     = 'sell'
 AND s.filled_at > b.filled_at
WHERE b.side = 'buy'
ORDER BY b.filled_at;
```

**Expected outputs:**
- P&L distribution histogram (profitable vs unprofitable round-trips)
- Scatter: entry z-score vs realized P&L (one point per round-trip)
- Holding period distribution (box plot by regime)
- Slippage as % of gross P&L (aggregate)

---

## Notebook 3 — Pair selection (`phase35_03_pair_selection.ipynb`)

**Questions to answer:**
- Is the composite score at time of entry correlated with realized P&L?
  (If not, the scorer is not selecting better pairs.)
- Does `corr_short` (short-window correlation) predict P&L better than
  `corr_long` (long-window correlation)?
- Are winning pairs concentrated in specific sectors or cluster types?
- Are ETF pairs treated differently, and do they perform better/worse?
- What is the distribution of `pairs.correlation` (long-horizon) for
  winning vs losing pairs?
- How stable are pairs over time — what fraction of pairs that were active
  at day 1 of a regime are still active at day 30?

**Key tables:** `trades`, `pairs`, `ticker_metadata` (if available)

**Expected outputs:**
- Scatter: composite score at entry vs realized P&L
- Box plot: `pairs.correlation` for profitable vs unprofitable pairs
- Sector breakdown of winning vs losing pairs (if metadata available)
- Pair survival curve: fraction of initial pairs still active at each day

---

## Notebook 4 — Portfolio characteristics (`phase35_04_portfolio_characteristics.ipynb`)

**Questions to answer:**
- What is the daily return correlation between the strategy and SPY?
  A market-neutral strategy should be near 0; positive correlation means
  we're not hedged and SPY exposure is leaking in.
- What is the net beta of the portfolio? (Compute from daily returns.)
- In vol_shock_2020, does the strategy actually protect capital, or does
  it decline alongside SPY?
- What fraction of total return variance is explained by SPY (R-squared)?
- How does `avg_correlation` across active pairs evolve over time?
  Correlation collapse (all assets correlate to 1 in a crash) would
  destroy the pairs signal and should be visible here.

**Key tables:** `portfolio_snapshots`, `stock_prices` (for SPY daily returns)

**Method:**
Compute daily returns from `portfolio_value` changes. Fetch SPY daily returns
from `stock_prices`. Run OLS regression: `strat_return ~ spy_return`. Report
alpha (intercept), beta (slope), R-squared.

**Expected outputs:**
- Scatter: daily strategy return vs SPY return (one point per day, coloured
  by regime)
- Rolling 20-day beta over time per regime
- `avg_correlation` over time per regime
- OLS summary table: alpha, beta, R-squared per run

---

## Notebook 5 — Timing analysis (`phase35_05_timing_analysis.ipynb`)

**Questions to answer:**
- What is the z-score distribution at entry across all trades?
  If most entries are at z < 1.5, the entry threshold is not being respected
  or the z-score window is too short to produce clean signals.
- What z-score level do sells occur at? Are we exiting at reversion
  (z near 0) or being stopped out at wider spreads?
- For losing trades, was the spread still widening at exit, or had it
  already started reverting?
- Is there a holding period sweet spot — a hold duration range that
  correlates with profitability?
- Are there day-of-week or time-in-regime effects? (e.g., pairs entered
  early in a regime perform differently from those entered late.)

**Key tables:** `trades`, `stock_prices` (to reconstruct spread at exit),
`portfolio_snapshots` (for daily z-score context)

**Note:** Exact z-scores at entry/exit are not stored in `trades`. Reconstruct
them from `stock_prices` price ratios for the lead/lag symbol over the
`zscore_window` days preceding each fill date.

**Expected outputs:**
- Histogram: z-score at entry across all trades
- Histogram: z-score at sell across all trades
- Scatter: hold days vs P&L (look for sweet-spot cluster)
- Time-in-regime vs P&L: scatter coloured by regime

---

## Notebook 6 — Tuning engine health (`phase35_06_tuning_engine_health.ipynb`)

**Question to answer:** Is the tuning engine set up to succeed, independent
of whether the underlying alpha is real? Covers parameter bounds, search
efficiency, objective function calibration, and structural issues that would
cause Optuna to waste trials or converge to the wrong region.

### 6a. Parameter bound analysis (from Optuna trial history)

Pull all completed trials from the Phase 1 Optuna study:

```python
import optuna
study = optuna.load_study(study_name='tier2_proof_v2', storage=OPTUNA_STORAGE)
trials_df = study.trials_dataframe()
```

For each parameter in the best trial, compute its normalised position within
its bound: `(value - low) / (high - low)`. Flag any parameter where this
is > 0.90 or < 0.10 — the bound is likely too tight and Optuna is pressing
against a wall rather than finding the true optimum.

Also plot the distribution of sampled values for each parameter across all
trials. Bimodal or boundary-hugging distributions are a warning sign.

**Known concerns to verify:**

| Parameter | Concern |
|---|---|
| `hdbscan_selection_method` | Default is `eom` but plan recommends `leaf` as starting config; if all best trials use `leaf`, the default is wrong |
| `hdbscan_cluster_selection_epsilon` | Default 0.0, plan recommends 0.3; if best trials cluster near 0.3–0.5, the range should shift up |
| `zscore_window` | Upper bound 40 may be too low — academic literature often uses 60-day windows for pairs; check if best trials hit the upper bound |
| `max_k` | Upper bound 50; if best trials approach 50, the bound may be suppressing the true optimum |
| `max_position_pct` | Upper bound 0.35 is very concentrated (35% in one pair); check if best trials want even higher concentration or lower |
| `corr_long_window` | Upper bound 252 (one year); if best trials approach 252, the strategy may want a full-year lookback |
| `target_deployed_pct` | High deployment (>0.80) with noisy pairs may amplify losses; check what the best trials preferred |

**Missing parameters to flag:**
- No **stop-loss** / max spread-widening before forced exit (currently a pair
  is held until z-score exits or it's displaced by K-rotation)
- No **max holding period** (pairs can be held indefinitely if z stays elevated)
- `quality_scale_min` and `quality_scale_max` are listed as magic numbers to
  expose in the Phase 0 plan but are absent from `parameter_space.py` — verify
  whether they were actually added to `BobsBrain` and just not registered here

### 6b. Search efficiency

- **Trials per parameter**: with 50 trials over ~15 Tier 2 parameters, that
  is roughly 3 trials per dimension. TPE needs ~10–20 trials to model a single
  dimension reliably. Compute effective search budget vs dimensionality and
  flag if severely underpowered.
- **Weight sampling geometry**: `w_corr_long`, `w_corr_short`, `w_z_depth` are
  each sampled on [0,1] and then normalised. This means Optuna sees a
  3-dimensional cube but the actual search space is a 2D simplex. TPE will
  waste trials in the corners. Check what fraction of trials had a raw weight
  sum > 1.5 or < 0.5 before normalisation (indicates wasted sampling budget).
- **Pruning effectiveness**: from `tuning_studies`, compute
  `n_trials_pruned / (n_trials_completed + n_trials_pruned)`. A pruning rate
  below 30% suggests the pruner is not firing early enough. A rate above 80%
  suggests the intermediate reports are too noisy or arrive too late.

### 6c. Objective function calibration

The current objective is:
```
score = Sharpe  -  λ * max_drawdown  -  μ * trade_count_penalty
```

Check:
- What are the actual values of λ and μ? Pull from `tuning/objective.py`.
- In the Phase 1 study, what was the range of raw Sharpe, drawdown term, and
  trade penalty across all trials? If the drawdown term dominates (i.e., it
  accounts for >50% of score variance), we are effectively optimising for
  low drawdown, not return.
- Is the "hard constraint: OOS return > SPY in ≥60% of holdout windows"
  actually enforced in `objective.py`, or is it aspirational? If not enforced,
  the tuner can ignore it entirely.
- Does the Phase 1 best trial actually beat SPY on the holdout, or just beat
  the defaults? A tuner that maximises Sharpe but never beats SPY is
  mis-specified for the stated goal.

### 6d. Walk-forward window structure

- For each fold window, how many completed round-trips occurred in the Phase 1
  holdout period? If < 20 trades, there is insufficient sample size to evaluate
  Sharpe reliably — each fold is essentially noise.
- Is the 12-month train / 3-month holdout ratio appropriate for this strategy's
  signal horizon (`zscore_window` default 20 days, `corr_long_window` default
  90 days)? The holdout should be at least 5× the longest signal window;
  3 months = ~63 trading days, which is only marginally longer than
  `corr_long_window`.
- Does the fold structure align with known regime transitions? If a fold
  straddles a regime change (e.g., train in bull, holdout in crash), the
  trained params will be structurally wrong for the holdout.

### 6e. Tier classification audit

Review each Tier 3 parameter and ask: does this parameter actually respond to
weekly regime changes, or is it more stable? Misclassified parameters waste
the fast-tuning cadence and add noise.

| Parameter | Current tier | Audit question |
|---|---|---|
| `min_position_pct` / `max_position_pct` | 3 | Does position sizing actually need to change weekly, or is it a Tier 2 risk preference? |
| `quality_scale_pivot` | 3 | This controls K scaling; does it respond to VIX/regime, or is it stable? |
| `entry_threshold` | 3 | Does optimal entry z vary measurably across the 3 Phase 3 regimes? |
| `zscore_window` | 3 | A 20-day window is slow to change; weekly re-tuning may add noise, not signal |

**Expected outputs:**
- Table: normalised bound position for each param in best trial (flag >0.9 / <0.1)
- Plot: sampled value distributions for top-5 most important params
- Table: objective component breakdown (Sharpe, drawdown, trade penalty) across trials
- Scalar: trades per holdout fold (flag if < 20)
- Scalar: pruning rate
- List: structural issues found (missing stop-loss, weight geometry, etc.)

---

## Findings writeup (`STRATEGY_DEEPDIVE_FINDINGS.md`)

Produced after all 5 notebooks are complete. Structure:

### 1. Summary table
One-row-per-regime summary of key metrics from Notebook 1.

### 2. Hypotheses (1–3)
Each hypothesis must follow this format:

> **Hypothesis N: [Short title]**
> **Evidence:** [Specific charts/numbers from notebooks that support it]
> **Mechanism:** [Why this causes underperformance vs SPY]
> **Proposed change:** [Specific code, parameter, or design change to test]
> **How to validate:** [What a follow-up backtest should show if the fix works]

### 3. Analyses performed
Brief narrative of what each notebook found, with key charts referenced.

### 4. Tuning engine issues
Separate from the strategy alpha question: list any structural issues found
in Notebook 6 (bound violations, weight sampling geometry, underpowered
search, objective mis-calibration, fold size problems). Each issue should
have a concrete fix (e.g., "widen `zscore_window` upper bound to 60",
"switch weight sampling to Dirichlet", "add stop-loss parameter").

### 5. Recommended changes
Ranked list of changes to implement before Phase 4, ordered by expected
impact and implementation effort. Separate the list into two tracks:
(a) strategy design changes (addresses alpha hypotheses) and
(b) tuning engine fixes (addresses Notebook 6 findings).

### 6. Go / no-go for Phase 4
Explicit statement: does the evidence suggest Phase 4 (regime conditioning)
is the right next step, or should a strategy design change be made first?

---

## Execution checklist

- [ ] Phase 3 baseline backtests complete (all 3 regimes, `completed_at` not NULL)
- [ ] Run IDs identified and tagged (best_trial vs baseline, regime label)
- [ ] Notebook 1 run and output committed
- [ ] Notebook 2 run and output committed
- [ ] Notebook 3 run and output committed
- [ ] Notebook 4 run and output committed
- [ ] Notebook 5 run and output committed
- [ ] Notebook 6 run and output committed
- [ ] `STRATEGY_DEEPDIVE_FINDINGS.md` written and reviewed
- [ ] Go / no-go decision made and recorded in `TUNING_ENGINE_PLAN.md` Decision Point 4.5
