# Strategy Deep Dive — Findings (Phase 3.5)

> Auto-generated 2026-04-22 from Phase 3 battery results. Updated 2026-04-23 with H4 displacement analysis.
> Source runs: best_trial (3f7def, feac3e, 815f18) · baseline (5015a8, d83ad4, 8757f1)
> Windows normalised to matching dates (vol_shock: Feb–Apr 30 2020; sideways: Jan–Aug 4 2022).

---

## 1. Summary table

| label | regime | ret% | SPY% | vs SPY | Sharpe | max DD% | avg cash% | avg pairs |
|---|---|---|---|---|---|---|---|---|
| best_trial | calm_bull_2017 | -37.8% | +19.0% | ✗ | -3.70 | 38.7% | 39% | 1011 |
| best_trial | vol_shock_2020 | -4.0% | -6.7% | ✗ | -2.07 | 8.2% | 39% | 228 |
| best_trial | sideways_2022  | -16.4% | -11.2% | ✗ | -4.75 | 16.9% | 38% | 394 |
| baseline | calm_bull_2017 | -18.6% | +19.0% | ✗ | -2.10 | 22.1% | 35% | 436 |
| baseline | vol_shock_2020 | -2.3% | -6.7% | ✗ | -1.18 | 7.7% | 36% | 118 |
| baseline | sideways_2022  | -16.2% | -11.2% | ✗ | -4.90 | 16.3% | 36% | 186 |

### Phase 3 gate result: **FAIL** — best_trial outscores baseline in 1/3 regimes (need ≥ 2)

| regime | best_trial score | baseline score | winner |
|---|---|---|---|
| calm_bull_2017 | -5.9569 | -4.2806 | baseline |
| vol_shock_2020 | -2.1786 | -1.2917 | baseline |
| sideways_2022  | -6.8990 | -7.0500 | best_trial (marginal) |

---

## 2. Trade activity

| label | trades | win rate | avg P&L/trip | median hold |
|---|---|---|---|---|
| best_trial | 928 | 40.9% | -14.18 | 2.0d |
| baseline   | 1158 | 43.2% | -19.03 | 2.0d |

> **Note (added 2026-04-23):** Deterministic exit-reason analysis on run 815f18 (sideways_2022)
> shows the 2-day median hold is driven primarily by **displacement** (75.6% of exits), not
> z-score threshold crossings (22.9%). See Hypothesis 4.

---

## 3. Market neutrality (beta vs SPY)

| label | regime | beta | R² | return corr |
|---|---|---|---|---|
| best_trial | calm_bull_2017 | 0.717 | 0.123 | 0.351 |
| best_trial | vol_shock_2020 | 0.496 | 0.609 | 0.781 |
| best_trial | sideways_2022  | 0.508 | 0.430 | 0.656 |
| baseline   | calm_bull_2017 | 0.819 | 0.281 | 0.530 |
| baseline   | vol_shock_2020 | 0.435 | 0.505 | 0.711 |
| baseline   | sideways_2022  | 0.483 | 0.424 | 0.651 |

---

## 4. Candidate funnel (discovery vs entry gate)

| label | regime | avg found | avg buy-ready | conversion |
|---|---|---|---|---|
| best_trial | calm_bull_2017 | 487 | 8 | 1.6% |
| best_trial | vol_shock_2020 | 487 | 14 | 3.0% |
| best_trial | sideways_2022  | 487 | 9 | 1.8% |
| baseline   | calm_bull_2017 | 200 | 4 | 2.2% |
| baseline   | vol_shock_2020 | 195 | 5 | 2.7% |
| baseline   | sideways_2022  | 200 | 5 | 2.5% |

---

## 5. Hypotheses

### Hypothesis 1: The strategy is structurally long-only with significant SPY beta — it is not market neutral

**Evidence:**
- Average beta vs SPY: best_trial = 0.717 / 0.496 / 0.508 (calm/vol/sideways)
- Average R² across regimes: ~0.39 — roughly 40% of daily return variance is explained by SPY.
- A beta-neutral strategy would show beta ≈ 0 and R² < 0.05. These values (~0.5) indicate the
  strategy moves roughly half as much as SPY on any given day.
- Both param sets are negative across ALL 3 regimes — a market-neutral strategy should be
  flat-to-positive in at least one of these environments.

**Mechanism:**
LumiBob enters pairs by buying the lagging symbol (the underperformer). It never shorts the
leading symbol. In a down market, both symbols decline — the lag symbol (the one bought) often
declines MORE, leading to losses even when the z-score spread eventually converges. The net
portfolio is a long-only book dressed as pairs trading.

**Proposed change:**
Add a short leg: when entering a pair (lead → lag), simultaneously short the lead symbol in
equal notional. This creates genuine dollar-neutral exposure, eliminating directional SPY risk.
This is a fundamental strategy design change, not a parameter change.

**Intermediate mitigation:**
Add a `max_portfolio_beta` parameter (e.g. 0.2). If rolling 20-day beta exceeds the cap,
stop adding new longs until positions rotate out. This reduces (but does not eliminate) the
directional exposure without requiring short-selling infrastructure.

**How to validate:**
Run calm_bull_2017 with short legs enabled. Beta should drop from ~0.72 to < 0.1.
The strategy should no longer lose 37% in a bull market.

---

### Hypothesis 2: Trade expectancy is negative — losers are larger than winners

**Evidence:**
- best_trial: win rate 40.9%, avg P&L = -14.18 per round-trip
- baseline: win rate 43.2%, avg P&L = -19.03 per round-trip
- Median holding period: 2.0 days (best_trial), 2.0 days (baseline)
- Pair correlation vs realized P&L: r = +0.024, p = 0.421
  (not statistically significant — long-horizon correlation does not predict pair P&L)

**Mechanism:**
A 2-day median holding period is extremely short for a pairs trading strategy. H4 analysis
(see below) establishes that displacement, not z-score exits, is the dominant driver (75.6%
of exits in sideways_2022). Two z-score-specific factors worsen the 22.9% of exits that
*are* threshold-driven:

1. **Exit threshold too tight (0.5):** The z-score exits at 0.5 σ. In a down market, the
   spread can reach 0.5 through directional decline of both stocks rather than genuine
   convergence, triggering exits at a loss. Big losses come from adverse directional moves
   accumulating before the 0.5 exit fires.

2. **Z-score window too short (20 days):** A 20-day rolling window produces noisy z-scores.
   A spread that looks like z=2.3 might be a legitimate signal, or a short-term noise spike
   with no mean-reversion tendency. The strategy cannot distinguish these.

**Proposed change:**
The parameter space already covers the desired ranges: `exit_threshold` low=0.1/high=1.5,
`zscore_window` low=10/high=40. The best_trial values (exit_threshold=0.5, zscore_window=20)
are not at the bounds — they are what Optuna found optimal when trained on Jan–Mar 2024, which
is a bull market. The tuner exploited the fact that in a trending market, tight exits and short
windows happen to be locally optimal. Changing defaults is irrelevant once the tuning engine is
deployed. The real fix is retraining on a sideways or mixed window so Optuna finds the higher
exit_threshold and longer zscore_window that generalise better.

In Phase 4, entry and exit thresholds become regime-conditioned Tier 3 parameters, which is the
correct long-term fix.

**How to validate:**
Re-run the Phase 1 Optuna study on a sideways_2022 window. Expect best_trial to converge on
exit_threshold > 1.0 and zscore_window > 25. The displacement rate should fall as z-score exits
fire closer to the z~1.23 level where displacement currently acts.

---

### Hypothesis 3: The entry threshold (z > 2.0) is too strict for calm-market regimes, thinning the selection pool

**Evidence:**
- Only 1.6–3.0% of daily scored candidates meet the entry threshold (z > 2.0).
- Average cash ratio across all runs: 35–39%, which is close to target (40% cash at
  `target_deployed_pct=0.60`). Capital utilization is not the problem — the portfolio sizing
  logic deploys capital to whatever candidates are available. The low conversion rate does not
  leave cash sitting idle.
- The 8–14 buy-ready candidates per day is the output of the portfolio construction algorithm,
  not a count of how many buys are actually executed. How many pairs get entered is determined
  by capital sizing, not by the number of candidates that cleared the threshold.

**Mechanism:**
In calm and trending markets (calm_bull_2017, sideways_2022), z-scores stay suppressed: spreads
rarely diverge to z = 2.0 because correlated stocks tend to move together without sharp
dislocations. A fixed entry threshold of 2.0 across all regimes means the entry gate is
calibrated for volatile conditions and becomes too restrictive in calm ones.

The consequence is not cash drag — it is **thin candidate pool quality**. When only 8–14 pairs
pass the gate, the composite score ranker has little material to differentiate. It selects the
best of a narrow sample, which may still be mediocre pairs. In volatile regimes (vol_shock_2020),
the conversion rate rises to 3.0% and the ranker has better signal-to-noise because the pairs
that do reach z = 2.0 are more likely to represent genuine dislocations.

The `max_daily_candidates=487` scan size is relevant here: a larger scan surface produces a
higher-quality *scored* pool (better correlations, more representative z-scores) but does not
directly change how many pairs pass the entry gate. Scan size and entry threshold are
independent levers — the former controls selection quality within the pool, the latter controls
the gate strictness.

**Proposed change:**
Make `entry_threshold` regime-conditional in Phase 4 Tier 3: lower to ~1.5 in calm/low-vol
environments, keep at 2.0–2.5 in high-vol regimes. The parameter space already supports this —
`entry_threshold` has low=1.0 and high=3.5, so no bound changes are needed. The Phase 4
regime-conditional tuner just needs to be pointed at different training windows per regime.

**How to validate:**
Rerun calm_bull_2017 with entry_threshold=1.5. The buy-ready pool should widen, giving the
composite ranker more candidates to differentiate. Monitor whether the quality of selected pairs
improves (higher composite scores, better win rates) rather than just counting entries.
This fix should be combined with H2 fixes — a wider pool only helps if the trade expectancy
of individual entries is also positive.

### Hypothesis 4: Portfolio displacement is the dominant exit mechanism and produces excessive churn

> Analysis source: run 815f18 (best_trial, sideways_2022, Jan–Aug 2022).
> Deterministic z-score replay using Alpaca price data. All 275 sells resolved.

**Evidence:**

| exit_reason | count | % of sells |
|---|---|---|
| displaced | 208 | 75.6% |
| zscore_exit | 63 | 22.9% |
| data_missing | 4 | 1.5% |

- Z-score at displacement time: median = **1.23 σ** (mean = 1.25, p75 = 1.53). Pairs are
  being kicked out of the portfolio with significant spread still remaining — well above the
  0.5 exit threshold — because a higher-scoring candidate has taken their slot in the top-K.
- Counterfactual test: for each displaced exit, the price data was replayed forward to find
  when the z-score would have crossed 0.5. Result: holding to the z-score exit would have
  produced avg P&L of **-126.01** vs the actual displaced avg P&L of **-15.50** — meaning
  displacement **accidentally functions as a stop-loss**, cutting losses 60 days earlier than
  the z-score exit would have.
- Holding to the z-score exit was better in only **18.3%** of displaced cases.
- Total counterfactual P&L cost of *not* displacing: **-22,987** across the run.

**Mechanism:**
The top-K portfolio construction re-ranks all active pairs and incoming candidates every day.
A freshly discovered pair with a high composite score displaces an existing position even if
that position is still in the middle of a valid spread divergence (z~1.23). In a down market
(sideways_2022), waiting for the spread to revert to z=0.5 means holding a long-only position
while both stocks continue declining — so the "reversion" arrives via both stocks falling
together rather than the laggard recovering. Displacement exits first and avoids this.

This is a symptom of H1 (long-only bias): in a genuinely market-neutral strategy, waiting for
z-score reversion is safe because the short leg hedges directional decline. Without the short
leg, the z-score exit is a poor exit signal in down markets, and displacement inadvertently
compensates.

**Conclusion:**
The displacement mechanism is working as intended and should not be suppressed. The counterfactual
confirms that displacement is beneficial — holding longer is worse 81.7% of the time. The root
cause of losses is H1 (long-only bias): every position loses money in a down market regardless
of how long it is held. Displacement limits that damage by cutting exposure sooner than the
z-score exit would.

A `min_hold_days` guard was considered but rejected: forcing positions to stay on for N days
would override a mechanism that is already producing better outcomes than the alternative. In
a genuinely market-neutral strategy (H1 fixed), holding longer to capture spread convergence
is safe because the short leg hedges directional decline — at that point, displacement churn
would naturally reduce because positions would no longer be bleeding directional losses.

No targeted intervention for H4 is needed before Phase 4. The `exit_reason` tracking added
in PR #37 provides observability to monitor whether displacement rates shift as Phase 4 tunes
exit and entry thresholds. If displacement remains dominant after Phase 4 regime conditioning,
revisit displacement hysteresis (require a minimum score advantage to trigger displacement)
as a lightweight alternative to a hard hold floor.

### Hypothesis 5: Pair selection finds correlated stocks, not cointegrated ones — the mean-reversion signal is structurally weak

> Analysis source: run 815f18 (best_trial, sideways_2022). Dollar-neutral simulation and
> ADF stationarity test on 50 traded pairs using Alpaca price data.

**Evidence:**

| Metric | Value | Interpretation |
|---|---|---|
| Pairs with stationary spread (ADF p < 0.05) | 6 / 50 (12%) | Only 1 in 8 pairs has a genuine mean-reverting spread |
| Spread narrowed during hold | 61.8% | Weak directional signal — barely above coin-flip |
| Avg \|z\| change during hold | −0.15 | Convergence exists but is small in magnitude |
| Dollar-neutral avg P&L | −8.37 | Still negative after perfectly hedging market direction |
| Win rate at z_entry 2.0–2.5 vs 1.5–2.0 | 39% vs 39% | Entry z-score has no discriminatory power on outcomes |

- The short leg (H1 fix) improves avg P&L by +4.85 per trade (~35%), but the dollar-neutral
  strategy still loses money. H1 alone is insufficient.
- The z-score entry signal is not discriminatory: higher entry z-scores do not predict higher
  win rates or better P&L. This is expected when spreads are non-stationary — a large z-score
  reflects a random drift, not a predictable reversion.

**Mechanism:**
LumiBob selects pairs using rolling correlation (corr_long, corr_short). Correlation measures
whether two stocks move together over a window — it does not measure whether their spread has a
stable long-run equilibrium. Cointegration (Engle-Granger or Johansen) tests for that property
directly. Two stocks can be highly correlated without being cointegrated: they may trend
together for months and then permanently diverge.

Without stationarity, the z-score is computed against a rolling mean that itself is drifting.
A z-score of 2.3 may indicate a genuine stretched spread or simply that the spread has moved to
a new regime. The strategy cannot distinguish these cases, which is why win rates are flat
across entry z-score levels and why higher conviction entries are not better.

**Proposed change:**
Add a cointegration gate to pair discovery. After computing correlation, run the Engle-Granger
two-step test on the spread series. Only admit pairs where the spread is stationary at p < 0.05
(or a tunable `coint_pvalue_threshold`). This replaces the implicit assumption that correlated
= cointegrated with an explicit test.

Implementation note: the ADF test adds ~1ms per pair at 252 days of data. With `max_daily_candidates=300`
and the existing per-trial timeout, this is manageable. The test can be cached per pair per
recompute window alongside `corr_long`.

**How to validate:**
Re-run pair discovery with the cointegration gate active. The fraction of stationary spreads
among entered pairs should rise from 12% to > 60%. The dollar-neutral win rate should improve
above 50% and avg P&L should turn positive. This validation should be run before Phase 4
to confirm the gate produces better inputs to the tuner.

---

## 6. Recommended changes (ranked by expected impact)

| Priority | Change | Type | Addresses | Effort |
|---|---|---|---|---|
| 1 | Add cointegration gate to pair discovery (`coint_pvalue_threshold`, Engle-Granger ADF on spread) | Strategy design | H5 | Medium |
| 2 | Add short leg for lag symbol (true dollar-neutral pairs) | Strategy design | H1 | High |
| 3 | Add `max_portfolio_beta` cap parameter (interim H1 mitigation while short leg is built) | New parameter | H1 | Medium |
| 4 | Phase 4 regime-conditioned study — run after Priority 1 is implemented so the tuner has quality pair inputs | Tuning (Phase 4) | H2, H3 | — |

---

## 7. Go / no-go for Phase 4

**Decision: CONDITIONAL GO — cointegration gate (Priority 1) must be implemented and
validated before the 600-trial Phase 4 study launches. Running Phase 4 on the current
pair selection would optimise parameters for a strategy where 88% of pairs lack a
mean-reversion basis. The tuner cannot fix signal quality.**

**Reasoning:**

Two structural problems now sit above the parameter-tuning layer:

1. **Weak signal (H5):** Only 12% of traded pairs have stationary spreads. The dollar-neutral
   simulation confirms that even with perfect market hedging (H1 fixed), the strategy still
   loses money (-8.37 avg P&L per trade). The z-score entry signal has no discriminatory power
   across entry levels. This is the deepest problem and must be fixed first.

2. **Long-only bias (H1):** The short leg improves P&L by ~35% per trade but is not sufficient
   on its own. It becomes high-value once pair quality is fixed — a market-neutral strategy with
   genuinely cointegrated pairs is the correct design target.

Phase 4 is still the right architecture for regime conditioning and the Tier 3 lookup table.
But running 600 trials on pairs that mostly lack mean-reversion properties will produce 600
noisy results that confound pair quality with parameter choices. The cointegration gate is a
prerequisite, not a parallel workstream.

**Recommended path:**
1. Implement the cointegration gate (Priority 1). Run a quick validation backtest on
   sideways_2022. Confirm: stationarity rate among entered pairs > 60%, dollar-neutral win
   rate > 50%, avg P&L per trip > 0.
2. If validation passes, launch Phase 4 coarse. The tuner now has quality inputs to work with.
3. Build the short leg (Priority 2) in parallel with Phase 4. Target completion before
   Phase 4.5 (dense) so the dense study runs with the correct strategy design.
4. `max_portfolio_beta` cap (Priority 3) as an interim guard during Phase 4 coarse.
5. SPY-beating gate is reinstated for the paper trade phase once both H1 and H5 are fixed.

---

## 8. Execution checklist

- [x] Phase 3 baseline backtests complete (all 3 regimes with matched windows)
- [x] Run IDs identified and tagged (best_trial: 3f7def, feac3e, 815f18 · baseline: 5015a8, d83ad4, 8757f1)
- [ ] Notebooks 1–5 executed (run `jupyter lab notebooks/` to review charts)
- [x] `STRATEGY_DEEPDIVE_FINDINGS.md` written
- [x] H4 displacement analysis complete (run 815f18, deterministic z-score replay via Alpaca)
- [x] `exit_reason` column added to `trades` table (PR #37, migration 001)
- [x] H5 signal quality analysis complete (dollar-neutral simulation + ADF stationarity on 50 pairs)
- [ ] Cointegration gate implemented and validation backtest run (Priority 1 pre-Phase-4 gate)
- [ ] Go/no-go decision recorded in `TUNING_ENGINE_PLAN.md` Decision Point 4.5
