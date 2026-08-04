# Entry Criteria Overhaul: Dislocation + Magnitude Selection

**Date:** 2026-08-01
**Status:** PROPOSED
**Supersedes:** the scoring workstreams of `docs/plans/2026-07-31_composite-score-overhaul.md` (WS1 dead-weight removal stands; WS4 Chronos-2 is deferred)
**Evidence:** `docs/deepdives/2026-08-01_score-selection-and-the-missing-entry-gate.md`, `docs/deepdives/2026-08-01_disasters-surviving-the-event-gate.md`, PR #50

---

## 1. Why

Four findings, each established on the replay pool with date-clustered
intervals (the pair-level intervals used earlier overstate precision because
replay observations share legs):

1. **Entry is not signal-gated.** `before_market_opens` admits any candidate
   that ranks into `k_target` with `action='buy'`; nothing consults the
   z-score. Only 6% of live entries met the strategy's own stated entry
   condition (`z ≤ −entry_threshold`); 54% had no buy-side dislocation at all.
2. **The composite does not select positively.** Random top-20 beats it on the
   point estimate; score-outcome Spearman is negative in all three folds
   (pooled −0.297). Its largest variance term is `z_depth`, a binary direction
   flag carrying the smallest weight.
3. **Trade magnitude predicts outcome; the z-score alone does not.**
   `expected_gross = (|z| − exit_threshold) × spread_std` ranks outcomes at
   ρ +0.398 (11/12 dates positive) against +0.048 for |z| alone. Predicted
   magnitude is ~60–70% realised in the median.
4. **Concentration dominates the ranker.** Applying the live per-symbol dedup
   changes design outcomes by ~+69 bps; the choice of ranker moves ~+15.

## 2. Terminology

| Term | Meaning |
|---|---|
| **bps** | Basis points; 100 bps = 1% of the position's gross notional. |
| **Dislocation** | The spread sits far from its recent mean, measured in standard deviations (the z-score). Direction matters: the strategy buys the lag leg only when the lag is *cheap* (`z` very negative). |
| **Spread std** | Standard deviation of the pair's log-spread over the z-window — the size of a typical dislocation *in money*. Already computed inside `compute_zscore` and currently discarded. |
| **expected_gross** | `(|z| − exit_threshold) × spread_std`, in bps. "Reverting from here to the exit is worth this much." |
| **Emergency floor** | A minimum `expected_gross` below which a candidate is never bought, however it ranks. A cost-viability backstop, not a selector. |
| **k_target** | How many pairs the target portfolio holds. |

---

## 3. WS-A — Replace the composite with a dislocation + magnitude score

**Remove** `_composite_score` and its weights (`w_corr_long`, `w_corr_short`,
`w_z_depth`) from the entry path.

**Add** the entry criteria, evaluated in this order inside the discovery loop:

1. **Direction + dislocation gate** — require `z ≤ −entry_threshold`. Today's
   code never checks this. Note this is the same test `z_depth` encodes, which
   is why `z_depth` becomes constant (= 1.0) among admitted candidates and is
   removed rather than reweighted.
2. **Emergency floor** — require `expected_gross ≥ min_expected_gross_bps`.
   Preregistered starting value **25 bps** (≈ 1× the ~20 bps round-trip
   friction estimate). Rationale: across 12 dates a 20-bps floor never binds
   while a 60-bps floor prevents filling top-k on 4 of 12 dates — a third of
   days is not "rare", and the floor's job is viability, not selection.
3. **Rank survivors by `expected_gross`**, take top-k.

**Ordering matters for cost.** Both gates come free from one `compute_zscore`
call; cointegration/half-life (ADF) is the expensive step. Gating first skips
ADF for the ~60% of candidates that were never tradeable — `scoring_replay`
measured a ~10× speedup from exactly this reordering. It also changes what
`max_daily_candidates` buys: it becomes "qualified candidates per day", so a
separate, larger cap on pairs *examined* is needed, and both bounds must be
revisited in `parameter_space.py`.

**Sizing must be decoupled.** `base_budget` currently scales with
`composite_score` ([BobsBrain.py:729](BobsBrain.py:729)). Under the new design
that would size by `expected_gross`, and disaster rate rises monotonically with
magnitude (5.1% → 23.2% across quintiles). Use **flat sizing** within the
selected set, retaining the deployment-gap logic. Principle: we have no
validated ranking of *quality* among qualified candidates, so we should not bet
more on any of them — and it keeps the backtest interpretable by not
confounding the gates with a sizing rule.

**Retain for observability, not selection:** `corr_long`, `corr_short`,
`coint_pvalue`, `halflife_days`, and the new `expected_gross` and `spread_std`
persisted per pair.

### Consistency work (pr-reviewer Step 11)
- Remove the three weights from `parameter_space.py`, `_WEIGHT_NAMES`,
  `normalize_weights`, `create_run()` settings, `main.py`, README.
- Register `min_expected_gross_bps` (Tier 3 — cost/regime dependent) and the
  new examine-cap.
- `study1_pass_a.PASS_A_PARAMS` loses the weights; version-bump any re-run.

## 4. WS-B — Remove dynamic K

**Delete** `quality_scale` (`quality_scale_pivot`, `quality_scale_min`,
`quality_scale_max`) and set `k_target = max_k`.

**Evidence.** `quality_scale = clip(pool_corr / 0.7, 0.5, 1.0)` with `pool_corr`
in 0.8–1.0 clips to 1.0 on **every one of the 12 scoring dates** — the
mechanism has never varied. Replacing it with a cluster-quality indicator
would introduce variation where none exists, and every dynamic scheme tested
underperformed fixed k:

| scheme | mean | disaster % | positions |
|---|---|---|---|
| fixed k = 20 | **+2.4** | 18.8% | 234 |
| scale by pool_corr (current shape) | +2.4 | 18.8% | 234 |
| scale by pool expected_gross | −9.4 | 25.3% | 170 |
| inverse scale by expected_gross | −21.4 | 19.6% | 209 |
| scale by supply over floor | −27.1 | 29.8% | 131 |
| fixed k = 10 | −62.4 | 26.7% | 120 |

Two reasons no opportunity-linked scaling should be adopted:

- **Fewer positions is worse** (k=10 → −62.4 vs k=20 → +2.4). Diversification
  is doing real work; a mechanism that can only *reduce* k is
  harmful-or-neutral by construction.
- **Opportunity peaks on the worst days.** The three worst dates by realised
  mean have median pool `expected_gross` of 134 bps vs 66 elsewhere. Scaling up
  on opportunity levers into the regime-break dates that carry 73% of
  catastrophic loss.

`max_k` stays a tunable and its bounds should be re-examined upward, since the
evidence favours more positions.

## 5. WS-C — Concentration control *(recommended addition, outside the original brief)*

Not requested; included because it measured as the largest single lever
(~+69 bps vs ~+15 for the ranker). Cut it if scope matters more.

- **Verify** the existing per-symbol dedup actually binds in backtests (it
  exists in code but has never been measured).
- **Add** a per-cluster cap so one cluster cannot supply the whole book —
  the PTNQ case was one instrument decoupling from 30 partners at once.
- **Instrument-structure screen** (deepdive H1) — depends on the metadata
  repair landing first.

---

## 6. Validation tests

Per `.claude/skills/optuna-study.md`. **Unit: the pair round-trip.** All
intervals **date-clustered** — replay observations share legs and are not
independent. Preregistered before running; nothing below is fitted on the
final-test window.

### V1 — Replay-level design comparison *(no fitting; confirms the build matches the analysis)*
Re-score the replay pool through the *implemented* code path and reproduce the
design table. Old composite vs new criteria, dedup applied.
**Pass:** the implementation reproduces the analysis within noise — new-criteria
mean ≥ old, median materially higher. A mismatch means the implementation
diverges from what was analysed; stop and reconcile.

### V2 v1 — Floor and k selection *(walk-forward)* — **FAILED 2026-08-01**
Free parameters: `min_expected_gross_bps` ∈ {0, 15, 25, 40, 60}; `max_k` ∈
{10, 20, 30, 40}. Selection on `walk_forward_splits` with embargo ≥ 40 trading
days. Reserved 2023-10-16, scored once.
**Result: FAIL.** Complexity green (506 units/param) and holdout gap passed
(train +96.2, holdout +60.1). But the **null baseline failed hard** — real
−79.1 vs random null −21.0 (1st percentile), shuffled null +11.4 (0th) — and
**fold stability failed**: the floor held at 60 across all drops while `k`
swung 10 / 20 / 40, i.e. `k` is not identified.

Diagnosis (recorded rather than patched): the objective was the **median**
book outcome while `null_baseline` scores the **mean** of the top-k. Those
metrics diverge by construction here, and the divergence is mechanical, not
statistical — see the calibration table below. Two further defects: the
selected floor (60) sat at the top edge of the grid, so it was bounded rather
than identified; and the final-test date was uninformative because all 20 top
picks that day already cleared 60 bps, making baseline and chosen identical.

### V2 v2 — The band hypothesis *(version bump; new study name)*

**Why a new version rather than a re-cut.** V2 v1's grid searched a floor with
no ceiling, implicitly assuming "more expected gross is better." The
calibration data contradicts that assumption:

| expected-gross quintile | predicted | real median | **real mean** | disaster % | worst |
|---|---|---|---|---|---|
| 0 | 30.0 | 9.6 | +2.6 | 5.1 | −380 |
| 1 | 77.2 | 40.3 | +3.4 | 16.1 | −922 |
| 2 | 127.8 | 85.5 | **+58.8** | 11.6 | −511 |
| 3 | 200.8 | 130.8 | **+59.6** | 15.2 | −1,661 |
| 4 | 446.7 | **207.6** | **−42.9** | **27.6** | −4,861 |

The top quintile has the best median and a *negative* mean. Trimming its three
worst trades lifts it to +21.5 — still below q2 (+65.6) and q3 (+77.0) — so
this is not only a few blow-ups. Win rate is flat (65–76%) across all
quintiles, so none of it is a hit-rate effect.

**Hypothesis (mechanism, stated before testing).** A very large expected gross
means the spread has moved far in absolute terms relative to its own recent
history. Beyond some level that stops indicating "a big opportunity" and starts
indicating **the relationship itself is breaking** — a structural repricing
rather than a dislocation. If true, capping expected gross should remove
non-converging pairs and therefore improve the **mean**, not merely the median.

**Design changes from v1:**

1. **Primary objective is the MEAN**, not the median. A book of equal-weighted
   positions earns the mean; the median is reported as a secondary diagnostic.
   This also makes the objective and `null_baseline` coherent — the exact
   incoherence that sank v1. The band hypothesis predicts a mean improvement,
   so the mean is also the sharper test of it.
2. **Free parameters are floor and ceiling**; `k` is FIXED at 20. v1 showed `k`
   is not identifiable from this data, so leaving it free only adds variance.
   Preregistered grids: floor ∈ {0, 25, 50}, ceiling ∈ {150, 250, 400, ∞}.
3. **Mechanism checks, not just performance.** If the hypothesis is right, high
   expected-gross pairs must look like *broken relationships*: higher
   cap-exit (non-convergence) rate, longer holds, deeper drawdowns. These are
   checked directly. A performance improvement without the mechanism signature
   is treated as curve-fitting and does not pass.
4. **Held-out estimate is leave-one-fold-out** (3 genuinely different regimes),
   reported with its weakness stated — 3 units is thin.

**Confirmatory status — read this before acting on the result.** The band
hypothesis was *generated* from the same 12 scoring dates it would be tested
on, and the v1 final-test date is spent. **No clean confirmatory test exists
inside this dataset.** V2 v2 is therefore explicitly **hypothesis-generating**:
its job is to decide whether the band is worth a real test, not to validate it.
A genuine test requires either fresh folds (a cache rebuild on new date ranges)
or the V4 comparative backtest over different periods.

**Pass (to justify a confirmatory test, not to ship):** a band beats the
no-ceiling configuration on the **mean** with a date-clustered CI excluding
zero; the null baseline clears the 90th percentile; the selected ceiling
survives dropping any one fold; **and** the mechanism checks show the predicted
non-convergence signature at high expected gross.
**Fail:** report and stop. Do not widen the grid.

**Executed 2026-08-01 — FAIL. The hypothesis is refuted at the mechanism
level, which matters more than the performance result.**

*Mechanism — ABSENT.* High-expected-gross pairs do **not** stop converging:

| quintile | expected | cap-exit % | hold days | abs disaster % | **rel disaster %** |
|---|---|---|---|---|---|
| 0 | 30 | 5.1 | 11 | 5.1 | 22.6 |
| 1 | 77 | 4.2 | 12 | 16.1 | 25.8 |
| 2 | 128 | 1.9 | 10 | 11.6 | 13.4 |
| 3 | 201 | 3.7 | 11 | 15.2 | 15.2 |
| 4 | 447 | 4.2 | 12 | 27.6 | 23.0 |

Cap-exit rate (non-convergence) is **flat at ~4%** across the whole range, and
holding periods are flat at 10–12 days. Pairs at 447 bps expected revert as
reliably as pairs at 30 bps. There is no relationship-breakdown signature.

*By-product worth recording:* the rising **absolute** disaster rate
(5.1% → 27.6%) is largely a **threshold artifact**. "Disaster" is defined as
gross < −100 bps, an absolute line; a pair whose spread is 3× larger crosses it
3× more easily with identical relative behaviour. Normalising the outcome by
each pair's own expected gross, the **relative** disaster rate is flat
(22.6 / 25.8 / 13.4 / 15.2 / 23.0). Several earlier statements in this program
— "magnitude selection concentrates fat tails" — are partly restatements of
this artifact and should be read with it in mind.

*Performance — fails on 3 of 4 remaining checks.* In-sample the band looks
strong (floor 50 / ceiling 250: mean +58.5 vs +9.8 baseline, disaster 10.6% vs
17.2%). But `holdout_gap` **FAILS** decisively — train +58.9, holdout **+1.5**,
gap +57.4 — the improvement does not survive out of fold. The ceiling is
**not stable**: dropping bull_2023 flips it to ∞. The date-clustered CI on the
mean is **+48.7, CI −21.8 .. +115.6, includes zero**. Only the null baseline
passes (99th pct), and that is the least meaningful check here because the band
was derived from the same data.

*Why it fails:* per-fold top-quintile means are +327.5 (mixed_2023_q4), −65.3
(bull_2023), −97.0 (sideways_2022). A ceiling helps in two regimes and would
**destroy value in the third**. The apparent effect is a handful of large
losses concentrated in two folds — bull's three worst top-quintile trades alone
sum to −13,721 bps.

**Conclusion: no ceiling. `expected_gross` stays a floor-and-rank quantity.**
Replay-level parameter selection has now failed twice on this dataset (V2 v1,
V2 v2) for the same underlying reason: 12 correlated scoring dates, outcomes
dominated by a few large trades, and any apparent optimum traceable to a
specific fold. **Stop tuning on the replay pool.** The remaining honest test is
V4, the comparative backtest over different periods with costs on.

### V3 — Supply and deployment *(the failure mode that matters operationally)*
Log per day: candidates examined, candidates qualified, `candidates_buy_ready`,
positions held, `cash_ratio`.
**Pass:** the book fills `max_k` on ≥ 90% of days and average `cash_ratio` does
not rise materially versus baseline. A gate that starves the book shows up as
idle capital, not as a bad Sharpe.

### V4 — Comparative backtest *(the real test)*
Backtest-agent workflow, two arms (current vs new), **≥ 2 regimes**, costs on.
Watch: mean and median P&L per round trip, win rate, trade count, disaster
count, max drawdown, SPY comparison, `cash_ratio`, and realised per-symbol /
per-cluster concentration.
**Pass (preregistered):** median P&L per round trip improves; mean does not
degrade beyond its bootstrap interval; disaster count does not rise faster than
trade count. **Explicitly not expected:** a step change in mean return — every
validated finding in this program is a median/tail effect, and V4 should be
read as confirming the gates remove unviable trades, not as discovering edge.

### V5 — Live-entry audit *(closes the population gap)*
After V4, re-measure the dislocation distribution of realised entries.
**Pass:** ~100% of entries fully dislocated, versus 6% today. This is what makes
every replay-based study (PR #50, E1, E2) applicable to live behaviour for the
first time.

---

## 7. Sequencing

```
metadata repair (in flight)          -> unblocks WS-C instrument screen
WS-A + WS-B implemented together     -> one behavioural change, one backtest
  V1 replay parity check             -> GATE: implementation matches analysis
  V2 floor/k walk-forward            -> GATE: held-out CI excludes zero
  V3 supply check                    -> GATE: book fills, cash not idle
  V4 comparative backtest            -> GATE: median improves, mean holds
  V5 live-entry audit                -> confirms the population gap closed
WS-C concentration control           -> separate change, separate backtest
```

## 8. Risks and honest expectations

- **No validated positive-selection mechanism remains in the pipeline.** The
  gates are validated as viability filters; `expected_gross` ranking ties
  correlation ranking within noise. This design removes trades that cannot pay
  for themselves — it does not claim to find winners.
- **Disaster rate will rise.** Magnitude-linked selection raises it from ~10%
  to ~19% in replay. That is the accepted cost of trading bigger moves, and it
  is why WS-C matters.
- **The floor's level rests on an unmeasured cost figure.** The ~20 bps
  round-trip estimate comes from the Pass A v3 notebook, not a cost model
  (H-D remains unbuilt). The floor's *shape* is validated; its *level* is not.
- **All replay evidence carries survivorship bias** (current-membership
  universe, inherited from PR #50) and non-independent observations.

## 9. Execution checklist

- [ ] Metadata repair verified (ETF/fund coverage, sector taxonomy normalised)
- [ ] WS-A: dislocation gate + floor + `expected_gross` ranking, gates before ADF
- [ ] WS-A: sizing decoupled from score; flat within selected set
- [ ] WS-A: `parameter_space.py` / `create_run()` / `main.py` / README swept
- [ ] WS-B: `quality_scale` removed; `k_target = max_k`; `max_k` bounds revisited
- [ ] V1 replay parity; V2 walk-forward + guard panel; V3 supply; V4 comparative
      backtest; V5 live-entry audit
- [ ] Ledger rows in `docs/plans/2026-07-12_cloud-tuning-studies.md`
- [ ] WS-C decision (in or out of scope)
