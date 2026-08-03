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

### V2 — Floor and k selection *(walk-forward)*
Free parameters: `min_expected_gross_bps` ∈ {0, 15, 25, 40, 60}; `max_k` ∈
{10, 20, 30, 40}. Two parameters over ~1,000 qualified round-trips —
complexity green. Selection on `walk_forward_splits` with embargo ≥ 40 trading
days (the outcome horizon). **Reserve 2023-10-16, score once.**
**Guards:** null baseline (shuffle `expected_gross` across candidates within a
date — the rule must beat the 90th percentile), holdout gap, drop-one-fold
fragility.
**Pass:** held-out median improves with CI excluding zero; the chosen config
survives dropping any one fold.

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
