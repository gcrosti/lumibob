# Implementation Plan — Pass A v4 Scoring Study

> Created 2026-07-22. Source analysis:
> `docs/deepdives/2026-07-17_pass-a-score-signal-and-exploitability.md`
> (Update 2026-07-22). Supersedes the ad-hoc "raise `w_corr_short`" idea.

## Problem

The composite score's weights are **noise-selected**. The Pass A objective is
`0.7 × Spearman(score, P&L) + 0.3 × clip(sharpe/3)`; the deep dive proved the rho
term ≈ 0 (noise, se 0.13–0.19), and the Sharpe term is computed on a **zero-cost**
simulator. Across `study1_pass_a_v3`'s 97 trials the objective is statistically
indifferent to 4 of 5 weights (`w_corr_short` p=0.12, `w_coint` p=0.65,
`w_halflife` p=0.60). The frozen best-trial weights put **40% on a dead half-life
component** (constant 0.962) and **0.2% on `corr_short`** — the one input shown to
separate the catastrophic tail out-of-fold. The winning vector is a lucky draw; a
near-opposite vector scored the same.

We cannot fix this by hand-setting a weight — the objective that would justify any
weight is broken. Three prerequisites must land before a retrain means anything.

## Goal / success criteria

A retrained composite whose weights are set by a **signal-bearing, cost-aware
objective**, validated in a comparative backtest with costs on.

- **G1** — the new objective is deterministic and discriminating: weight↔objective
  Spearman significant (p<0.05) for the components that carry signal; CI width on
  the gate metric ≤ ±0.05 (vs the old ±0.13–0.19).
- **G2** — post-cost mean per round trip is **positive in every fold** at a
  defensible cost assumption (≥15 bps round-trip), vs the current negative-under-cost
  baseline.
- **G3** — the retrained `w_corr_short` is set by the objective (expected materially
  above 0.002), not by hand.

## Workstreams (ordered by dependency)

### WS1 — Fix the half-life component *(prerequisite; small)*

**Change.** Replace the linear ceiling map
`halflife_score = max(0, 1 − halflife_days / max_halflife_days)` with a
**cross-sectional percentile transform**: score each pair by the rank of its
half-life within the current candidate pool (lower half-life → higher score). This
guarantees real [0,1] variance regardless of the absolute clustering (~1–3 d today)
and is invariant to the systematic 3–5× optimism. Apply the measured **2.09×**
calibration constant (`HL_CAL`) only where the half-life *level* is consumed (honest
reporting; the rejected time-stop) — it does **not** fix variance and is irrelevant
to the ranker.

**Files.** `StockEvaluator.py:265` (fresh-compute path), `BobsBrain.py:377-378`
(stored-hl path) and `BobsBrain.py:488-490` (fresh path) — all three must use the
same transform. Persist the transformed value into `pairs.score_halflife` as today.

**Tuning-engine.** `max_halflife_days` (`parameter_space.py:109`, Tier 2) becomes a
no-op under a percentile transform — deprecate it or repurpose as a floor below which
half-life is treated as pathological. `w_halflife` stays registered; its range is
unchanged. No new score weight added.

**Validation.** On the three gate runs' pairs, recomputed `score_halflife` must show
variance (std ≫ 0.03, vs today's 0.029). Confirm the composite's cross-sectional
spread widens.

**Why, not just what.** A near-constant component with any weight is dead weight
*and* — because the composite is a weighted average — it flattens the spread of every
other component. No weight allocation is interpretable until this is fixed. (Rescue,
not delete: half-life *rank* carries weak reversion-timing signal, +0.24, that a real
objective may still choose to use.)

### WS2 — Cost model in the simulator *(prerequisite; small/infra)*

**Change.** Inject a slippage + commission model into the backtest fill path so every
fill records a non-zero, realistic cost, and P&L/Sharpe reflect it. Costs are
**exogenous environment config, not tunable** — do NOT add them to the Optuna search
space (you cannot optimize your way out of costs). Fix a defensible default
(bid-ask + impact; target ~15 bps round-trip) and expose a sensitivity dial for
10/20/30 bps runs.

**Files.** Backtest execution path invoked from `BobsBrain.backtest(...)`
(`main.py:90`); `DatabaseClient.record_trade` already carries a `slippage` column
(`DatabaseClient.py:863`) — populate it with the modeled value instead of 0.0.

**Validation (H-D).** Re-run one known run with costs on; assert recorded
`slippage > 0` on every fill and that Sharpe/total-return move down as expected.

**Why.** The entire measured edge (median ~13, mean ≤9 bps) lives *inside* the
10–30 bps cost band, so profitability is decided by this model — currently "zero."
Decisively, it is the **only** thing that makes the `corr_short` trade-off visible to
the tuner: under real costs, high-corr_short pairs' ~5.5 bps median wins go
net-negative while low-corr_short fat medians survive but carry the tail — so costs
determine which direction `corr_short` should be weighted. Without WS2, G3 is
undefined.

### WS3 — New objective *(depends on WS1 + WS2; medium/large — the design piece)*

**Change.** Replace `0.7 × Spearman(score, P&L) + 0.3 × clip(sharpe/3)` with:

- **(a) P&L-free full-pool gate** (H-A): a discovery-replay that scores the *entire*
  candidate pool each day and correlates `score` with a P&L-free forward outcome
  (e.g. realized forward `corr_short`, spread ADF, calibrated reversion) over all
  scored pairs — removing the range restriction that crushed the old gate's power
  (n ×10–20, minutes per trial, no trading sim).
- **(b) post-cost expectancy term** (needs WS2): mean net bps per round trip on the
  entered subset, so the objective rewards economics that survive frictions.

Combine (a) and (b) explicitly (weights TBD in design); (a) supplies discrimination
power, (b) anchors it to money.

**Files.** `tuning/objective.py` (`BacktestObjective` / `score_run`);
`tuning/studies/study1_pass_a.py` (new `study1_pass_a_v4`, new objective string in the
module docstring). Build the discovery-replay harness (does not exist yet).

**Validation (G1).** On a fixed weight sweep, the new objective's weight↔objective
Spearman is significant for signal-bearing components and its gate-metric CI ≤ ±0.05.

### WS4 — Retrain + comparative backtest *(depends on WS1–WS3)*

Run `study1_pass_a_v4` (register in `parameter_space` unchanged weight set; confirm
`_WEIGHT_NAMES` and `normalize_weights` cover the set). Extract best weights. Then a
**comparative backtest** — current mechanics vs new (fixed half-life + costs + new
weights) — across the three folds, costs on, per the backtest-agent workflow.

**Validation (G2/G3).** Post-cost mean positive in every fold; retrained
`w_corr_short` materially > 0.002 and set by the objective.

### WS5 — Soft corr_short floor *(conditional; only if WS4 shows a linear weight can't express the threshold)*

If the retrained linear weight still can't cut Q1 without over-rotating into
low-median Q5, add a **deployment-safe nonlinear transform**: steep score penalty
below a `corr_short` knee, flat above — pairs below the knee still deploy when they're
the best available (no hard gate, no stranded capital). New tunable
(`corr_short_knee` or similar) → **must** be registered in `parameter_space.py`
(correct tier), added to `create_run()` settings, and — if it enters the composite —
reflected in `normalize_weights`.

## Dependency graph

```
WS1 (half-life) ─┐
                 ├─► WS3 (objective) ─► WS4 (retrain + comparative bt) ─► WS5 (cond.)
WS2 (cost model)─┘
```

WS1 and WS2 are independent and can proceed in parallel. WS3 needs both. WS5 is
gated on WS4's result.

## Risks / open questions

- **Cost magnitude is a modeling choice.** 15 bps is defensible but not measured;
  run 10/20/30 bps sensitivity so conclusions aren't knife-edge on the assumption.
- **Discovery-replay harness is net-new** (WS3a) — the largest build item; scope it
  before committing to WS3's timeline.
- **Range restriction persists in retro.** The +2→+13 screen lift is removal-only; a
  reweight *substitutes* pairs we can't see retro. WS4's full-pool backtest is the
  only honest test of the reweight — treat the retro number as motivation, not proof.
- **Percentile half-life changes the composite distribution** and may shift other
  tuned params; the retrain absorbs this, but don't compare pre/post weights
  component-by-component as if the scale were unchanged.
- **Residual irreducible tail.** Even a perfect corr_short weight leaves NET/SNOW,
  JSMD, VONG-type blowups (normal corr_short). Position sizing / diversification is a
  separate, complementary workstream — not covered here.

## Execution checklist

- [ ] WS1: percentile half-life transform in `StockEvaluator.py` + both `BobsBrain.py`
      paths; deprecate/repurpose `max_halflife_days`; add `HL_CAL` where level is used
- [ ] WS1 validation: `score_halflife` variance restored on gate-run pairs
- [ ] WS2: slippage/commission model in fill path; populate `trades.slippage`; costs
      as fixed config (NOT in Optuna space)
- [ ] WS2 validation (H-D): known run with costs on → `slippage > 0`, Sharpe moves
- [ ] WS3a: build full-pool discovery-replay gate (P&L-free)
- [ ] WS3b: post-cost expectancy term; combine into new objective
- [ ] WS3 validation (G1): weight↔objective significance; gate-metric CI ≤ ±0.05
- [ ] WS4: run `study1_pass_a_v4`; extract best weights; comparative backtest 3 folds,
      costs on (backtest-agent workflow)
- [ ] WS4 validation (G2/G3): post-cost mean positive per fold; `w_corr_short` > 0.002
      by objective
- [ ] WS5 (conditional): soft corr_short floor; register param in `parameter_space.py`
      + `create_run()` settings + `normalize_weights`
- [ ] Commit notebook §Phase 2 (entry-discriminability analysis) alongside these docs
- [ ] Update `docs/plans/2026-07-12_cloud-tuning-studies.md` ledger: Study 2/3 remain
      blocked until WS4 validates
