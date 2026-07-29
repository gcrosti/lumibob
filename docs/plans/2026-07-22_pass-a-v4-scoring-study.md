# Implementation Plan — Pass A v4 Scoring-Quality Study

> Created 2026-07-22. Source analysis:
> `docs/deepdives/2026-07-17_pass-a-score-signal-and-exploitability.md`
> (Update 2026-07-22). Supersedes the ad-hoc "raise `w_corr_short`" idea **and** the
> earlier backtest-objective framing of this plan — the study runs a scoring script,
> not a backtest.

> **⛔ OUTCOME — NO-GO (reweighting the composite is not a lever).** The study was built
> and run (WS1–WS3 below). Under **out-of-fold validation at the live-faithful lookback
> (152)** it does **not** rank dislocated pairs by forward P&L: pair-level
> `Spearman(component, forward_gross)` ≈ 0 (well-powered, n 572–946/fold), top-K-by-score
> ≈ random-K, held-out (leave-one-fold-out) objective **negative in all three folds**. The
> initially-reported +52 / `w_corr_short`→0.2 was on a **non-live-faithful 400-day cache
> without out-of-fold validation** and is **retracted as overfit** — it is the post-mortem
> that produced `tuning/overfit_guards.py` and the `optuna-study` skill. This re-confirms
> the deep dive: the composite is a good **filter**, a poor fine-grained **ranker**.
> **WS1 (log half-life) stands as a valid standalone fix.** Success criteria and workstream
> results below are kept for the record but must be read through this verdict.

## Problem

The composite score's weights are **noise-selected**. The old Pass A objective was
`0.7 × Spearman(score, round_trip_P&L) + 0.3 × clip(sharpe/3)`, evaluated over the
~30–60 *entered* pairs per run. Three defects made it noise: **range restriction**
(you only observe outcomes for pairs the score already selected → attenuated rho),
**P&L noise** (round-trip P&L conflates ranking with beta, exit, sizing, and zero
costs), and **small n** (se(rho) ≈ 0.13–0.19). Across `study1_pass_a_v3`'s 97 trials
the objective is statistically indifferent to 4 of 5 weights (`w_corr_short` p=0.12)
**and to `corr_short_window`** (p=1.00). The frozen best-trial weights put 40% on a
dead half-life component and 0.2% on `corr_short` — the one input shown (Phase 2) to
separate the catastrophic tail out-of-fold. The winning vector is a lucky draw.

## Core idea — evaluate the score, not a portfolio

The score's job is **admission ranking**. Evaluating that does not require a trading
simulator, costs, or portfolio accounting. Replace the backtest objective with a
**scoring-quality study**: score the full candidate pool, and optimize the weights so
the score ranks pairs by their **future spread behavior**. This fixes all three
defects at once — full pool (no range restriction), P&L-free spread outcome (no
beta/sizing noise), thousands of pairs (tight CI), seconds per trial. Economics
(cost-clearance) is confirmed **once at the end**, not tuned in the loop.

Two things make or break it (see WS3): the outcome must be **tail-sensitive and
edge-aligned** (a plain rank correlation is blind to the catastrophic ~8% and would
recreate the original failure), and gross ranking must be **confirmed against costs**
exactly once, outside the tuning loop.

## Goal / success criteria

- **G1 (power)** — the scoring metric resolves weight vectors: the best clearly
  separates from the field; metric stable across folds (vs the old ±0.13–0.19 noise).
- **G2 (tail + edge)** — the retrained weights raise the **top-K mean gross forward
  spread P&L, positive in every fold**, materially above the current noise-weight
  baseline (Phase 2 baseline pooled mean +2.0).
- **G3 (weight by signal)** — `w_corr_short` is set by the metric (expected materially
  above 0.002), not by hand.
- **G-window** — the tuned corr windows generalize (stable across folds, not overfit);
  `corr_short_window` landing near the Phase 2 25-day horizon corroborates that finding
  (a large divergence is a flag to investigate, not auto-accept).
- **G-cost (final gate)** — one post-cost comparative backtest shows net-positive
  economics in the folds, or quantifies the residual gap (→ H-C / sizing).

## Workstreams (ordered by dependency)

### WS1 — Fix the half-life component *(prerequisite; small)* — **DONE**

A near-constant component (`score_halflife` = 0.962 ± 0.029) holds dead weight *and*
flattens the composite's spread, so no weight is interpretable until it is fixed.

**Design decision — log transform, not percentile.** The plan originally called for a
cross-sectional percentile transform, but the live scoring is **per-pair** with a
**variable-size daily candidate pool** (gated by `max_daily_candidates`); a
percentile-within-today's-pool would be unstable and non-comparable across days and
would force a two-pass restructure of both scoring loops. A per-pair **log-spaced**
map achieves the same goal (restore variance) while staying a drop-in and stable:
`score = clip((ln(ceiling) − ln(hl)) / ln(ceiling), 0, 1)`. It *repurposes*
`max_halflife_days` as the log-space ceiling (no dead param). Result: the 1–3 day
cluster now spreads 1.00→0.80 (was 0.98→0.96).

- **Implemented:** shared `halflife_to_score()` in `StockEvaluator.py`, called from
  `compute_spread_scores` and both `BobsBrain` scoring paths (re-score + discovery).
- **Tuning-engine:** `max_halflife_days` retained (now the log ceiling); `w_halflife`
  unchanged. No new/removed params.
- **Tests:** `TestHalflifeToScore` (range, endpoints, monotonicity, variance-restore);
  full suite green (128 passed).
- **2.09× calibration constant** — deferred to reporting/half-life-level use only; not
  needed for the ranker.

### WS2 — Scoring-quality replay harness + cache *(the core new build; medium)* — **DONE**

**Implemented** as `tuning/studies/scoring_replay.py`. Reconstructs the real candidate pool via
the strategy's own `TickerClusterer` at sampled dates (4/fold), scores with the real
`StockEvaluator`, and computes the P&L-free forward gross outcome under the frozen exit.
Look-ahead controls: position-based alignment to the last close ≤ T, hedge frozen
pre-T, outcome from (T, T+40td] only. Dislocation-first gate (|z|≥2) keeps only
tradeable pairs and skips ~90% of the ADF cost. Cache: `tuning/studies/_scoring_cache/*.parquet`
(gitignored; regenerable). **Built: 3,104 tradeable observations** (sideways 1,164 /
bull 1,293 / mixed 647). `score_halflife` std 0.18 on this pool (WS1 variance confirmed
at scale). Trailing log-returns stored per obs for WS3 corr re-windowing.


An **offline script** — no lumibot, no portfolio, no orders — that emits, per fold,
the pooled pairs with their **window-independent** artifacts (forward outcome, coint,
half-life) plus everything needed to compute the corr components at a trial's chosen
windows:

1. **Candidate pool.** Reconstruct the admitted pool once per fold with **fixed,
   evidence-grounded clustering params** (clustering *gates*; the composite *ranks*).
   Cheaper fallback if pool reconstruction is heavy: a broad hard-gated pair sample
   (penny/coverage gates only). Decision default: fixed-clustering, computed once.
2. **Component matrix.** `corr_long, corr_short, z_depth, coint_pvalue, halflife` for
   **every** pooled pair (not just selected), at the scoring date — from
   `StockDataCache` / `stock_prices`.
3. **Forward outcome.** Per-pair **gross forward spread P&L** under the **frozen
   outcome contract** (below) — this is "future behavior" in units of harvestable
   edge, P&L-free of costs/sizing.

**Outcome contract (frozen — never in the search space):** `zscore_window`, the exit
rule (`sim_exit(exit_z=0.5)` convention), and the hedge/coint lookbacks. These *define
the target*; tuning any of them is target leakage (§Risks). Freezing them keeps the
forward outcome cacheable regardless of which corr windows a trial picks.

Cache the **window-independent** artifacts — forward gross outcome, `coint_pvalue`,
`halflife`, and the raw price windows — **once per fold**; the expensive pass (forward
path building, ADF/coint) never repeats. The corr components (`corr_long`,
`corr_short`) depend on the tuned corr windows (WS3), so compute them either **on the
fly per trial** (cheap Pearson over the cached price windows) or from a **precomputed
component tensor over a discretized window grid**. Because the corr windows feed only
the score — confirmed absent from `TickerClusterer.py`, so they touch neither the pool
nor the outcome — this stays light.

- **Reuses:** existing scoring/clustering code paths in "score-all, trade-none" mode.
- **No live-strategy persistence change** — scores are recomputed per trial, never
  stored (a stored score is frozen to one weight vector, useless for the study).
- **Validation:** the cached matrix reproduces the Phase 2 numbers (catastrophic
  counts, quintile table) when scored with the gate-run weights **and windows**.

### WS3 — The scoring-quality study *(depends on WS1 + WS2)* — **DONE — NO-GO**

**Implemented** as `tuning/studies/scoring_study.py` (Optuna, top-K mean forward gross with the
0.5·mean + 0.5·min per-fold floor; corr windows tuned; corr recomputed per trial from
cached returns).

**Verdict: NO-GO — reweighting the composite does not generalize.** At the live-faithful
lookback (152) with **out-of-fold validation** the composite does **not** rank dislocated
pairs by forward P&L: pair-level `Spearman(component, forward_gross)` ≈ 0 (well-powered, n
**572–946/fold**), top-K-by-score is **indistinguishable from random-K**, and the
leave-one-fold-out held-out objective is **negative in all three folds**. Composite
reweighting is **not a lever**; this re-confirms the deep dive — the composite is a good
**filter**, a poor fine-grained **ranker**. WS4/WS5 downstream of the reweight are moot.
**WS1 (log half-life) stands as a valid standalone fix.**

**Retracted in-sample result (2026-07-23, 300 trials — do NOT use):** the first pass was run
on a **non-live-faithful 400-day cache without out-of-fold validation**, and reported an
in-sample lift that did not survive the guard panel:

| | objective | corr_long | **corr_short** | coint | halflife | per-fold (side/bull/mixed) |
|---|---|---|---|---|---|---|
| Baseline (gate weights) | +36.7 | 0.19 | **0.002** | 0.41 | 0.41 | +39 / +33 / +49 |
| Best trial *(retracted — overfit)* | **+52.2** | 0.32 | **0.187** | 0.38 | 0.11 | +55 / +73 / +46 |

The apparent G1/G2/G3 "wins" (objective resolves weights, `w_corr_short`→~0.2, top-K
positive per fold) were **artifacts of the non-live-faithful cache and the absence of
held-out validation**; under OOF at the live lookback they vanish (held-out negative in all
three folds). This is the exact failure — a great in-sample number that is pure
overfitting — that motivated the guard panel in `tuning/overfit_guards.py`. **Do not treat
the +52 or the `w_corr_short`→0.2 reweight as a result.**

**Why the original run misled:** metric was **gross** (costs deferred to WS4); the 400-day
cache was **not live-faithful** (live scoring uses lookback 152 → the +2 window dims fit
data the live path never sees); pool sampled at 4 dates/fold; `z_depth` excluded as inert
among tradeable pairs. The decisive defect was the **absence of out-of-fold validation** —
add the guard panel and the optimum is exposed as noise.


Optuna optimizes the composite **weights and the two predictor windows**
(`corr_long_window`, `corr_short_window`) over the cached matrix to maximize a
**tail-sensitive, edge-aligned** ranking metric.

- **Metric (primary):** mean **top-K gross forward P&L** (rank pool by score, take the
  top-K the strategy would select, average their forward gross), **averaged across
  folds with a per-fold floor** (every fold positive — the LORO discipline). This is
  directly tail-sensitive (a mean the −1554 can move) and operationally faithful
  (top-K selection). **Do NOT use Spearman** — rank correlation ignores outlier
  magnitude and would re-optimize the median while ignoring the tail.
- **Tuned as predictor dims:** `corr_long_window`, `corr_short_window`. They feed only
  the score, not the outcome or the pool, so tuning them is leakage-free and cheap (WS2
  recomputes the two correlations per trial / from a window tensor). This replaces the
  earlier fixed-value-plus-sensitivity-grid workaround — the old 25 / 84 values were
  noise-selected (p=1.00 for corr_short) and are no longer imposed; the study sets them.
  The +2 dims are controlled by the per-fold floor + held-out discipline. The Phase 2
  evidence (25-day horizon) is a **prior/sanity check** on where corr_short lands, not
  a constraint.
- **Frozen (outcome contract, never tuned):** K, exit convention, `zscore_window`,
  hedge/coint lookbacks — see WS2.
- **Files:** new `tuning/studies/study1_pass_a_v4.py` + a scoring objective in
  `tuning/objective.py`; confirm `_WEIGHT_NAMES` / `normalize_weights` cover the weight
  set and that `corr_long_window` / `corr_short_window` are in the v4 search space
  (already registered Tier 2 in `parameter_space.py:84,87`).

### WS4 — Single post-cost confirmation backtest *(depends on WS3; the only backtest)*

Take the retrained weights and run **one** comparative backtest — current vs new
weights — across the three folds, **costs on**. This is the economics gate the scoring
study deliberately does not tune.

- **Cost model (demoted from prerequisite to here):** implement a slippage/commission
  model in the fill path so `trades.slippage > 0` (`DatabaseClient.record_trade`
  already carries the column). Costs are **exogenous config, not tunable**. Run
  10/20/30 bps sensitivity.
- **Files:** backtest fill path via `BobsBrain.backtest(...)` (`main.py:90`).
- **Validation (G-cost):** net-positive per fold, or a quantified residual gap that
  routes to H-C (entry magnitude) / sizing — *not* back into the scoring loop.

### WS5 — Soft corr_short floor *(conditional)* — **MOOT under the NO-GO** (was "NOT NEEDED, 2026-07-23")

> Superseded by the WS3 NO-GO: since composite reweighting does not generalize, there is no
> `w_corr_short` lever for a floor to refine. The analysis below ran on the **same retracted
> non-live-faithful cache**, so its `+54.7 / +56.4` figures are not live-faithful either.
> Its conclusion (*do not build the floor*) still holds — now for the stronger reason that
> the whole reweighting question is closed.

Tested directly on the WS2 cache (no new data). A knee transform on `score_corr_short`
strictly contains the linear case (knee=0 ≡ linear), yet with equal budget it scored
**+54.7 vs the linear +56.4 (−1.7 bps)** and chose a large knee (0.627) that fits noise.
Diagnostic under the best linear weights: only **11%** of top-K picks are in the bottom
corr_short quintile (the linear `w_corr_short`≈0.21 already excludes them), and those
are *not* the catastrophic drivers (+33 bps / 2 catastrophic vs the rest +64 / 27). The
Phase 2 threshold-shape was a property of the *removal screen on entered pairs*; in the
top-K *selection* framework the linear weight subsumes it. **Do not build the floor.**

Caveat: verdict is on gross outcomes; if WS4's cost model materially reshapes the tail,
re-check — but there is no current evidence for a floor.

## Dependency graph

```
WS1 (half-life) ─┐
                 ├─► WS3 (scoring study + window grid) ─► WS4 (one post-cost backtest) ─► WS5 (cond.)
WS2 (harness+cache)┘
```

WS1 and WS2 are independent. WS3 is cheap once WS2's cache exists. WS4 is the single
backtest and the only place the cost model is needed. WS5 is gated on WS3's result.

## What this design deliberately drops or defers

- **No trading simulator in the tuning loop** — the scoring study is a ranking script.
- **No cost model in the objective** — one confirmation backtest at the end (WS4).
- **No clustering replay per trial** — pool built once per fold (WS2); the tuned corr
  windows feed only the score, so the pool is invariant to them.
- **Portfolio-level effects (K interactions, capital allocation, held-pair
  correlations) are out of scope for the *weights* question** — per-pair ranking is
  the correct unit; portfolio construction is the separate sizing/diversification
  layer the deep dive named as the complement for the residual irreducible tail.

## Risks / open questions

- **Metric mis-specification is the #1 risk.** A tail-blind metric (Spearman, or a
  top-K *median*) silently recreates the original failure. The metric must be a
  tail-sensitive **mean** of an edge-aligned outcome. Lock this in review before build.
- **Gross ranking ≠ profitability (H-C).** The study optimizes gross; if WS4's
  post-cost gate fails, the lever moves to entry magnitude / sizing, not more weight
  tuning. WS4 is a real gate, not a rubber stamp.
- **Pool definition conditions results** — fixed-clustering (faithful) vs broad sample
  (simple). Default to fixed-clustering-once; note the choice in results.
- **Target leakage via the outcome contract.** Any param that defines the forward
  outcome (`zscore_window`, exit rule, hedge/coint lookbacks) must stay **out of the
  search space** — tuning it would improve the metric by making the outcome
  predictable, not by ranking better. `zscore_window` is the trap: it has a dual role
  (feeds `z_depth` *and* the exit), so it is frozen; `z_depth` is then evaluated at a
  fixed window but still carries a tunable weight.
- **Overfit from the +2 predictor window dims.** Controlled by the per-fold positivity
  floor (a window that only helps one regime can't win) and held-out validation.
- **Forward outcome depends on the frozen exit convention.** Held at current values;
  optional sensitivity to the exit rule if WS3 conclusions look fragile.

## Execution checklist

- [ ] WS1: percentile half-life transform in `StockEvaluator.py` + both `BobsBrain.py`
      paths; deprecate/repurpose `max_halflife_days`; wire `HL_CAL` where level is used
- [ ] WS1 validation: `score_halflife` variance restored on the gate-run pool
- [ ] WS2: build the score-all/trade-none replay harness; cache the window-independent
      artifacts (forward_gross, coint, half-life, price windows) per fold; corr
      components computed per trial / from a window tensor
- [ ] WS2 validation: cached matrix reproduces Phase 2 numbers under gate-run weights
- [ ] WS3: scoring objective = cross-fold mean of top-K gross forward P&L, per-fold
      floor; **not** Spearman; outcome contract (K, exit, `zscore_window`, hedge/coint
      lookbacks) frozen out of the search space
- [ ] WS3: tune `corr_long_window` + `corr_short_window` as predictor dims; verify
      they generalize across folds (G-window)
- [ ] WS3 validation (G1/G2/G3): metric resolves weights; top-K mean gross positive per
      fold; `w_corr_short` > 0.002 by the metric
- [ ] WS4: implement cost model (fill path; `trades.slippage > 0`; not tunable); one
      comparative backtest current vs new weights, costs on, 10/20/30 bps sensitivity
- [ ] WS4 validation (G-cost): net-positive per fold, or quantified residual → H-C/sizing
- [ ] WS5 (conditional): soft corr_short floor; register param + settings + normalize
- [ ] Update `docs/plans/2026-07-12_cloud-tuning-studies.md` ledger: Study 2/3 blocked
      until WS4 validates
