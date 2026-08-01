# Score Selection and the Missing Entry Gate

**Date:** 2026-08-01
**Data:** Pass A v4 replay pool (2,043 tradeable observations, 12 dates, 3 folds)
and the 175 entered pairs from the three gate runs (`4f419e`, `4b26c6`,
`bcb308`).
**Scripts:** `tuning/studies/study3_score_selection.py`
**Triggered by:** the E2 v2 side-finding that the top-20 composite book had a
mean of −31.9 bps against +26.5 for the full pool.

## Summary

The investigation started as "why is the score anti-predictive?" and found
something more fundamental: **the strategy does not gate entry on a
mean-reversion signal at all.** New candidates are admitted purely by composite
rank. Only 6% of live entries were fully dislocated in the tradeable direction
and 54% had no buy-side dislocation whatsoever. The score cannot predict
per-trade outcomes in part because there is no dislocation setup to predict.

Two corrections to earlier claims are recorded in §1 — the "anti-predictive"
framing was overstated.

---

## 1. First, a correction: "anti-predictive" is not established

The E2 v2 note reported top-20 −31.9 vs +26.5 pool as evidence the score is
anti-predictive. Two things weaken that:

**(a) The deficit is mean-only, i.e. tail-driven.**

| | mean | median |
|---|---|---|
| top-20 by composite | −28.9 | **+23.8** |
| full pool | +28.8 | **+43.7** |

The medians are ordered the same way, but the −57.8 bps mean gap comes from
three trades (−4,861, −4,716, −4,332), two of which share a leg (LNZA) on a
single date.

**(b) Under date-clustered resampling the effect includes zero.**

| estimate | 95% CI |
|---|---|
| top-20 − pool deficit, date-clustered bootstrap | **−160.5 .. +9.3** |

Only 5 of 12 dates show top-20 beating the pool, but two dates (2023-02-15
at −398.8, 2022-02-15 at −191.5) carry the pooled result.

**What survives:** there is **no evidence the score selects positively**. A
random ranker's top-20 (+20.1) beats the composite's (−28.9) point-estimate,
and among the 175 live entered pairs the score-outcome Spearman is negative in
all three folds (−0.112 bull, −0.389 mixed, −0.345 sideways; pooled −0.297).
**What does not survive:** the stronger claim that the score is *actively*
anti-predictive at the portfolio level. The plan and ledger entries have been
amended.

---

## 2. Score decile behaviour reproduces PR #50 exactly

| decile | mean | median | win rate | **win size** | disaster % |
|---|---|---|---|---|---|
| 0 (lowest score) | +70.9 | +67.3 | 71.7% | **+217.8** | 16.1% |
| 1 | +80.9 | +85.1 | 74.5% | +177.3 | 15.2% |
| 4 | +40.1 | +19.8 | 73.2% | +77.2 | 4.9% |
| 8 | +13.8 | +28.2 | 67.7% | +75.9 | 14.2% |
| 9 (highest score) | −3.7 | +14.6 | 69.3% | **+61.1** | 7.8% |

Win *rate* is flat at 68–75% across every decile; win *size* falls monotonically
from +218 to +61. This is PR #50's "correlation is a variance/tail dial, not an
edge dial" reproduced on independent machinery: the score ranks pairs by
correlation, higher correlation means tighter spreads, and tighter spreads mean
smaller payoffs per convergence. **The score is working exactly as designed —
the design just doesn't select for profit.**

---

## 3. The structural finding: entry is rank-gated, not signal-gated

### 3a. There is no dislocation check on new entries

In `BobsBrain.before_market_opens` Phase 3 ([BobsBrain.py:564](BobsBrain.py:564)),
any candidate that ranks into `k_target` is written with `'action': 'buy'`.
No branch consults the candidate's z-score, direction, or `z_depth` before
admitting it.

`StockEvaluator.get_zscore_action` — which does encode the entry rule
(`z < -entry_threshold` → `'buy'`) — is called only in Phase 2a for pairs
**already held**, and only its `'sell'` return is consumed:

```python
if symbol in position_symbols and action == 'sell':
    pair['action'] = 'sell'; pair['exit_reason'] = 'zscore_exit'
else:
    pair['action'] = 'hold'
```

The `'buy'` return value is dead for entry purposes. The Z-score gates the
**exit** only.

### 3b. What the live strategy actually bought

Across 175 entered pairs in the three gate runs:

| dislocation at discovery | n | share | mean | median | disasters |
|---|---|---|---|---|---|
| `z_depth = 0` — no buy-side dislocation | 95 | **54.3%** | +3.5 | +13.9 | 8 |
| `0 < z_depth < 1` — partial | 69 | 39.4% | +3.5 | +7.8 | 5 |
| `z_depth = 1` — full (`z ≤ −entry_threshold`) | 11 | **6.3%** | −19.9 | +17.0 | 1 |

**Only 6% of entries met the strategy's own stated entry condition.** The
README describes trading "the lag leg using Z-score spread signals"; in
practice the book is the top-K most-correlated pairs, entered on rank.

Note the mean is flat across buckets (+3.5 / +3.5 / −19.9, the last on n=11) —
entering *with* a dislocation signal did not perform better than entering
without one. That is itself evidence the current signal carries little
information as implemented.

### 3c. `z_depth` is a direction flag, not a depth measure

```python
if z >= -exit_threshold:
    return 0.0, z
depth = min((-z - exit_threshold) / (entry_threshold - exit_threshold), 1.0)
```

The test is on signed `z`, not `abs(z)`. A spread at z = +5 (lag expensive) is
scored 0.0 — "not diverged" — despite being maximally diverged. This is
*consistent* with a long-only lag strategy, but it means the component measures
**which way** the spread moved, not **how far**.

In the replay pool (where every observation has |z| ≥ 2 by construction) the
consequence is stark — `z_depth` is binary:

| component | sd | frac at max | weight | **w × sd** |
|---|---|---|---|---|
| corr_long | 0.140 | 0.1% | 0.3 | 0.042 |
| corr_short | 0.082 | 0.1% | 0.5 | 0.041 |
| **z_depth** | **0.499** | **53.1%** | 0.2 | **0.100** |

Despite carrying the smallest weight, `z_depth` contributes **more score
variance than both correlations combined**. The composite's ranking is
dominated by a binary direction flag, with correlation acting as a tie-breaker
— the inverse of what the weights imply, and of what every weight-tuning study
to date has assumed it was optimising.

---

## 4. Population mismatch: the replay pool studies a different strategy

`scoring_replay._score_pair` discards any candidate with |z_entry| < 2.0. So:

| | dislocated at entry |
|---|---|
| replay pool (PR #50 Pass A v4, E1, E2) | **100%** by construction |
| live entered pairs | **6%** fully, 46% partially, 54% not at all |

Every cache-based study in this program — PR #50's Pass A v4 conclusion, E1's
event discrimination, E2's exclusion test, and the planned Chronos-2 C3-A — has
been evaluating a **dislocation-triggered strategy that the live code does not
implement.** Their internal validity stands; their transfer to live behaviour
does not, and should be stated wherever those results are cited.

---

## 5. Hypotheses

**H1: Entry is not signal-gated, so the score has no setup to predict.**
**Evidence:** §3a code path; 54% of entries with `z_depth = 0`; only 6% fully
dislocated; flat mean across dislocation buckets.
**Mechanism:** the composite ranks pair *quality* (correlation), not trade
*opportunity* (dislocation). Ranking by quality with no opportunity gate buys
well-behaved pairs at arbitrary points in their spread cycle, where expected
convergence profit is ~0 by construction.
**Proposed change:** require `z ≤ −entry_threshold` for a new candidate to be
admitted to the target portfolio; rank the qualifying set by composite.
**How to validate:** comparative backtest, gate off/on — win rate should hold
while mean P&L per trip rises; candidate supply must be checked (the replay
pool shows ~170 dislocated pairs per scoring date, so supply exists).

**H2: `z_depth` is miscast — it belongs in the gate, not the score.**
**Evidence:** §3c — binary in the dislocated population, dominates score
variance at the smallest weight, and encodes direction rather than magnitude.
**Mechanism:** a [0,1] component that is really an indicator adds a large
constant offset to half the population, swamping the continuous components.
**Proposed change:** move the dislocation test into the entry gate (H1) and
remove `w_z_depth` from the composite, leaving a two-component correlation
score; or, if kept, fix the asymmetry to use `abs(z)`.
**How to validate:** score-variance decomposition after the change; rank
correlation of the reduced score vs the current one; the same
rank-equivalence check used for the WS1 removal.

**H3: Replay-based conclusions need re-scoping, not retraction.**
**Evidence:** §4.
**Mechanism:** the harness gate (|z| ≥ 2) selects a population the live
strategy rarely enters.
**Proposed change:** annotate PR #50 / E1 / E2 conclusions with the population
caveat; if H1 lands, the two populations converge and the studies become
directly applicable.
**How to validate:** re-measure the live entered-pair dislocation distribution
after H1; it should approach 100%.

---

## 6. Recommended changes (ranked)

| # | Change | Type | Effort | Rationale |
|---|---|---|---|---|
| 1 | **Add the entry dislocation gate** (`z ≤ −entry_threshold` required to enter) | Structural | Low | The strategy does not currently implement its own stated entry rule. Highest-impact item in this program; also makes `entry_threshold` a real parameter. |
| 2 | **Move `z_depth` from score to gate** (drop `w_z_depth`, or fix the `abs(z)` asymmetry) | Structural | Low | Removes a binary flag that dominates ranking at the smallest weight. Depends on #1. |
| 3 | **Annotate replay-based conclusions with the population caveat** | Correction | Trivial | PR #50, E1, E2 describe a strategy the live code doesn't implement. |
| 4 | **Amend the "anti-predictive" claim** in the E2 ledger row and plan | Correction | Trivial | Date-clustered CI includes zero (§1). Done in this commit. |
| 5 | Re-examine `entry_threshold`'s tuning role | Tuning | Low | Currently near-inert for entry; after #1 it becomes a genuine lever and its bounds should be revisited. |

**Explicitly not recommended:** re-weighting the composite (PR #50 settled
that), or adding new score components before #1 — with no entry gate, a better
ranker still buys pairs at arbitrary spread positions.

## 7. Go / no-go

**GO on #1 and #2, gated on a comparative backtest.** This is a behavioural
change to the strategy, not a scoring tweak: it will reduce trade count and
change the return profile. It should run as a two-arm comparative backtest
(backtest-agent workflow) over at least two regimes, watching mean P&L per
round trip, win rate, trade count, `candidates_buy_ready`, and cash ratio —
the last because a gate that starves the book will show up as idle capital.

**Sequencing note:** this supersedes the Chronos-2 workstream (WS4) in
priority. C3-A's selection test is defined against the replay pool and the
current composite; both change under #1 and #2, so running it first would
measure a configuration about to be replaced.

**Caveats.** The 175 live entered pairs come from three runs sharing one
parameter set; the dislocation distribution may differ under other settings.
The replay-pool statistics inherit the non-independence documented in
`2026-08-01_disasters-surviving-the-event-gate.md` §3b. Nothing here is
evidence that a gated strategy *will* be profitable — only that the current one
is not doing what it claims, which makes its measured edge uninterpretable.

## 8. Execution checklist

- [ ] Implement the entry dislocation gate in `BobsBrain` Phase 3
- [ ] Verify candidate supply: log qualifying candidates/day vs `k_target`
- [ ] Decide `z_depth`'s fate (drop from score vs fix `abs(z)`); update
      `parameter_space.py`, `_WEIGHT_NAMES`, `create_run()`, README
- [ ] Comparative backtest (gate off / on), 2+ regimes
- [ ] Re-measure live entry dislocation distribution; expect ~100% fully dislocated
- [ ] Annotate PR #50 / E1 / E2 write-ups with the population caveat
- [ ] Revisit `entry_threshold` bounds in the tuning space once the gate is live
