# Composite Score Overhaul: Dead-Weight Removal, Filing Events, NAV Data, Chronos-2

**Date:** 2026-07-31
**Status:** PROPOSED
**Depends on:** PR #50 (Pass A v4 corrected conclusion), 2026-07-31 earnings-event
retro analysis (session work, formalized as Step 0 below)

## 1. Background — what we know

From PR #50 (optimization-free re-analysis of the composite score):

- **Cointegration and half-life components are dead weight for ranking** — near-zero
  rank correlation with forward P&L at every lookback tested.
- **Correlation quality shapes the tail, not the average**: higher correlation means
  smaller wins and fewer disasters, roughly cancelling. Reweighting the score cannot
  manufacture average edge.
- The average P&L problem is **concentrated losses**: the worst ~3 pairs per regime
  own 45-80% of all losses. Remove the catastrophic ~8% and the strategy is solidly
  positive in every regime tested.

From the 2026-07-31 earnings-event retro analysis (175 entered pairs, 3 regimes,
14 catastrophic round-trips, where *catastrophic* = a round trip losing more than
100 bps — one basis point (bp) = 0.01% of the position's notional value):

- **66% of catastrophic loss magnitude came from pairs with an earnings
  announcement inside the holding window.** Among pairs with at least one common
  stock as a leg, 6 of 7 disasters had in-window earnings vs 23% of survivors
  (Fisher exact p = 0.004).
- Earnings exposure is **two-sided**: exposed pairs also produced the biggest wins
  (+150 bps average among exposed survivors). A full veto nets +1,795 bps pooled,
  but the entire benefit sits in one fold (bull_2023, one -1,554 bps blow-up);
  in the other two folds the veto costs ~270 bps each.
- **Design decision:** event-exposed candidates are **excluded from consideration
  entirely**. The program's edge is finding statistical relationships between
  stocks and harvesting their mean reversion; an earnings surprise is exactly the
  kind of shock that breaks those relationships, and the strategy has no ability
  to predict surprise direction — so the event-window wins above are coin-flip
  variance, not edge. Excluding them trades ~270 bps per normal fold for removing
  the left tail that has been flipping fold means negative. That is variance
  reduction at roughly zero expected cost, and it is accepted here as a design
  principle, not tuned as a parameter.
- An EDGAR spot-check showed Yahoo's earnings calendar misses interim results
  disclosures (the RGLD case): item-coded 8-K filings are the better source.
- The 7 fund-only disasters (30% of catastrophic losses) are not earnings-shaped:
  3 look like closed-end-fund discount dislocations (addressable with NAV data),
  the rest are macro shocks (not addressable with any filing data — that slice
  belongs to position sizing).

This plan turns those findings into four workstreams, each gated by a small study
that follows the rigor rules in `.claude/skills/optuna-study.md`.

## 2. Terminology

- **bps** — Basis points. 100 bps = 1% of the position's gross notional value.
- **Catastrophic pair** — A round trip that loses more than 100 bps gross.
- **8-K** — An SEC filing companies must submit when something material happens.
  Each 8-K carries **item codes** saying what kind of event it is (e.g. 2.02 =
  results of operations, 1.01/2.01 = deals/M&A, 5.02 = executive changes, 4.02 =
  restatement).
- **NAV / discount** — A closed-end fund's Net Asset Value = per-share value of
  what it holds. The **discount** is how far the market price sits below (or above)
  NAV. Sudden discount moves can break a fund pair.
- **Independent unit** — The thing you can legitimately count as one observation
  when judging statistical significance. Here: one pair round-trip, not one fold.
- **Walk-forward split** — Train/decide on an earlier time window, evaluate on a
  later one, with an **embargo** (a mandatory time gap between the two so
  information can't leak across the boundary).
- **Final test window** — A time slice that no design decision ever touches, scored
  exactly once at the end for the honest number.
- **Veto horizon** — How many trading days ahead the strategy looks for a
  scheduled earnings date when deciding whether a stock may enter the candidate
  pool today.
- **Guard panel** — The mandatory checks in `tuning/overfit_guards.py`: unit-level
  signal, null baseline, held-out gap, seed stability, complexity ratio.

## 3. Step 0 — Formalize the earnings retro analysis (prerequisite)

The 2026-07-31 analysis lives in session scratchpad scripts. Before anything builds
on it:

1. Move the three scripts into the repo as `tuning/studies/study2_event_retro/`
   (build outcomes, fetch earnings, event join), runnable end-to-end.
2. Write the findings up as `docs/deepdives/2026-07-31_earnings-events-catastrophic-losses.md`
   (numbers, caveats: n = 14 disasters, duration confound, yfinance date fuzziness,
   actual-vs-expected date proxy).

**Size:** about half a day. No study gates needed — it is a record of completed
descriptive work, not a new claim.

## 4. Workstream 1 — Remove dead weight from the composite score

**Change:** drop `w_coint` and `w_halflife` from `_composite_score` in
`BobsBrain.py`; the score becomes corr_long + corr_short + z_depth. Keep any
*gating* use of cointegration untouched — PR #50 only proved these components
can't rank pairs that already passed the gates; it says nothing about the gates
themselves.

**Consistency work (required, per pr-reviewer Step 11):**
- Remove both weights from `tuning/parameter_space.py` (including `_WEIGHT_NAMES`)
  and from `create_run()` settings and `BobsBrain.initialize()` defaults.
- Update tests; update README scoring description.
- The DB `pairs` columns (`score_coint`, `score_halflife`, `coint_pvalue`,
  `halflife_days`) **stay** — they are observability and future-study inputs, and
  WS2/WS4 read them.

**Verification (not an Optuna study — nothing is fitted):**
- Re-rank the cached study pairs with and without the two components; report the
  rank correlation between old and new orderings and the P&L of the top-K under
  both. Expected: near-identical (the components were flat), which is the point —
  this is a simplification, not a performance claim.
- One ~2-week smoke backtest to confirm nothing structural broke
  (backtest-agent workflow, smoke-test row).

**Go/no-go:** if removal materially reorders the top-K (rank correlation below
~0.9), stop and investigate before merging — that would contradict PR #50.

**Size:** ~1 day including the smoke test.

## 5. Workstream 2 — EDGAR filing events

### 5a. Data pipeline

Build a `filing_events` DB table with columns `symbol`, `cik`, `form`, `items`,
`filed_at` (acceptance timestamp), and `source`. Backfill from EDGAR's free
submissions API for the tradeable stock universe, 2021 to present; nightly
incremental refresh.
Point-in-time honesty: an event "exists" only from its acceptance timestamp
forward. Keep the yfinance earnings fetch as a cross-check, not a source.

For **scheduled future earnings** (needed live, not in retro): nightly pull from
Finnhub or FMP's free earnings-calendar endpoint into the same table with
`form = 'EARNINGS_SCHEDULED'`.

**Size:** ~2 days (CIK-to-ticker mapping is the fiddly part).

### 5b. Study E1 — do item-coded events separate disasters? (descriptive)

- **Objective:** measure, per 8-K item group, how strongly "event inside the
  holding window" separates catastrophic from normal round trips — replacing the
  Yahoo-calendar version of this finding with the better source, on more data.
- **Data:** widen beyond the 175 entered pairs by replaying the full-pool
  observation cache (`scoring_replay`) restricted to tradeable entries
  (abs(z) at least 2) — thousands of pair-observations instead of 175, which fixes
  the thin-disaster-count problem (n = 14) of the session analysis.
- **Independent unit:** the pair round-trip. Complexity ratio: green (no free
  parameters — item groups and the window definition are **preregistered here**:
  groups {2.02 results, 1.01/2.01 deals, 5.02 exec changes, 4.02 restatements,
  7.01 guidance}, window = actual holding period plus the 7 days before entry).
- **Gates:** none needed beyond preregistration — nothing is fitted. Report
  per-fold rates with confidence intervals; report the duration confound
  explicitly (longer holds mechanically catch more events).
- **Success criterion:** at least one item group shows disaster separation with a
  confidence interval excluding zero in the pooled data **and** the direction is
  consistent in every fold. Expected winner: 2.02.
- **Size:** ~1 day once the pipeline exists.

### 5c. Study E2 — event exclusion (the tradeable rule)

**Design decision (fixed, not studied):** candidates exposed to a scheduled
earnings event are removed from consideration entirely — no scoring, no sizing
haircut. Rationale in section 1: events break the statistical relationships this
program exists to exploit, and the strategy has no edge in surprise direction.
The study below does not revisit *whether* to exclude; it right-sizes *how*.

**Where the exclusion lives.** At the **daily candidate scan, before scoring**: a
stock with a scheduled earnings date within the veto horizon simply does not
enter that day's pool (and no pair containing it does). Not inside clustering —
clusters group months of price behavior and are recomputed every
`cluster_recompute_days`, while earnings exposure is a property of a specific
date (every stock reports quarterly, so nothing is permanently event-free).
Excluding at the scan gives the same protection earlier and cheaper than scoring
ever would, which is the spirit of "remove before clustering": the ticker never
becomes a candidate that day.

**Three mechanisms, because only scheduled events are knowable in advance:**

1. **Entry veto** — either leg has scheduled earnings within the veto horizon:
   the pair cannot be entered today.
2. **Pre-event exit** — a *held* pair's leg has earnings coming inside the next
   few days: exit before the event. This is required for coverage, not optional:
   earnings dates are typically announced only 2-4 weeks ahead, so an entry veto
   alone has blind spots (date not yet public at entry, or a hold that outlasts
   the horizon). In the retro data the disaster events hit 8-20 trading days
   into the hold — several would only have been caught by this exit.
3. **Reactive exit** — a surprise filing (deal/M&A 1.01/2.01, restatement 4.02)
   arrives on a held leg: exit on the next open. Surprises can never be vetoed
   in advance; this is the only possible defense, driven by the nightly EDGAR
   refresh.

- **Objective:** choose the two timing parameters and confirm the net effect of
  the exclusion package on held-out data.
- **Free parameters (preregistered grids):** veto horizon H in {10, 15, 20, 25}
  trading days; pre-event exit lead in {2, 5} trading days. Reactive exit is
  on/off (compared, not tuned). Two to three free parameters over thousands of
  replay round trips — complexity ratio green.
- **Rigor (per the optuna-study skill):**
  - Unit-level signal gate: E1 *is* that gate — E2 does not start unless E1 passed.
  - Temporal split: choose parameters on walk-forward windows
    (`walk_forward_splits` with positive embargo at least the max holding period,
    40 trading days, so overlapping forward outcomes can't leak); **reserve a
    final test window** (most recent period with data, untouched by E1/E2 design)
    and score the chosen configuration there exactly once.
  - Guard panel: null baseline (shuffle event dates across the calendar — the
    rule must beat at least 90% of shuffled versions), holdout gap, fragility
    (does the chosen H survive dropping any one fold?).
- **Also measured (honesty about cost, not a decision input):** share of
  stock-candidate-days excluded — at H = 20 the quarterly reporting cycle
  (about 63 trading days) implies roughly a third of stock candidacies vanish —
  plus the effect on `candidates_found` / `candidates_buy_ready`, and the
  pooled-mean cost in folds without a blow-up. These are reported so the
  premium being paid is on the record.
- **Success criterion (set before running):** on held-out windows the exclusion
  package removes the earnings-driven catastrophic losses (worst-trade loss cut
  materially, at least 30%) without degrading the pooled mean by more than its
  bootstrap confidence interval. Report the final-test number as the headline.
- **Go/no-go:** pass, then implement in `BobsBrain` (veto at the scan, the two
  exit rules in the exit path; new params registered in `parameter_space.py` +
  `create_run()`; a new `exit_reason` value for event exits), then a comparative
  backtest (backtest-agent workflow, two runs: exclusion off / on).
- **Size:** ~2-3 days plus one comparative backtest.

## 6. Workstream 3 — NAV / discount data for fund pairs

### 6a. Data pipeline

Daily NAVs via the free Nasdaq mirror symbols (form `X` + ticker + `X`, e.g.
`XVKQX` / `XVMOX` / `XMXEX`, verified working), stored in the DB (either
`stock_prices` under the mirror symbol or a small `nav_prices` table). Backfill
2021 to present for every closed-end fund in the universe; nightly refresh.
Day-T NAV becomes usable at day T+1 (it is struck after the close). Report
coverage — some funds (e.g. MXF) have no mirror symbol and stay uncovered; do
not silently proxy.

**Size:** ~1 day.

### 6b. Study N1 — does the discount z-score flag fund-pair disasters?

- **Objective:** test whether an unusually stretched discount at entry (the
  discount's z-score against its own trailing history) separates catastrophic
  fund pairs from normal ones — the analogue of E1 for the fund side.
- **Data / unit:** fund-pair round trips from the same replay pool as E1.
- **Preregistered feature:** max absolute discount z-score across the two legs at
  entry, trailing 1-year window. No fitting.
- **Success criterion:** disaster separation with CI excluding zero pooled +
  consistent direction per fold. In-sample sanity anchor: it should flag the
  known cases (MXF/MXE, VKQ/VMO).
- **If it passes:** fold into E2's exclusion rule under the same design
  principle — a violently dislocated discount is a broken relationship, not an
  opportunity — so "event-exposed" becomes "scheduled earnings OR stretched
  discount", with the discount threshold preregistered from N1's separation
  point. Re-run E2's walk-forward selection once with the combined flag — not a
  separate tuning pass.
- **Honest expectation:** in-sample support is 3 disasters; this study mainly
  exists to size the effect on the wider replay pool. If the CI includes zero,
  record the negative result and stop — the macro-shock remainder of the fund
  tail is a position-sizing/diversification problem, not a data problem.
- **Size:** ~1 day once the pipeline exists.

## 7. Workstream 4 — Chronos-2 for candidate selection and sizing

Chronos-2 (Amazon's open-source time-series foundation model) forecasts a series
zero-shot — no training on our data — and returns a forecast *distribution*
(a range of likely paths with probabilities, not a single guess).

**Framing.** Chronos-2 is **not** disaster insurance — that job is done upstream
by the WS2/WS3 exclusion rules, and no event information (scheduled earnings,
filings) is fed to it. It runs on the **cleaned candidate pool**: dead-weight
components removed (WS1), event-exposed candidates excluded (WS2c/N1). That pool
is deliberately the world Chronos-2 has the best chance in — what remains after
exclusion is the statistical mean-reversion behavior a price-history model can
in principle capture. Its job is to **add edge, not avoid losses**, in one or
both of two roles:

1. **Candidate selection** — rank candidates by how likely the spread is to
   actually converge, information the current score components (correlations,
   z-depth) measure only indirectly. PR #50 showed the existing components
   cannot rank average outcomes; a calibrated convergence forecast is the first
   candidate feature with a plausible mechanism for doing so.
2. **Position sizing** — size positions by forecast confidence: tight forecast
   distribution, larger position; wide or left-skewed distribution, smaller.
   The current dynamic-K quality scaling uses the composite score for this; a
   forecast distribution is a more direct measurement of the same intent.

### Study C3-A — feature evaluation on the cleaned pool (no integration)

- **Objective:** do Chronos-2 forecast features add candidate-ranking signal
  beyond the current score components, and/or sizing signal beyond the current
  quality scaling, on the post-exclusion replay pool?
- **Setup:** for each post-exclusion replay-pool entry, run Chronos-2 zero-shot
  on price history only (the spread series, and the two legs jointly), producing
  three preregistered features:
  1. convergence probability (share of forecast paths reverting past the exit
     threshold within the horizon),
  2. forecast interval width (how uncertain the forecast is),
  3. left-tail width (distance to the 5th-percentile path).
- **Independent unit:** pair round-trip. The model is frozen and the features
  preregistered, so the main risks are metric-shopping and eval reuse, handled
  by: metrics fixed in advance (below), and the **same reserved final test
  window as E2, still scored exactly once**.
- **Selection test:** unit-level marginal signal — does convergence probability
  rank forward gross P&L *after controlling for* the existing components
  (corr_long, corr_short, z_depth)? Evaluated with `unit_level_signal` on the
  residual ranking, per fold, bootstrap CIs. Plus the portfolio version: P&L of
  top-K selected by score-plus-feature vs score alone, on walk-forward windows.
- **Sizing test:** simulated P&L with positions scaled by forecast confidence
  (inverse interval width) vs (a) equal sizing and (b) the current quality
  scaling — same capital budget, pooled mean and left tail, walk-forward
  windows. Beating equal sizing but not quality scaling = failure; the baseline
  to beat is what the strategy already does.
- **Success criteria (set before running, per role):** selection — marginal
  ranking signal with CI excluding zero on held-out windows AND top-K P&L
  improvement in at least 2 of 3 folds. Sizing — held-out pooled improvement
  over quality scaling with CI excluding zero. Either role can pass
  independently; matching the existing baseline = failure for that role (the
  baseline is free).
- **Compute:** small model, CPU-feasible; roughly a weekend of wall-clock for
  the full replay pool. `price_cache_only` semantics apply — forecasts read the
  DB cache, never live APIs.
- **Size:** ~3 days.

### Study C3-B — integration (only for roles that pass C3-A)

- **Selection passes:** add the convergence-probability feature to candidate
  ranking — as a new weighted score component (`w_conv`), registered in
  `parameter_space.py` and `_WEIGHT_NAMES`, defaulting to a weight chosen on
  walk-forward windows with the full guard panel.
- **Sizing passes:** replace or blend the quality-scaling input with forecast
  confidence in the dynamic-K sizing path; parameters registered the same way.
- Either way: comparative backtest per the backtest-agent workflow (feature off
  / on), full guard panel on any fitted threshold or weight.
- **Operational note for live use:** forecasts must be produced within the daily
  cycle for every scored candidate; measure per-candidate inference time in
  C3-A and record the projected daily cost in the study ledger before C3-B
  starts.

If C3-A fails both roles, record it in the ledger and stop — the cleaned
baseline stands, and that negative result is itself the deliverable.

## 8. Sequencing and decision points

- Step 0 (formalize retro) — 0.5 day
- WS1 dead-weight removal — 1 day; merge on rank-equivalence + smoke test
- WS2a EDGAR pipeline — 2 days
- WS2b Study E1 (descriptive) — 1 day; GATE: an item group separates disasters
- WS3a NAV pipeline (parallel with E1) — 1 day
- WS3b Study N1 (descriptive) — 1 day; fold into E2 flag if it passes
- WS2c Study E2 (event exclusion) — 2-3 days; GATE: held-out tail removal at
  acceptable mean cost; final test once; then implement + comparative backtest
- WS4 Study C3-A (selection + sizing value on the cleaned pool) — 3 days;
  GATE: beats the existing score / quality-scaling baselines held-out;
  C3-B integration only for roles that pass

Total: ~2 weeks of focused work, with three explicit stop points where a negative
result ends a branch cleanly and is written into the study ledger
(`docs/plans/2026-07-12_cloud-tuning-studies.md`).

## 9. Execution checklist

- [ ] Step 0: retro scripts to `tuning/studies/study2_event_retro/`; deepdive doc written
- [ ] WS1: weights removed; `parameter_space.py`, `_WEIGHT_NAMES`, `create_run()`, README
      updated; rank-equivalence verified; smoke backtest clean
- [ ] WS2a: `filing_events` table + EDGAR backfill + nightly refresh + scheduled-earnings feed
- [ ] WS2b: Study E1 run; per-item-group separation reported with CIs; ledger row
- [ ] WS3a: NAV mirror-symbol backfill + coverage report
- [ ] WS3b: Study N1 run; ledger row (positive or negative)
- [ ] WS2c: Study E2 walk-forward selection (veto horizon + exit leads) + guard
      panel + final-test score; veto and event exits implemented; params
      registered; new `exit_reason` value; comparative backtest
- [ ] WS4: Study C3-A run (selection + sizing tests vs existing baselines);
      ledger row; C3-B only for roles that pass
- [ ] Ledger updated in `docs/plans/2026-07-12_cloud-tuning-studies.md` after each study
