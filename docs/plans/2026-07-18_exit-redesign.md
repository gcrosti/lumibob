# Exit Redesign — Implementation Plan

> Created: 2026-07-18. Implements recommendation #1 (H-B) from
> `docs/deepdives/2026-07-17_pass-a-score-signal-and-exploitability.md`.
> Scope deliberately narrow: **exit mechanics only, gross P&L, no cost model.**
> Entry redesign (H-C) and the simulator cost model (H-D) are explicitly out of
> scope for this pass — see "Deferred."

> **OUTCOME (2026-07-18): Phase 1 retro-validation = NO-GO. H-B falsified.**
> No loss-side stop (divergence, health, time, hard-level, wide-z) beats the
> stopless pooled gross mean; most are worse. The mean ≪ median gap is not
> "losers held too long" — every fold's median is positive (+9 to +24); the mean
> is dragged by a few rare catastrophic non-converters (bull: one pair −1554 bps).
> On the way down these are indistinguishable from the many pairs that dip and
> revert, so any stop tight enough to catch them sacrifices more winners than it
> saves. Phase 2 (implement stops) is **cancelled** — no strategy code from this
> pass. Redirect: entry / pair-quality screening (can the catastrophic pairs be
> identified *at entry*?), then position sizing. Evidence:
> `notebooks/pass_a_v3_score_signal_retroactive.ipynb` §Phase 1. The phased plan
> below is retained as the record of what was tested.

---

## Why

The deepdive confirmed the score identifies pairs that revert (82–87% touch the
exit threshold within 20 td) and whose relationships persist (rho +0.66). The
problem is not signal — it is that the exit system is **one-sided**: a take-profit
(`zscore_exit` at |z| ≤ exit_threshold) with no loss-side control. The ~25% of
spreads that diverge ride until `displaced` (a lagging monitor) notices.

This shows up as **mean ≪ median, at zero cost**:

| Static exit policy | Gross median (bps/trip) | Gross mean (bps/trip) |
|---|---|---|
| z-exit 0.5, 20d cap (≈ current `exit_threshold`) | +12.8 | **−1.1** |
| z-exit 0.75, 20d cap | +12.7 | +5.1 |
| z-exit 1.0, 20d cap (best stopless mean) | +11.0 | **+9.1** |
| Oracle 20d (perfect-foresight ceiling) | +38 | +74 |

The tail — not the take-profit threshold — is what destroys expectancy. The median
is healthy everywhere; the mean is dragged down whenever the exit is loose enough
to let losers run. A tighter take-profit (z=1.0) partly dodges the tail but sacrifices
median capture (+11.0 vs +12.8). **A stop lets a loose, high-median take-profit keep
its median while removing the tail that otherwise forces a lower-median exit.** That
is the entire thesis of this redesign, and the oracle's +74 mean proves the motion
to capture is really there.

## Goal & gate

**Goal:** a loss-side exit system whose gross mean expectancy per round trip beats
the best stopless static policy — **+9.1 bps gross mean** — while preserving the
loose take-profit's median (~+12.8), on a **per-fold** basis across all three folds.

**Gate to proceed from retro-test (Step 1) to implementation (Step 2):**
stop-augmented policy gross **mean ≥ +9.1 bps** in ≥ 2 of 3 folds **and** median not
degraded below ~+11. If no stop configuration clears this, **stop — write no strategy
code** and reconsider the mechanism.

Cost is excluded by design: if gross mean can't beat the stopless baseline, costs are
moot; if it can, the net-of-cost verdict is a later question (see Deferred).

---

## Mechanisms to test

Three loss-side stops, each defined to be computable live (paper/live parity) and
retro-simulable on the 175-pair ladder. Entry is at |z_entry| ≥ entry_threshold; the
reversion thesis is |z| → 0.

1. **Divergence stop.** Exit when the spread moves *against* entry by margin Δ:
   `|z_current| ≥ |z_entry| + Δ_div`. Anchored per-position to the actual entry z
   (entries occur across a range of |z|, not exactly at threshold). Retro sweep:
   Δ_div ∈ {0.5, 1.0, 1.5, 2.0}. Candidate param: `divergence_stop_delta` (Tier 3).

2. **Health stop.** Exit when the statistical relationship deteriorates — the
   leading indicator of divergence that `displaced` catches late. Primary proxy:
   trailing `corr_short_window` correlation falls below a floor (absolute floor, or
   `entry_corr − Δ`). Alternative proxy to test: rolling spread ADF p-value rising
   above a ceiling (more expensive live). Retro sweep: floor ∈ {0.3, 0.4, 0.5}.
   Candidate param: `health_stop_corr_floor` (Tier 3; 0 disables).

3. **Calibrated time stop.** Exit at `k × calibrated_half_life` days if still
   unconverged (|z| still above exit_threshold). Requires the **half-life calibration
   constant** `c` (realized/predicted ≈ 3–5×; derived in Step 1 from the notebook
   data): `calibrated_hl = c × predicted_hl`. Retro sweep: k ∈ {1.0, 1.5, 2.0}.
   Candidate params: `time_stop_halflife_mult` (k, Tier 3); `c` is a fixed derived
   constant in code, not tuned.

Also retro-test the **combined** policy (loose take-profit + best of each stop) — the
stops are complementary (divergence catches fast blowups, health catches slow
breakdowns, time catches dead-money non-events).

### Exit precedence (once implemented)

Per position, per day, first match wins:
`data_missing → zscore_exit (take profit) → divergence_stop → health_stop →
time_stop → displaced`.
Take-profit is checked first (book the win); loss-side stops before displacement so a
diverging pair exits on its own risk signal rather than waiting to be crowded out.

---

## Phases

### Phase 0 — Commit the investigation base
The deepdive, `notebooks/pass_a_v3_score_signal_retroactive.ipynb`, and the studies-doc
ledger edits are uncommitted. Land them (one PR) so this work builds on a reviewed base.

### Phase 1 — Retro-validate in the notebook (no strategy code, ~0 compute)
1. Derive the half-life calibration constant `c` from realized-vs-predicted
   (uncensored pairs) — prerequisite for the time stop.
2. Extend the exploitability ladder with the three stops (+ combined), each swept as
   above, evaluated **gross** per fold.
3. Report gross mean / median / %-positive per policy per fold; identify which stop(s)
   clear the gate.
4. **Decision point:** which stop(s) earn implementation. Only winners proceed.

### Phase 2 — Implement winning stop(s) in `BobsBrain` (conditional on Phase 1)
- Exit logic in the trading-iteration exit path, following the precedence above.
- New `exit_reason` values (`divergence_stop`, `health_stop`, `time_stop`) → update the
  `exit_reason` documentation in CLAUDE.md.
- **Register each new param in `tuning/parameter_space.py`** (Tier 3, default, bounds)
  and add to the `create_run()` settings dict — tuning-engine consistency rule.
- Unit tests per new exit branch (trigger, non-trigger, precedence).

### Phase 3 — Comparative backtest (first cloud compute)
Backtest-agent workflow: current vs. stop-augmented mechanics, **gross (no cost model)**,
2 folds across different regimes. Confirms the retro improvement survives the full
simulator (retro uses proxies: frozen hedge, entered-pairs-only). Gate: gross expectancy
positive and improved vs. baseline.

### Phase 4 — Study question (spec later)
Only after Phase 3 passes. New stop params are Tier 3; Pass A still wants its v4 full-pool
gate. Study shape depends on Phase 3 outcome — not specced here.

---

## Deferred (explicitly out of scope, not forgotten)

- **Cost model (H-D).** Excluded per decision 2026-07-18: we are first establishing
  whether *gross* expectancy is positive at all. Costs return before any "is this
  actually profitable" verdict and before live trading — a positive gross result here
  is necessary, not sufficient. Until then, all numbers in this work are gross and
  labeled as such.
- **Entry magnitude floor (H-C).** Deferred to a fast-follow pass. The exit redesign is
  isolated first so its effect is measurable in one comparative backtest. H-C remains a
  blocker on study resumption per the deepdive.
- **Half-life score-component rescale (deepdive #5).** Scoring fix, not exit; separate.
  (The half-life *calibration constant* is in scope here because the time stop consumes it.)

## Decision points

1. **Phase 1 gate** — which stop(s) clear +9.1 gross mean in ≥ 2/3 folds without
   collapsing median. Data-driven; no code before this.
2. **Param registration** — final Tier/bounds for the surviving stop params (Phase 2).
3. **Phase 3 gate** — comparative backtest confirms retro result in full simulator.

## Execution checklist

- [ ] Phase 0: commit deepdive + notebook + studies-doc edits
- [ ] Phase 1a: derive half-life calibration constant `c`
- [ ] Phase 1b: retro-simulate divergence / health / time / combined stops, gross, per fold
- [ ] Phase 1c: record which stop(s) clear the gate → decision point
- [ ] Phase 2a: implement winning stop(s) in `BobsBrain` + precedence
- [ ] Phase 2b: new `exit_reason` values + CLAUDE.md doc update
- [ ] Phase 2c: register params in `parameter_space.py` + `create_run()` settings
- [ ] Phase 2d: unit tests per exit branch
- [ ] Phase 3: comparative backtest (gross), 2 folds
- [ ] Update studies source-of-truth ledger with the outcome
