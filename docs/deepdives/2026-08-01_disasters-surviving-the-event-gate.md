# Disasters Surviving the Event Gate

**Date:** 2026-08-01
**Data:** Pass A v4 replay pool — 2,043 tradeable observations across 12 scoring
dates in 3 regime folds; 275 catastrophic (< −100 bps gross), −124,896 bps of
catastrophic loss. Forward paths verified 2,043/2,043 against the cached
outcomes.
**Scripts:** `tuning/studies/study3_disaster_anatomy.py`
**Follows:** E1 (GATE PASSED), E2 v1+v2 (NO-GO) — `docs/plans/2026-07-31_composite-score-overhaul.md` §5b/§5c

## Question

E1 showed filing events discriminate disasters (results-8-K odds ratio 3.6,
p = 1e-16). E2 showed excluding event-exposed candidates does not help the
portfolio. This deepdive asks why those coexist, what the disasters that
survive an event gate have in common, and what data would identify them.

---

## 1. Why E1 and E2 diverge — three distinct reasons

### 1a. The veto set was not the flag set (design gap)

E1 flagged an event **anywhere** in `(entry − 7d, exit]`. E2 vetoed only events
in the **first H ≤ 25 trading days after entry**. Splitting the pool by when the
event lands:

| results event at | n | mean bps | median | disaster % | knowable at entry? |
|---|---|---|---|---|---|
| pre-entry only | 37 | **+69.3** | +56.6 | 24.3% | yes — already filed |
| t = 1–10 (H=10 sees) | 231 | **+71.2** | +140.2 | 20.8% | usually |
| t = 11–25 (H=25 sees) | 79 | **+57.7** | +87.8 | 17.7% | often |
| t = 26–40 (no H sees) | 169 | **−91.4** | +106.2 | 19.5% | rarely |
| no results event | 1,527 | +33.3 | +34.8 | 11.2% | — |

Counterfactual pool means (baseline +28.8):

| exclusion | pool mean | Δ |
|---|---|---|
| exclude t = 1–10 | +23.4 | **−5.4** |
| exclude t = 11–25 | +27.7 | −1.2 |
| exclude t = 26–40 | +39.7 | **+10.8** |
| exclude all results-exposed | +33.3 | +4.4 |

**E2's veto excluded the profitable exposures and kept the harmful ones.** Early
events resolve the dislocation — often favourably (median +140 bps at t = 1–10).
Late events strike pairs that are *already* failing to converge, and those are
the ones that blow up. The disaster *rate* is roughly flat across event timing
(18–21%); only the loss *magnitude* is timing-dependent.

This is a genuine design gap, not a data limitation — but the fix is only
partly reachable: t = 26–40 events are mostly not on the calendar at entry
(earnings dates publish 2–4 weeks ahead). A veto extended to H = 40 would rely
on information that does not exist at decision time for the later half.

### 1b. Coverage: events mark a minority of disasters

Even taking *any* results exposure across the whole window: **104 of 275
disasters caught (38%)**, carrying 61% of catastrophic loss. **171 disasters
survive**, carrying 39% (−48,590 bps). Broadening to any filing type barely
helps — **162 disasters (59% of all) have no filing of any kind** in their
window.

### 1c. The flag marks variance, not negative expectancy

Flagged observations: mean −63.1, **median +95.4**, sd 722, p95 +557.
Unflagged: mean +45.2, median +41.0, sd 248, p95 +293. Events raise *both*
tails; the flagged median is more than twice the unflagged median. E1's design
(disaster vs rest) is blind to this by construction — it asks whether disasters
are event-enriched, not whether the flagged set has negative expectancy.

This is the third time this program has found the same shape: **PR #50's
correlation finding, the entered-pairs veto ledger, and now the event flag are
all variance/tail dials, not edge dials.**

---

## 2. The surviving disasters: discovery features do not see them

Median values, surviving disasters vs everything else:

| feature | surviving disaster | caught disaster | normal trade |
|---|---|---|---|
| corr_long | 0.895 | 0.796 | 0.884 |
| corr_short | 0.913 | 0.847 | 0.914 |
| z_entry | 2.358 | 2.291 | 2.302 |
| coint_pvalue | 0.557 | 0.574 | 0.498 |
| halflife_days | 9.29 | 9.55 | 7.52 |

**Surviving disasters are indistinguishable from normal trades on every
discovery-time feature** — their correlations are, if anything, slightly
*higher*. Only half-life and cointegration p-value differ at all, and both are
components PR #50 already showed cannot rank outcomes. This is the entry-
identifiability question from the 2026-07-18 exit-redesign update, answered
negatively for the current feature set.

What *does* differ is the path:

| group | median hold | median max drawdown | exits at 40-day cap |
|---|---|---|---|
| surviving disaster | 29 d | −338 bps | 30% |
| caught disaster | 29 d | −682 bps | 19% |
| normal | 10 d | −24 bps | 1% |

Disasters are **non-converging pairs that run to the cap**. Non-convergence is
visible early — by day 10 a drawdown ≤ −300 bps implies a 68.6% disaster rate
against a 13.5% base — but acting on that is a stop-loss, and the 2026-07-18
retro falsified every loss-side stop tested (a stop tight enough to catch the
blowups fires on 13–36% of pairs and sacrifices more winners than it saves).
Reported here as characterization, not a proposal.

---

## 3. The real structure: disasters are concentrated, not distributed

### 3a. Three dates carry 73% of all catastrophic loss

| date | n | mean | disasters | share of disaster loss |
|---|---|---|---|---|
| 2023-03-15 | 451 | +72.6 | 64 | **32.9%** |
| 2023-02-15 | 111 | −68.5 | 25 | **21.8%** |
| 2022-01-14 | 322 | +13.1 | 64 | **17.9%** |
| other 9 dates | 1,159 | — | 122 | 27.4% |

All three are regime breaks: the January 2022 rate-hike selloff, the February
2023 hot-CPI reversal, and the March 2023 SVB banking crisis. Note the pool mean
*excluding* those three dates is +25.5 vs +28.8 overall — the bad dates also
produced the big winners, so this is again two-sided.

### 3b. One broken relationship becomes N simultaneous disasters

The clearest single finding in this analysis:

| symbol | appearances | disasters | mean | concentration |
|---|---|---|---|---|
| PTNQ | 37 | **29** | −131 | 29 of 29 disasters on **2022-03-15 alone** |
| KBWY | 8 | 7 | −384 | all 8 on **2023-03-15** (SVB week) |
| HOMB | 10 | 4 | −229 | 7 of 10 on 2022-01-14 |
| ROBT | 27 | 8 | −38 | 5 of 8 disasters on 2022-01-14 |

PTNQ is not a persistently bad symbol — it traded fine on three other dates. On
2022-03-15 it appeared in 30 pairs and 29 of them blew up together. PTNQ is a
*trend-following* ETF that mechanically rotates between equity exposure and
T-bills; when it switched, it decoupled from every equity partner
simultaneously. Pairs sharing a leg used ≥ 5 times on the same date account for
**31% of all catastrophic loss**.

**Live-strategy caveat (important):** `BobsBrain.before_market_opens` skips any
candidate whose either leg is already in `self.pairs` or `position_symbols`, so
a live book cannot hold PTNQ 30 times. This concentration is partly an artifact
of the replay harness, which scores all candidate pairs without that dedup. Two
consequences: (i) the live exposure is smaller than these numbers suggest, and
(ii) **the replay pool's observations are not independent units.**

### 3c. Consequence — E1's confidence interval was overstated

Re-running E1's headline with the **date** as the resampling unit instead of the
pair:

| estimate | CI |
|---|---|
| pair-level bootstrap (as reported in E1) | +15.6 .. +27.0 pp |
| date-clustered bootstrap | **+9.4 .. +33.3 pp** |

The finding survives (the interval still excludes zero), but it is
substantially weaker than reported, and per-date direction is not universal:
2022-03-15 shows +0.0 pp and 2023-05-15 shows −2.0 pp. **E1's plan entry and
ledger row should be amended** to quote the clustered interval.

---

## 4. Side-finding: the ETF metadata that gates clustering is broken

Of 1,062 symbols in the pool, only **62** are flagged `is_etf = True`. Spot
checks: QQQ, TQQQ, PTNQ, ROBT, SDVY, MILN, PHO, FIXD, KBWY, VONV all return
`is_etf = False` with a NULL sector.

`TickerClusterer` **pre-partitions by sector with ETFs as their own partition**.
With `is_etf` this broken, that partition is largely non-functional and ETFs are
being clustered as if they were unknown-sector common stocks — which is exactly
how a trend-following ETF ends up paired with 30 equities. This is a live bug
affecting discovery today, not just a replay artifact.

## 5. Positive control: where economic linkage is real, the strategy works

| pair composition | n | mean | median | disaster % |
|---|---|---|---|---|
| common + preferred | 27 | **+271.8** | +207.1 | 14.8% |
| preferred + preferred | 157 | **+96.4** | +90.1 | 10.8% |
| common + common | 1,758 | +22.5 | +42.2 | 13.7% |
| ETF + ETF (as labelled) | 76 | −14.8 | +9.9 | 11.8% |
| common + ETF (as labelled) | 25 | −77.9 | +34.5 | 16.0% |

Best individual symbols are same-sector bank preferreds — SYF.PRA (+440 over 20
appearances, zero disasters), WTFC (+358), WRB.PRE (+293), USB.PRH (+274).
Preferred shares of comparable issuers are genuinely economically linked and
mean-revert reliably. (Note the ETF rows are unreliable given §4; the true ETF
population is far larger than 101 pairs.)

**This is the organizing insight:** disasters come from pairs with high
*statistical* correlation but no *economic* linkage. The strategy's edge is
real where a mechanism enforces convergence, and absent where correlation is
coincidence.

---

## 6. Hypotheses

**H1: Instrument structure, not events, is the primary identifiable disaster
driver.**
**Evidence:** PTNQ (trend-switching ETF) → 29 simultaneous disasters; the ETF
partition gate is non-functional (§4); economically linked pairs (preferreds)
are the best-performing cohort (§5) while structurally mismatched pairs blow up.
**Mechanism:** instruments with mechanical rebalancing (leveraged, inverse,
trend-following, bullet-maturity) or no shared economic driver exhibit
correlation without a convergence mechanism. When the mechanical rule fires, the
relationship breaks permanently rather than mean-reverting.
**Proposed change:** repair `is_etf`/sector metadata, then add an instrument-
structure screen that excludes mechanically-rebalancing fund types from pair
formation.
**How to validate:** re-score the replay pool with the screen; disaster count
and loss should fall without materially reducing the pool mean (unlike the
event veto, which cut both tails symmetrically).

**H2: Correlated exposure amplifies single breaks into portfolio events.**
**Evidence:** 31% of catastrophic loss comes from pairs sharing a leg used ≥ 5×
on one date; 73% of loss falls on 3 of 12 dates.
**Mechanism:** one symbol's structural break propagates to every pair containing
it; regime breaks hit many pairs at once.
**Proposed change:** verify the live per-symbol dedup is actually binding in
backtests, and consider a per-cluster (not just per-symbol) concentration cap.
**How to validate:** measure realized per-symbol and per-cluster book
concentration in a backtest; this is the one hypothesis whose test must be a
backtest, since the replay pool cannot express portfolio construction.

**H3: The current discovery feature set cannot identify disasters at entry.**
**Evidence:** §2 — surviving disasters match normal trades on corr_long,
corr_short, z_entry; only half-life and coint p-value differ, both already
falsified as rankers by PR #50.
**Mechanism:** all current features measure *statistical* association strength;
none measures whether an economic mechanism enforces convergence.
**Proposed change:** stop adding refinements to correlation-family features;
invest in linkage features (shared SIC/sector, index co-membership, issuer
relationship, fund-structure class).
**How to validate:** unit-level marginal signal of a linkage feature vs the
existing components on the same replay pool, per the E1 methodology.

---

## 7. Recommended changes (ranked)

| # | Change | Type | Effort | Rationale |
|---|---|---|---|---|
| 1 | **Fix `is_etf` / sector metadata** | Bug fix | Low | 62/1,062 coverage silently disables the ETF partition in live clustering. Everything else in H1 depends on it. |
| 2 | **Instrument-structure screen** (exclude leveraged / inverse / trend-following / bullet-maturity funds from pair formation) | Structural | Low–Med | Directly targets the largest identifiable disaster mechanism; knowable at entry at zero data cost once #1 lands. |
| 3 | **Verify per-symbol dedup + add concentration cap** | Structural | Med | 31% of loss traces to leg-sharing; live dedup exists but is unverified in backtest. |
| 4 | **Amend E1's reported CI to the date-clustered interval** | Correction | Trivial | The pair-level CI overstates precision; the plan and ledger should carry the honest number. |
| 5 | **Linkage features** (shared SIC, index co-membership, issuer relationship) | Research | Med | The only feature family with a plausible mechanism for the residual disasters. |
| 6 | Regime/volatility awareness (VIX/MOVE at entry) | Research | Med | 73% of loss on 3 regime-break dates — but those dates also carried the winners, so treat as sizing input, not a filter. |

**Explicitly not recommended:** extending the event veto to H = 40 (the
information does not exist at entry for late events); any drawdown-triggered
stop (falsified 2026-07-18); event-based exclusion as a mean-improving measure
(E2, NO-GO).

## 8. Go / no-go

**GO on #1 and #2** — the instrument-structure thread is the first disaster
mechanism in this program that is (a) identifiable at entry, (b) mechanistically
explained rather than statistically inferred, and (c) asymmetric — unlike
events, correlation, and exit stops, all of which cut both tails.

**NO-GO on further event-based work.** E1's discrimination is real but weaker
than reported, covers a minority of disasters, and marks variance rather than
negative expectancy. The event pipeline stays as observability and as an input
to the reactive-exit mechanism, which remains untested.

**Caveats.** 12 scoring dates in 3 correlated folds; observations are not
independent (§3b) and every pooled statistic here should be read with the
date-clustered interval in mind; the universe is current-membership, not
point-in-time (survivorship, inherited from PR #50); and the replay pool does
not model portfolio construction, so H2 cannot be settled here.

## 9. Execution checklist

- [ ] Audit `ticker_metadata` population path; determine why `is_etf` is 62/1,062
- [ ] Re-run `scripts/refresh_ticker_metadata.py` and verify ETF coverage
- [ ] Define the instrument-structure classes to exclude; source the data
- [ ] Re-score the replay pool with the screen; report disaster count, loss, pool mean
- [ ] Amend E1's CI in the overhaul plan §5b and the cloud-studies ledger row
- [ ] Backtest with per-symbol/cluster concentration measurement (H2)
