# Cloud Tuning Studies — Source of Truth

> Created: 2026-07-12. **This document is the source of truth for the tuning
> study sequence** — what each study is for, its configuration, its gate, and
> its current status. Update the status ledger as studies complete.
>
> Supersedes the study-design sections of
> `2026-05-18_tier3-regime-detector.md` (Step 4) and all of
> `2026-05-19_cloud-tuning-study1-onward.md` (whose infra half was already
> superseded by the cloud infrastructure plan).
>
> Related documents:
> - `2026-05-21_cloud-infrastructure-plan.md` — how to launch instances, tunnels, backups (operational)
> - `TUNING_ENGINE_PLAN.md` — engine design history, tier definitions, deep-dive findings (historical)

---

## Goal

**Ultimate goal** (unchanged from `TUNING_ENGINE_PLAN.md`): tune LumiBob's
parameters so the strategy consistently beats SPY.

The 2026-04-24 deep dive found two structural blockers above the tuning layer,
both since addressed:

- **H5 — pair selection**: only 12% of traded pairs had stationary spreads.
  Fixed by the cointegration gate; validated 2026-04-30 (stationarity rate
  80–88% on both test windows).
- **H1 — long-only beta ~0.57**: a long-only book cannot beat SPY through
  tuning. Fixed by replacing the `enable_short_leg` boolean with the
  continuous, regime-conditionable `short_leg_fraction`; Study 0 validated
  that the parameter is worth tuning (see ledger).

**Goal of this study sequence**: produce a fully tuned, regime-conditioned
parameter set with positive out-of-sample Sharpe on an unseen 2025 holdout.
The SPY-beating gate is deliberately **not** applied inside this sequence —
it is reinstated at the paper-trading phase (cloud infra Phase 2), where a
market-neutral strategy finally produces meaningful signal against it.

The sequence answers four questions in dependency order:

1. **Does the composite score rank pairs correctly, and at what timescales?**
   (Study 1 Pass A) — no point conditioning anything on a score that doesn't
   discriminate.
2. **Given a working score, how should discovery and position sizing exploit
   it?** (Study 1 Pass B)
3. **Do regime-conditioned Tier 3 parameters — including `short_leg_fraction`
   — beat a single static set?** (Study 2)
4. **Does the whole stack hold up out-of-sample on data no study has seen?**
   (Study 3)

```
Study 0 ✓ ──► Study 1 Pass A ──gate──► Study 1 Pass B ──► Study 2 ──gate──► Study 3 ──gate──► paper trading
(short leg     (score quality)          (portfolio         (per-regime        (dense + 2025
 viability)                              construction)      Tier 3)            holdout)
```

Do not proceed past a failed gate without a root-cause investigation.

---

## Status ledger

| Study | Optuna study name | Status | Result |
|---|---|---|---|
| Study 0 — short-leg viability | `study0_sideways_2022`, `study0_bull_2023` | **Done** (2026-05-19, local) | **Gate passed**: best `short_leg_fraction` = 0.96 (sideways), 0.058 (bull) — > 0.05 in ≥ 1 fold. Note: best objective was negative in both folds (−0.48, −0.26); the short leg earned its place as a *parameter*, not yet as a profit source. |
| Study 1 Pass A v1 | `study1_pass_a_v1` | **Deprecated** | 83 trials completed locally (2026-05-20/21, best blended score 0.270), but 3-month folds produced too few round-trips for reliable Spearman rho. Superseded by v2 (5-month folds). Do not use its results. |
| Study 00 — cloud pipe check | `cloud_smoke_v1` | **Done** (2026-07-14) | **Passed** — after catching and fixing a run-attribution bug: under concurrent workers, `_find_run_id`'s most-recent-run heuristic scored the same run for two trials. Fixed via `tuning_trial_token` in `backtest_runs.settings` (branch `feat/study1-pass-b`); rerun verified 3 workers → 3 distinct correctly-matched runs. Also required opening postgres on the DB instance to the VPC subnet (listen_addresses + pg_hba scoped to lumibob/172.31.0.0/16). Measured c6i.4xlarge trial times: 31–95 min per 5-month fold (median ~43). |
| Study 1 Pass A v2 | `study1_pass_a_v2` | **Invalid — do not use** (2026-07-14) | Fleet of 12 workers launched 16:01 UTC; at 16:17 one worker's mass Alpaca fetch failure wrote 4,607 false `failed_tickers` rows, and the global blocklist in `BobsBrain.initialize()` then emptied the universe of every subsequent run: 89 of 100 trials completed degenerate (0 pairs scanned, 0 trades, penalty scores) and the gate's "FAIL 0/3" ran on 0-trade backtests. 11 healthy trials (best 0.2061) are salvageable. Poison rows deleted 2026-07-15 (backup kept); root causes fixed in `fix/failed-ticker-poisoning`. |
| Study 1 Pass A v3 | `study1_pass_a_v3` | **Done** (2026-07-17) — **GATE FAILED** | 97 healthy completed trials (11 seeded + 86 new; 0 failed/pruned, all runs with real fills — incident fixes verified). Best value 0.2061 (a *seeded* v2 trial; 86 new trials did not beat it). Gate re-ran best params on all 3 folds with full backtests: rho = **−0.040 (sideways), −0.111 (bull), −0.046 (mixed)** — 0/3 pass, not borderline. Per sequencing: **do not run Pass B**; investigate score structure first. Note: the best trial's own fold re-run produced negative rho vs its positive in-sample score — in-sample rho estimates may be noise-dominated. Trial times ~54 min avg on c6a.4xlarge with `price_cache_only` (vs ~100 min v2). Ran on spot (reclaimed 2026-07-15 after 10 trials) then on-demand to completion. |
| Study 1 Pass A v4 | cache harness (`tuning/studies/scoring_replay.py` + `scoring_study.py`) | **Done — NO-GO** | Cache-based P&L-free scoring-quality study. **Reweighting the composite does NOT generalize — this is not a lever.** At the live-faithful lookback (152) with out-of-fold validation: pair-level `Spearman(component, forward_gross)` ≈ 0 (well-powered, n 572–946/fold), top-K-by-score is indistinguishable from random-K, and the leave-one-fold-out held-out objective is **negative in all three folds**. The earlier +52 / `w_corr_short`→0.2 result was on a **non-live-faithful 400-day cache without out-of-fold validation — retracted as overfit** (it is the post-mortem behind `tuning/overfit_guards.py`). Re-confirms the deep dive: the composite is a good **filter**, a poor fine-grained **ranker**. **WS1 (log half-life) stands as a valid standalone fix.** Full spec below; plan: `docs/plans/2026-07-22_pass-a-v4-scoring-study.md`. |
| Study 1 Pass B | `study1_pass_b_v1` | Script ready (PR #46) — **re-scoped, blocked** | Now absorbs the params the cache can't optimize (`zscore_window`, `cooldown_days`, `lookback_window`) on top of its discovery/sizing set. Still blocked on the WS4 cost model + a sound post-cost objective. See below. |
| Study 2 — per-regime Tier 3 | `study2_<regime>_v1` (one per regime) | **Blocked — structural NO-GO** | |
| Study 3 — dense walk-forward | `study3_v1` | **Blocked — structural NO-GO** | |

> **Sequencing halt (2026-07-17, updated 2026-07-18)**: the Pass A root-cause
> deepdive (`docs/deepdives/2026-07-17_pass-a-score-signal-and-exploitability.md`,
> computations in `notebooks/pass_a_v3_score_signal_retroactive.ipynb`) found the
> score works as a filter (persistence rho +0.66; 82–87% reversion hit-rate) but
> the harvested dollar-neutral edge (median ~13 bps, mean ≤ ~9 bps gross per
> round trip) does not clear realistic costs — and the simulator models zero
> slippage.
>
> **2026-07-18:** the exit redesign was retro-tested and **rejected** — the mean
> ≪ median gap is concentrated catastrophic outliers (worst-3 pairs = 45–80% of
> all losses per regime), not held-too-long losers, and no stop recovers it.
> Median is positive in every regime; removing the ~8% catastrophic pairs makes
> gross mean +22 to +45. Studies now resume after: (1) **determine whether the
> catastrophic pairs are identifiable at entry** (new lead question) → an entry
> screen if so, position sizing if not; (2) entry magnitude floor (H-C); (3)
> simulator cost model (H-D); (4) Pass A re-gated with the v4 full-pool P&L-free
> objective (H-A). Exit-mechanics work is closed. See the deepdive's Update
> section.

> **Restructure (2026-07-23) — Pass A becomes a cache-based scoring-quality
> study; two-study split.** The 2026-07-22 investigation (deepdive Update
> 2026-07-22; plan `2026-07-22_pass-a-v4-scoring-study.md`) showed the old Pass A
> objective was **70% noise**: it was indifferent to 4 of 5 weights, so the
> "best" weights were a lucky draw (40% on a dead half-life component, 0.2% on
> `corr_short`). Pass A is rebuilt as a **P&L-free scoring-quality study** that
> scores the *full* candidate pool (removing the range restriction that crushed
> the old gate) and optimizes for a **tail-sensitive top-K forward-P&L metric**
> (below) — no backtests, seconds per trial. This cleanly optimizes only the
> **ranking** params (composite weights + `corr_long/short_window`).
>
> The params the cache **cannot** judge move to a **re-scoped Pass B** (backtest):
> `zscore_window` (defines the outcome → target leakage if tuned on the cache),
> `cooldown_days` (discovery timing, no per-pair scoring effect), and
> `lookback_window` (a cache-build dimension — bounds the component window). This
> collapses the earlier "1b" idea into Pass B: everything a backtest must judge
> lives in one backtest study, co-tuned so their interactions are captured. The
> two studies are coupled (Pass A's cache is built at a fixed
> `zscore_window`/`lookback`; a material Pass B change there means refreshing the
> Pass A cache — coordinate descent). WS1 (log half-life) and the WS2/WS3 harness
> are done; WS4 cost model + Pass B objective redesign remain.
>
> **Conclusion (Pass A v4 = NO-GO):** the WS3 scoring study ran and, under out-of-fold
> validation at the live-faithful lookback, **reweighting the composite does not
> generalize** — held-out objective negative in all three folds, top-K ≈ random, pair-level
> component→outcome Spearman ≈ 0. The earlier +52 was overfit (non-live-faithful 400-day
> cache, no OOF) and is retracted. See the ledger row and the v4 spec. The composite stays
> a *filter*, not a fine-grained *ranker*; **WS1 stands as a valid standalone fix**, and
> WS4/WS5 are moot for the reweighting question.

---

## Objective functions

Two objectives, both implemented in `tuning/objective.py` via
`discriminatory_weight`:

**Scoring-quality** — Study 1 Pass A **v4** (cache harness). Computed on the full-pool
replay cache, no backtest. **NOTE: this objective was run and returned a NO-GO** — under
out-of-fold validation at the live-faithful lookback it does not resolve weights (held-out
negative in all three folds; see the ledger row and the v4 spec below). Documented here as
the objective that was *tried*, not one that succeeded:

```
fold_metric = mean over the fold's scoring dates of
              [ mean forward_gross of the top-K (K=20) pairs by composite score ]
objective   = 0.5 · mean_over_folds(fold_metric) + 0.5 · min_over_folds(fold_metric)
```

Tail-sensitive by construction: it is a *mean* (not a rank correlation) over the
*top-K selection*, so a catastrophic pair the weights admit to the top-20 directly
craters that fold. The `0.5·min` term is the per-fold floor (LORO discipline). The
in-sample number looked good but did **not** survive out-of-fold validation — the failure
that motivated `tuning/overfit_guards.py`.

**Blended discriminatory** (`discriminatory_weight=0.7`) — **superseded for Pass A**
(kept for historical reference; the old v1–v3 runs used it):

```
score = 0.7 × Spearman_rho(composite_score_at_entry, round_trip_pnl)
      + 0.3 × clip(sharpe / 3, −1, 1)

subject to: mean(round_trip_pnl) > pnl_floor (−100)
            n_round_trips ≥ 10
```

The intent was to measure discrimination directly, but the `Spearman_rho` term was
≈ 0 (range-restricted to entered pairs, n 30–60, se 0.13–0.19), so 70% of the
objective was noise and the weights it selected were arbitrary. The v4 scoring
objective fixes this by scoring the full pool P&L-free.

**Pure Sharpe** (`discriminatory_weight=0.0`) — everything else:

```
score = annualized_sharpe − 0.5 × max_drawdown_pct − 0.1 × trade_count_penalty
```

Pass B, Study 2, and Study 3 all take a calibrated score as given; their
question is how best to exploit it, and risk-adjusted return is the right
measure for that.

---

## Study specifications

### Study 1 Pass A v4 — scoring quality (cache-based, ranking params only)

Cache harness: `tuning/studies/scoring_replay.py` (builds the full-pool observation cache) +
`tuning/studies/scoring_study.py` (Optuna over the cache). No backtests, no cloud fleet —
runs in seconds locally. Plan: `docs/plans/2026-07-22_pass-a-v4-scoring-study.md`.

| Setting | Value |
|---|---|
| Free params (ranking only) | `w_corr_long`, `w_corr_short`, `w_coint`, `w_halflife`, `corr_long_window`, `corr_short_window`. (`w_z_depth` excluded — `z_depth`=1 for all tradeable pairs, inert here; its live weight is untouched.) |
| Frozen (outcome contract) | `zscore_window`, exit rule, hedge/coint lookback, `COINT_CEIL`, `MAX_HALFLIFE` — tuning any would leak into the target. `lookback_window` frozen at the live value (bounds the component window; caps `corr_long_window`). |
| Pool | Full candidate pool reconstructed via the strategy's own clusterer at 4 sampled dates/fold (clustering params fixed) |
| Outcome | Per-pair **gross forward spread P&L** under the frozen exit, dislocated (`|z|≥2`) pairs only |
| Folds | Same three 5-month windows as v2 |
| Objective | Scoring-quality (above) |

**Half-life fix (WS1, done):** `score_halflife` was a near-constant 0.96 (linear
`1 - hl/60` over 1–3 day half-lives) → replaced by a log-spaced `halflife_to_score`
so it varies. Without this, no weight over half-life is interpretable.

**Outcome — NO-GO (reweighting does not generalize).** The study *was* the evaluation,
and it returned a negative verdict. The initial +37→+52 in-sample number was produced on
a **non-live-faithful 400-day cache without out-of-fold validation**; under the guard
panel (`tuning/overfit_guards.py`) at the live-faithful lookback (152) the result
collapses: pair-level `Spearman(component, forward_gross)` ≈ 0 (well-powered, n
572–946/fold), top-K-by-score ≈ random-K, and the leave-one-fold-out held-out objective
is **negative in all three folds**. The +52 / `w_corr_short`→0.2 is **retracted as
overfit**. Re-confirms the deep dive: the composite is a good filter, a poor fine-grained
ranker — **composite reweighting is not a lever**, and WS4/WS5 downstream of it are moot.
WS1 (log half-life) stands as a valid standalone fix. WS5 (soft floor) was separately
tested on the cache and **rejected**.

### Study 1 Pass A v2 — signal construction (timescales + weights)  *(superseded by v4)*

Everything below is already implemented in `tuning/studies/study1_pass_a.py`.

| Setting | Value |
|---|---|
| Free params (10, all Tier 2) | `lookback_window` [60, 252], `zscore_window` [10, min(40, lookback//3)], `cooldown_days` [max(3, zscore//2), 21], `w_corr_long`, `w_corr_short`, `w_z_depth`, `w_halflife` [0, 1], `w_coint` [0.1, 1], `corr_long_window` [45, 252], `corr_short_window` [10, min(40, corr_long)] |
| Fixed params | All Tier 1 and Tier 3 at current defaults (`short_leg_fraction` = 0.0) |
| Folds (5-month, round-robin) | 2021-12→2022-04 (sideways/bear) · 2023-02→2023-06 (bull) · 2023-07→2023-11 (mixed) |
| Trials | 90 (~30/fold) |
| Objective | Blended discriminatory (0.7) |
| Per-trial timeout | 3 h (`trial_timeout_secs=10800`) |
| Weights | Normalised to sum 1.0 via `normalize_weights()` |

**Gate:** best-trial Spearman rho > 0.15 in ≥ 2 of 3 folds, evaluated by
re-running best-trial params on each fold after the study completes
(`_run_gate()` in the script). If the gate fails but the blended objective
improved over defaults, widen to 300 trials before concluding the score
structure is broken.

### Study 1 Pass B — discovery + portfolio construction

Implemented in `tuning/studies/study1_pass_b.py` — same fold-rotating
pattern as Pass A, same three 5-month folds. Because **Pass A v4 was a NO-GO**
(composite reweighting does not generalize), Pass B **keeps the composite weights +
corr windows at their existing live defaults** — the score's *filter* role is intact;
only its fine-grained *ranking* weights were shown to be un-tunable. There is no tuned
"Pass A v4 result" to load.

**Re-scoped (2026-07-23):** Pass B now also absorbs the three params the cache
study cannot judge — `zscore_window`, `cooldown_days`, `lookback_window` — on top
of its discovery/sizing set. These are backtest-only (they define the outcome,
entry timing, and component window respectively); co-tuning them with sizing in one
backtest captures their interactions. This collapses the interim "1b" idea into
Pass B.

| Setting | Value |
|---|---|
| Free params (signal) | `zscore_window`, `lookback_window` — moved from old Pass A |
| Free params (discovery/timing) | `cooldown_days` (moved from old Pass A), `max_daily_candidates` [50, 300], `cluster_recompute_days` [14, 90], `hdbscan_min_samples` [1, 5], `hdbscan_cluster_selection_epsilon` [0, 0.5], `min_intra_cluster_corr` [0.1, 0.6] |
| Free params (sizing) | `target_deployed_pct` [0.4, 0.9], `max_k` [5, 50], `max_halflife_days` [20, 120] |
| Fixed params | Composite ranking weights + corr windows at live defaults (Pass A v4 NO-GO — no retuned set); Tier 1/Tier 3 defaults |
| Folds | Same three 5-month folds as Pass A |
| Objective | **Post-cost** Sharpe/expectancy — **NOT** the old blended rho (needs the WS4 cost model + a redesign that avoids the noise-objective trap) |

**Blocked on:** the WS4 cost model and the objective redesign above. **Coupling:** if
Pass B's tuned `zscore_window`/`lookback_window` move materially from the values the
Pass A cache was built at, rebuild the Pass A cache and re-run it (coordinate descent).

**Gate:** none — output feeds Study 2 as `base_params`.

### Study 2 — per-regime Tier 3 optimization

For each regime label, an independent Optuna study finds the jointly optimal
Tier 3 set, with Tier 1/2 fixed at Study 1 outputs.

| Setting | Value |
|---|---|
| Free params (6, all Tier 3) | `short_leg_fraction` [0.0, 1.0], `entry_threshold` [1.2, 3.0], `exit_threshold` (suggested as fraction of entry, enforcing exit < entry), `min_position_pct` [0.01, 0.08], `max_position_pct` [0.05, 0.30], `quality_scale_pivot` [0.4, 1.0] |
| Fixed params | Study 1 Pass A + Pass B outputs; Tier 1 defaults |
| Folds | 9 × 3-month folds (table below), each tagged with its regime label |
| Trials | 20 per fold, 180 total |
| Objective | Pure Sharpe |
| Structure | One Optuna study per regime label, so TPE builds independent per-regime distributions |

Fold set (from the superseded 05-19 plan, plus a 2020 stress fold that the
PR #44 price backfill made possible — the old 2022+ restriction existed only
because the local cache started 2022-01-24):

| Fold | Window | Indicative regime (confirm with detector v2) |
|---|---|---|
| `covid_2020` | 2020-02-03 → 2020-05-29 | vol_shock / stress |
| `sideways_2022` | 2022-02-01 → 2022-04-30 | sideways |
| `bear_2022_q2` | 2022-06-01 → 2022-08-31 | vol_shock |
| `volatile_2022_q3` | 2022-09-01 → 2022-11-30 | vol_shock |
| `sideways_2023_q1` | 2023-01-01 → 2023-03-31 | sideways |
| `bull_2023` | 2023-04-01 → 2023-06-30 | trend_bull |
| `recovery_2023_q3` | 2023-07-01 → 2023-09-29 | calm_bull |
| `mixed_2023_q4` | 2023-09-01 → 2023-11-30 | sideways |
| `bull_2024_q1` | 2024-01-02 → 2024-03-29 | trend_bull |

**`covid_2020` feasibility caveat:** the price cache starts 2019-08-05, giving
this fold only ~124 trading days of pre-window history. It is viable only if
Study 1's best `lookback_window` / `corr_long_window` / `cluster_lookback_days`
come out ≤ ~120 trading days (Alpaca's history ceiling means the cache cannot
be extended earlier). Check after Pass A completes; if the tuned windows are
longer, shift the fold to 2020-03-02 → 2020-06-30 (~+20 days of headroom) or
drop it and accept vol_shock coverage from the 2022 stress folds only.

**Regime labelling — detector v2 required (decision 2026-07-13):** the
existing window-based classifier in `tuning/regime_detector.py` is **not
usable for Study 2**. An empirical check against the cloud DB (2026-07-13)
produced a degenerate partition of the folds above: zero folds labelled
`vol_shock` (its 0.30 vol threshold was calibrated on COVID's 0.55 and misses
the entire 2022 bear at 0.22–0.26), one fold `trend_bull`, and
`recovery_2023_q3` labelled `calm_bull` despite a −3.1% window return. It
also computes features with hindsight over the completed window — a shape of
computation with no live equivalent.

Replace it with the **daily, backward-looking, FRED-based detector** specified
in `2026-05-18_tier3-regime-detector.md` Step 2 (which remains the
authoritative spec for the detector itself):

- Features per date, all trailing/backward-looking: VIX level + 1m change,
  SPY 20d realized vol, SPY 50/200d trend, SPY 20d return, FRED `T10Y2Y`,
  FRED `BAMLH0A0HYM2` (HY spread), cross-sectional dispersion.
- **Publication-lag safety**: use as-of-lagged (T−1) values for FRED series
  so backtest labels contain no information the live detector wouldn't have
  at 9:28am.
- Rule-based labels first; **stability gate**: < 1 label flip/week on average
  across 2020–2024 daily history (3-day minimum-hold smoothing if needed).
  Recalibrate thresholds against the known failure: the 2022 folds must not
  collapse into `sideways` (mid-2022 HY spread ~5.0 > the 4.5 stress
  threshold already handles this).
- Fold tagging: label every trading day in a fold, tag the fold with its
  dominant label — the identical code path then serves live classification
  in paper/live mode. This is the live-parity property the window classifier
  lacked.
- Macro series cached in the cloud DB (extend the nightly refresh cron) so
  EC2 workers never call FRED during trials.

This work is **not on the Study 1 critical path** — build it while Pass A/B
run. It blocks Study 2 only.

**Gate:** regime-conditioned params beat the static Study 1 best-trial set
(run through the same 9 folds as comparator, ~9 extra backtests) in ≥ 6 of 9
folds (the original plan's 2/3 proportion), and in aggregate Sharpe at
p < 0.10 (permutation test). If the gate fails *after* the detector v2
recalibration, feature or label poverty is the leading suspect — revisit
detector features or collapse to a 2-label risk-on/risk-off scheme before
spending more compute.

### Study 3 — dense walk-forward + 2025 holdout

Launch only if the Study 2 gate passes.

| Setting | Value |
|---|---|
| Folds | The 9 Study 2 folds + 4 × 2024–2025 folds (13 total) |
| Trials | 25 per fold, 325 total |
| Holdout | 2025-01 → 2025-12, split into 4 quarters — **never used in any training fold** |
| Objective | Pure Sharpe |

The cloud DB's price cache covers 2019-08 → 2025-12 (9.3M rows, backfilled in
PR #44), so the 2024–2025 folds and the 2025 holdout need no data work — the
old "warm up 2024–2025 prices first" precondition is already satisfied.

**Gate:** positive Sharpe on the 2025 holdout in ≥ 3 of 4 quarters.
SPY-beating is *not* required here — that gate belongs to paper trading.

---

## Overfitting gates

Moved to the **`optuna-study` skill** (`.claude/skills/optuna-study.md` /
`.cursor/skills/optuna-study/SKILL.md`), now the single source of truth for the
anti-overfitting methodology, the guard panel, the default thresholds, and the
pre-launch checklist. Mandatory for **every** tuning study before any result is
declared or committed.

---

## Cloud execution

Operational steps (launching instances, security groups, tunnels) live in
`2026-05-21_cloud-infrastructure-plan.md` § Running Tuning Studies. This
section covers only what the studies themselves require.

### Measured trial times (basis for all estimates)

From completed studies (local Mac, warm price cache):

| Study | Machine | Fold length | Avg/trial | Max/trial |
|---|---|---|---|---|
| Study 0 (both folds) | local Mac | 3 months | 75–81 min | 138 min |
| Study 1 Pass A v1 | local Mac | 3 months | 57 min | 589 min (pre-timeout outlier) |
| Study 00 pipe check (7 trials) | c6i.4xlarge | 5 months | ~57 min (range 31–95) | 95 min |

Working assumptions (calibrated by Study 00): **~60 min/trial for 5-month
folds on c6i**, **~40 min/trial for 3-month folds**, with the 3-hour timeout
capping pathological trials. Variance is driven by suggested params
(lookback/candidate volume), not load — times were similar solo vs. 3-way
concurrent.

### Instance sizing and wall-clock

Parallelism is capped by TPE quality, not by budget: with ~10–15 workers on a
90-trial study, early suggestions are made nearly blind. Keep
workers ≲ trials/8.

| Study | Trials | Instance | Workers | Est. worker-hours | Est. wall-clock | Est. spot cost |
|---|---|---|---|---|---|---|
| 1 Pass A v2 | 90 × ~60 min | c6i.4xlarge (16 vCPU) | 12 | ~90 | **~8 h** (overnight) | ~$2 |
| 1 Pass B | 75 × ~60 min | c6i.4xlarge | 8 | ~75 | **~10 h** | ~$3 |
| 2 | 180 × ~40 min | c6i.4xlarge | 12 (3 per regime study, 4 studies concurrently) | ~120 | **~10 h** | ~$3 |
| 3 | 325 × ~40 min + holdout runs | c6i.4xlarge | 14 | ~220 | **~16 h** (weekend launch) | ~$4 |

Total tuning compute for the full sequence: roughly $16 in spot, on top of
the ~$47/month always-on DB instance. Terminate (don't stop) the tuning
instance after each study — results are already in the cloud DB; there is no
retrieval step.

### Pipe check before the first study (Study 00)

The infra had never carried a tuning study end-to-end — PR #44 validated data
migration, not the worker → cloud-DB write path. Before Pass A, run a smoke
study on the freshly launched tuning instance. Env overrides on the real
Pass A runner (in practice a thin driver reusing the Pass A objective was
used instead, to skip the post-study gate check — with `TUNE_N_TRIALS=1` the
gate would add 3 full backtests):

```bash
# 1. One solo trial — full chain: VPC → DB private IP, cached prices
#    (no Alpaca storm), coint cache writes, one COMPLETE trial.
TUNE_STUDY_NAME=cloud_smoke_v1 TUNE_N_TRIALS=1 \
  RUN_MODE=backtest python3.12 -m tuning.studies.study1_pass_a

# 2. Then 3 workers × 1 trial — fold rotation covers all three folds
#    (this IS the per-fold warm-up), tests concurrent Optuna RDB access,
#    and yields a per-fold EC2 timing sample. Watch the DB instance's
#    CPUCreditBalance while these run.
for i in 1 2 3; do
  TUNE_STUDY_NAME=cloud_smoke_v1 TUNE_N_TRIALS=1 \
    RUN_MODE=backtest python3.12 -m tuning.studies.study1_pass_a \
    >> /tmp/smoke_w${i}.log 2>&1 &
done
```

Verify from the laptop through the tunnel (this also smoke-tests the
monitoring path):

```sql
-- All smoke trials COMPLETE, with values and params
SELECT t.trial_id, t.state, v.value
FROM studies s JOIN trials t USING (study_id)
LEFT JOIN trial_values v USING (trial_id)
WHERE s.study_name = 'cloud_smoke_v1';

-- Runs closed out, settings self-describing (incl. short_leg_fraction)
SELECT run_id, completed_at IS NOT NULL AS done,
       settings ? 'short_leg_fraction' AS has_slf
FROM backtest_runs ORDER BY started_at DESC LIMIT 4;

-- Snapshots and trades landed for the newest run
SELECT (SELECT COUNT(*) FROM portfolio_snapshots WHERE run_id = r.run_id) AS snaps,
       (SELECT COUNT(*) FROM trades WHERE run_id = r.run_id) AS fills
FROM (SELECT run_id FROM backtest_runs ORDER BY started_at DESC LIMIT 1) r;
```

Pass criteria: all trials COMPLETE; `backtest_runs.completed_at` set;
snapshots > 0 and fills > 0 for each run; `pair_coint_cache` row count grew;
no Alpaca rate-limit errors in worker logs. **If it passes, keep the instance
and launch the Pass A fleet on it directly** — the environment is warm and
validated. Record the measured per-trial times in the timing table above.
Cost: ~4 h mostly unattended, ~$1. The smoke rows in `backtest_runs` are
ordinary runs under the scratch study name; no cleanup needed.

### Cache warm-up (revised meaning)

The old warm-up step existed to stop 8 workers from hammering Alpaca for
uncached prices. **Prices are no longer the issue** — the PR #44 backfill
cached the full 2019–2025 history, and since the 2026-07-14 incident fix,
**tuning trials run price-cache-only** (`price_cache_only=True` injected by
`BacktestObjective`): they never call Alpaca at all. Symbols without DB rows
for a window are simply absent from that run. New price data enters the cache
only via the nightly refresh cron or an explicit backfill — never via trials. What still benefits from warming is
`pair_coint_cache` (keyed by `lookback_window` + `window_end_date`):

- **Pass A**: warm-up has limited value — `lookback_window` is a free
  parameter, so most trials miss the cache regardless. Study 00 (pipe check
  above) already covers the per-fold environment shakeout; no separate
  warm-up step needed.
- **Pass B / Study 2 / Study 3**: `lookback_window` is fixed, so one warm-up
  trial per fold populates the coint cache for every subsequent trial on that
  fold. Do this — it is the difference between ~100 min and warm-trial times
  for all remaining trials on the fold.

### DB instance load — watch item

All workers share the t3.medium DB for Optuna storage *and* price reads.
Optuna RDB traffic is negligible, but 12 concurrent backtests reading price
history is untested against a burstable instance. During the first parallel
study, watch CPU credit balance (CloudWatch `CPUCreditBalance`) and query
latency; if the DB becomes the bottleneck, resize to t3.large for the study
window or stagger worker starts.

### Conventions

- **Study naming**: `study<N>_<qualifier>_v<K>`; bump `v` on any design
  change that invalidates comparability (as with Pass A v1 → v2). Never
  reuse a name with different semantics.
- **Worker safety**: all runners use `load_if_exists=True`; workers can be
  added or restarted at any time, and a spot interruption loses at most the
  in-flight trials.
- **Compute receipts**: the `tuning_studies` table (wall-clock, trial counts,
  parallelism, cost) exists but is currently unpopulated. Have each study
  runner write one receipt row on completion — after a few studies this gives
  a measured marginal-Sharpe-per-trial curve for future sizing decisions.
- **Monitoring**: from the laptop via the DB tunnel
  (`ssh -f -N -L 5433:localhost:5432 lumibob-db`), then query the `trials`
  table or `optuna.load_study` with the study names from the ledger above.
  Note the monitoring snippets in older docs reference `study1_pass_a_v1` —
  the live name is `study1_pass_a_v2`.

---

## Prerequisites before the next launch

1. ~~Write `tuning/studies/study1_pass_b.py`~~ — **done** (PR #46).
2. ~~Point local `.env` `DB_URL` at the tunnel~~ — **done** (2026-07-14);
   local analysis sessions now hit the cloud DB via port 5433.
3. **Build regime detector v2** (before Study 2, not before Pass A): daily,
   backward-looking, FRED-based, per `2026-05-18_tier3-regime-detector.md`
   Step 2 — with publication-lag-safe features, the < 1 flip/week stability
   gate over 2020–2024, macro series cached in the cloud DB via the nightly
   refresh cron, and thresholds recalibrated so the 2022 folds don't collapse
   into one label (the failure observed 2026-07-13 with the window-based
   classifier). Then tag every Study 2 fold with its dominant daily label.
4. **Decide on compute receipts** (convention above) — cheap to add to the
   Pass A runner before its launch so the sequence is instrumented from the
   start.
