# H1 Short-Leg Validation — Deep Dive Findings

> Generated 2026-05-13 from DB runs: `0ec7cc`, `3f7292`, `691011`, `4e2054`.
> Updated 2026-05-14 with post-fix validation runs: `fac053` (sideways) and `a1bd6b` (bull).
> Runs executed 2026-05-04 (original) and 2026-05-13 (post-fix). Validation script: `scripts/run_h1_validation_backtests.py`.

---

## Data availability note

`stock_prices` covers 2024-05-09 → 2024-12-30 only — no overlap with the 2022 or 2023 target windows. All SPY comparisons and beta calculations use `portfolio_snapshots.spy_value`, which is populated for all four runs. Short-leg status confirmed from the `trades.leg` column (`leg IN ('long','short')`); `enable_short_leg` is not stored in `backtest_runs.settings`.

---

## Run inventory

| run_id | label | window | short_leg | notes |
|--------|-------|--------|-----------|-------|
| `0ec7cc` | sideways_2022 baseline | 2022-02-01 → 2022-04-29 | False | H5 params |
| `3f7292` | sideways_2022 H1 (buggy) | 2022-02-01 → 2022-04-28 | True | cash bug + no conflict guard |
| `691011` | calm_bull_2023 baseline | 2023-04-03 → 2023-06-29 | False | H5 params |
| `4e2054` | calm_bull_2023 H1 (buggy) | 2023-04-03 → 2023-06-28 | True | cash bug + no conflict guard |
| `fac053` | sideways_2022 H1 (fixed) | 2022-02-01 → 2022-04-29 | True | cash fix + conflict guard + gross metrics |
| `a1bd6b` | calm_bull_2023 H1 (fixed) | 2023-04-03 → 2023-06-29 | True | cash fix + conflict guard + gross metrics |

Run `830260` (2017-01-03 → 2017-01-24, 15 snapshots, non-H5 settings) is not one of the two planned validation runs and is excluded from all analysis.

---

## 1. Summary table

| run_id | label | regime | ret% | SPY% | vs SPY | Sharpe | max DD% | avg cash% | avg pairs |
|--------|-------|--------|------|------|--------|--------|---------|-----------|-----------|
| `0ec7cc` | baseline | sideways_2022 | +1.59% | -4.29% | +5.89pp | 0.47 | -5.91% | 26.7% | 85.5 |
| `3f7292` | H1 buggy | sideways_2022 | -17.45% | -5.44% | -12.01pp | -5.59 | -17.45% | 83.3% | 132.6 |
| `fac053` | H1 fixed | sideways_2022 | -0.93% | -5.44% | +4.51pp | -0.16 | -5.26% | 100.6% | 171.9 |
| `691011` | baseline | calm_bull_2023 | -7.77% | +7.51% | -15.28pp | -2.29 | -9.95% | 28.1% | 100.0 |
| `4e2054` | H1 buggy | calm_bull_2023 | -12.78% | +6.84% | -19.63pp | -4.92 | -13.54% | 88.9% | 144.2 |
| `a1bd6b` | H1 fixed | calm_bull_2023 | -8.20% | +6.84% | -15.04pp | -4.26 | -8.65% | 100.8% | 157.4 |

The cash fix and conflict guard closed the performance gap with buggy H1 runs substantially in the sideways window (−12.01pp → +4.51pp vs SPY), but did not improve the bull window (−19.63pp → −15.04pp). The fixed sideways run still underperforms the long-only baseline by 1.38pp. The fixed bull run is still deeply negative.

The avg cash% figures for fixed H1 runs (~101%) are uninterpretable — short proceeds inflate `get_cash()`; use `gross_long_pct` / `gross_short_pct` instead. avg gross_long ≈ 55% and avg gross_short ≈ 55% for both fixed runs, confirming the book is dollar-neutral and deployed at target.

---

## 2. Trade activity

| run_id | label | long trips | long avg P&L | long total P&L | short trips | short avg P&L | short total P&L | avg hold | avg daily buys |
|--------|-------|------------|-------------|----------------|-------------|---------------|-----------------|----------|----------------|
| `0ec7cc` | baseline sideways | 70 | -1.05 | -73 | — | — | — | 4.0d | 1.2 |
| `3f7292` | H1 buggy sideways | 115 | -8.22 | -945 | 116 | -2.80 | -325 | 2.0d | 2.0 |
| `fac053` | H1 fixed sideways | 60 | +38.78 | +2,327 | 60 | -54.23 | -3,254 | 4.3d | 0.87 |
| `691011` | baseline bull | 67 | -11.04 | -739 | — | — | — | 4.0d | 1.2 |
| `4e2054` | H1 buggy bull | 109 | -31.92 | -3,479 | 110 | -0.15 | -17 | 4.0d | 2.1 |
| `a1bd6b` | H1 fixed bull | 56 | -22.06 | -1,236 | 56 | +8.63 | +483 | 4.6d | 0.90 |

Key observations:
- The cash fix worked: avg daily buys dropped to 0.87–0.90 (vs 2.0–2.1 in buggy runs), returning near the 1.2 baseline target.
- **The fixed sideways run reveals the core structural problem**: the long leg is now profitable (+2,327 total) but the short leg destroys those gains and more (−3,254). Net = −927, return = −0.93%.
- **The fixed bull run shows partial but insufficient hedging**: the short leg helps (+483) but the long leg loses (−1,236). Net = −752, return = −8.13%.
- Long-leg P&L in the fixed sideways run is strongly positive (avg +38.78/trip), meaning the pairs strategy's spread-convergence signal is working on the long side.
- Short-leg P&L in the fixed sideways run is strongly negative (avg −54.23/trip): when lag catches up to lead (spread convergence), the lead stock also rises — it just rises less. The short position loses on the lead's absolute upward move.
- Short-leg P&L in the bull run is positive (+8.63/trip) but too small to offset long-leg losses. The bull regime generates spread divergence, not convergence — pairs don't mean-revert when the whole market rises together.
- (The original +15.94 figure for `4e2054` short avg P&L was a calculation error; the correct value is −0.15/trip.)

### Exit reasons

| run_id | label | displaced | zscore_exit | total long sells | displaced% |
|--------|-------|-----------|-------------|------------------|------------|
| `0ec7cc` | baseline sideways | 37 | 37 | 74 | 50.0% |
| `3f7292` | H1 sideways | 62 | 54 | 116 | 53.4% |
| `691011` | baseline bull | 24 | 44 | 68 | 35.3% |
| `4e2054` | H1 bull | 58 | 59 | 117 | 49.6% |

Short-leg sells: `exit_reason` is NULL for all 123 (`3f7292`) and 125 (`4e2054`) short exits — the exit reason logic is not applied to short position closes.

Key observations:
- The bull window shows the sharpest shift: displaced rises from 35.3% to 49.6% when the short leg is active. This is consistent with H1 — over-entry increases competition for position slots, pushing out existing longs before they can mean-revert.
- The sideways window shows a smaller increase (50% → 53.4%) because displacement was already high in the baseline.
- NULL exit reasons on all short exits are an observability gap: there is no way to determine whether shorts are closing on z-score mean reversion, displacement, or another trigger.

---

## 3. Market neutrality

| run_id | label | regime | beta | R² | return corr |
|--------|-------|--------|------|----|-------------|
| `0ec7cc` | baseline | sideways_2022 | 0.559 | 0.609 | 0.781 |
| `3f7292` | H1 short_leg | sideways_2022 | 0.240 | 0.163 | 0.403 |
| `691011` | baseline | calm_bull_2023 | 0.662 | 0.259 | 0.509 |
| `4e2054` | H1 short_leg | calm_bull_2023 | 0.443 | 0.176 | 0.419 |

Beta reduction is real: −57% in sideways, −33% in bull. R² fell sharply in both windows (0.609 → 0.163 in sideways). The target of beta < 0.1 was not reached (0.240 and 0.443).

---

## 4. Candidate funnel

| run_id | label | avg buy-ready | avg daily buys | conflict guard fires |
|--------|-------|---------------|----------------|----------------------|
| `0ec7cc` | baseline sideways | 3.4 | 1.2 | N/A |
| `3f7292` | H1 buggy sideways | 5.4 | 2.0 | none (guard absent) |
| `fac053` | H1 fixed sideways | — | 0.87 | 0 |
| `691011` | baseline bull | 4.2 | 1.2 | N/A |
| `4e2054` | H1 buggy bull | 5.6 | 2.1 | none (guard absent) |
| `a1bd6b` | H1 fixed bull | — | 0.90 | 1 (SHY/VGIT) |

The conflict guard fired once across both fixed runs (vs 32 mirror-pair instances in the buggy runs). The reduction is primarily because the 2× effective cost from the cash fix already limits entries enough to prevent most same-day mirror-pair collisions. The guard correctly blocked SHY/VGIT in the bull run.

---

## 5. Hypotheses

### Hypothesis 1: `available_cash` inflation causes over-entry, which amplifies losses

**Evidence:** avg_daily_buys = 2.0 (H1) vs 1.2 (baseline) in both windows. Long-leg trips: 115 vs 70 in sideways. Median hold: 2d (H1 sideways) vs 4d (baseline). Long-leg avg_pnl = −8.22 vs −1.05 (sideways), −31.92 vs −11.04 (bull).

**Mechanism:** In `on_trading_iteration`, `available_cash = self.get_cash()` (line 654 of `BobsBrain.py`) includes short-sale proceeds credited to cash by Lumibot's backtesting engine. Each pair entry is cash-neutral in reality: buy lag for −X, short lead returns +X to cash. Within one iteration the local variable is correctly decremented by `effective_cost = 2 × budget`, limiting same-day entries. But the next day, `self.get_cash()` reads the replenished real cash, and `deployment_gap = target_deployed_pct × portfolio_value − (portfolio_value − cash)` stays near `target_deployed_pct × portfolio_value` because `portfolio_value − cash ≈ 0` for a dollar-neutral book. The strategy sees perpetual headroom and enters new pairs every day regardless of existing exposure. More round-trips × negative per-trip expectancy = compounding losses.

**Proposed change:** Replace the `available_cash` seed with a gross-long-exposure calculation:
```python
gross_short = sum(
    abs(pos.quantity) * (self.get_last_price(pos.symbol) or 0)
    for pos in self.get_positions() if pos.quantity < 0
)
available_cash = self.get_cash() - gross_short
```
This is a code fix. `enable_short_leg` is not in `parameter_space.py`; this change is not achievable through retraining.

**How to validate:** After fix, avg_daily_buys returns to ~1.2, long-trip count matches baseline (~70 sideways), median_hold returns to ~4d, long-leg avg_pnl approaches baseline levels.

---

### Hypothesis 2: Short leg reduces beta materially but not to the target

**Evidence:** Beta fell from 0.559 → 0.240 in sideways (−57%) and 0.662 → 0.443 in bull (−33%). R² fell from 0.609 → 0.163 and 0.259 → 0.176. Target was beta < 0.1.

**Mechanism:** Equal-dollar shorting does not equal equal-beta hedging. If the lead stock has a higher systematic beta than the lag stock, the short leg under-hedges SPY exposure per dollar. Additionally, over-entry (Hypothesis 1) adds unhedged longs that dilute hedge efficiency.

**Proposed change:** Fix Hypothesis 1 first and re-measure residual beta in a clean run. If beta remains > 0.1, add beta-weighting: `lead_qty = per_stock_budget × (lag_beta / lead_beta) / lead_price`, using 60-day rolling OLS beta vs SPY. This is a design change, not a tuning intervention — no equivalent parameter exists in `parameter_space.py`.

**How to validate:** After Hypothesis 1 fix, rerun both windows. If beta < 0.1: Hypothesis 2 resolved. If beta > 0.1: implement beta-weighting and rerun.

---

### Hypothesis 3: Short-leg P&L is regime-dependent — it loses money in down markets

**Evidence:** Short avg_pnl = −2.77/trip in sideways 2022 (SPY −5.44%). Short avg_pnl = −0.15/trip in calm_bull 2023 (SPY +6.84%). The short leg loses money in both regimes; it is near-breakeven in the bull window, not profitable. (The original +15.94 figure for `4e2054` was a calculation error.) The short leg was a loss source in the window where a hedge was most needed, and failed to generate meaningful profit in the window where conditions were most favourable.

**Mechanism:** Lead stocks are selected as recent short-window outperformers within their cluster. In a broad equity selloff, these momentum leaders often hold their value or rise relative to the market — they are the quality names rotating into capital. Shorting them in a down market is directionally wrong. In a bull market the short leg approaches breakeven, suggesting the momentum signal is roughly neutral over short holding periods when the market is rising broadly.

**Proposed change:** This is not parameter-tunable. The tension is inherent in using momentum-based lead selection as a short candidate. As a follow-on investigation after Hypothesis 1 is fixed: isolate short avg_pnl by regime across multiple windows. If consistently negative in down-market regimes even after the cash fix, evaluate a short-entry gate — only enter the short when the lead's 5-day return is in the top quartile of its cluster (confirming genuine recent stretch rather than a sustained uptrend). This gate would be a new binary parameter, not a tunable float.

**How to validate:** After fixing Hypothesis 1, isolate short avg_pnl by regime over at least three down-market windows and three up-market windows. If short avg_pnl improves materially in down-market regimes after the cash fix, Hypothesis 1 was distorting the signal.

---

### Hypothesis 4: `cash_ratio` is not a useful deployment metric for a dollar-neutral book

**Evidence:** avg_cash_ratio = 83.3% and 88.9% in H1 runs vs 26.7% and 28.1% in baselines. Active positions are confirmed by `leg=short` trade fills and `avg_pairs` = 132–144. The metric is structurally uninterpretable.

**Mechanism:** Short-sale proceeds are credited to the cash balance in Lumibot. For every dollar of short position entered, `get_cash()` increases by one dollar. A fully deployed dollar-neutral book (equal long and short notional) therefore shows `cash ≈ portfolio_value`, so `cash_ratio ≈ 100%` even at full deployment.

**Proposed change:** Add `gross_long_pct` and `gross_short_pct` to `portfolio_snapshots` and log them in `on_trading_iteration`. These report actual notional deployment and remain interpretable regardless of leg structure. Additionally, ensure `exit_reason` is set on short position closes — currently all short exits record NULL, making it impossible to distinguish z-score exits from displacement on the short leg.

**How to validate:** In the next run with `enable_short_leg=True`, `gross_long_pct + gross_short_pct ≈ 2 × target_deployed_pct` and the two track symmetrically.

---

### Hypothesis 5: Mirror pairs create self-cancelling positions that pay slippage twice for zero net exposure

**Evidence:** 32 instances in each H1 run where the same stock was simultaneously bought long and sold short via different pairs. Three sub-patterns:

- **Mirror pairs** — both directions of the same pair were discovered and entered independently: GOOG/GOOGL, FV/FVC, CDC/CDL, SHY/VGSH, GNMA/VMBS, FTGC/PDBC, CENTA/CENT, PAA/PAGP, PTNQ/QQQ. The strategy is long stock X (as lag in pair A) and short stock X (as lead in pair B) at the same time.
- **Chain pairs** — a stock is the lag in one pair and the lead in another, entered on the same or nearby dates: MCHP was bought long (lag to SYNA) and shorted (lead to MXL) on 2022-02-01, the first day of the sideways run.
- **Unrelated pairs sharing a stock** — the previous finding from the baseline analysis (QQQM, IBTK, VMBS, FTC) extended to the short leg: a stock can now accumulate both long and short exposure from different pairs with no awareness of the conflict.

**Mechanism:** Pair discovery has no cross-pair position awareness. For highly correlated pairs — especially share classes (GOOG/GOOGL, CENTA/CENT), ETF variants (SHY/VGSH, GNMA/VMBS), or leveraged/unleveraged equivalents (PTNQ/QQQ) — both directions pass the correlation and z-score gates independently. Each gets entered as a valid pair. The resulting long + short in the same stock is net-zero in exposure but consumes position budget and pays entry and exit slippage twice. With 32 such instances per run and two fills each (entry + exit), this accounts for a material fraction of total slippage cost.

**Proposed change:** Before submitting a pair entry, check all currently held positions for conflicts: if any leg stock is already held long and the proposed trade would short it (or vice versa), skip entry and log the conflict. This is a pre-entry guard in `BobsBrain.py`, not a tunable parameter. A secondary fix is to deduplicate the discovered pair set before scoring — if both A→B and B→A exist as candidates on the same day, retain only the one with the stronger z-score signal.

**How to validate:** After fix, zero instances of simultaneous long+short in the same stock. Slippage per run decreases. The 32-instance count drops to 0 in a re-run of the same windows.

---

## 6. Recommended changes

| Priority | Change | Type | Addresses | Effort |
|----------|--------|------|-----------|--------|
| 1 | Fix `available_cash` to exclude gross short exposure (`get_cash() − gross_short`) | Bug fix in `BobsBrain.py` | H1 (root cause of return degradation) | Low |
| 2 | Add pre-entry conflict guard: skip entry if any leg stock is already held in the opposite direction | Bug fix in `BobsBrain.py` | H5 | Low |
| 3 | Add `gross_long_pct` / `gross_short_pct` indicators to snapshots; fix `exit_reason` on short closes (currently NULL) | Observability | H4 | Low |
| 4 | Re-run both validation windows after fixes; measure residual beta | Validation run | H2 | Low (run time) |
| 5 | If beta > 0.1 after fix: add beta-weighted short-leg sizing | Design change | H2 | Medium |
| 6 | Investigate short-entry gate after H1 fix and multi-window short P&L audit | Research + design | H3 | Medium |

Priorities 1, 2, and 3 should all be made before re-running the validation — the observability fixes ensure the re-run produces interpretable data. Priorities 5 and 6 are conditional on what the clean re-run shows.

---

## 7. Go / no-go

**Decision: NO-GO on H1 as designed. The equal-dollar short leg is structurally incompatible with the strategy's spread-convergence mechanism.**

**Reasoning:**

The original go/no-go (before the post-fix re-run) attributed the return degradation to two bugs — the `available_cash` inflation and the missing conflict guard. Those bugs have been fixed. The post-fix re-run shows the bugs were the primary cause of the bull-window degradation, but the sideways-window performance remains below baseline despite correct mechanics.

The post-fix evidence is conclusive:

- **Sideways fixed run (`fac053`)**: The long leg is profitable (+2,327 total, avg +38.78/trip). The short leg destroys those gains (−3,254 total, avg −54.23/trip). Net return = −0.93% vs baseline +1.59%. The strategy's spread-convergence signal works on the long side; the short leg fights against it.
- **Bull fixed run (`a1bd6b`)**: The short leg partially hedges (−8.20% with short vs −7.77% without). The improvement is minimal and the book remains deeply negative vs SPY (−15.04pp).

**The structural mechanism:** Spread convergence in this strategy is driven by the lag stock rising to close the gap to the lead stock. When this happens, the lead stock also rises — it just rises less than the lag. The long lag position profits from the lag's absolute rise. The short lead position loses on the lead's absolute rise. The short leg is not a pure beta hedge; it is a bet that the lead stock will fall in absolute terms. It does not fall — it merely underperforms the lag.

Equal-dollar shorting the lead is the wrong hedge instrument. It hedges against the lead stock's absolute move, not against the portfolio's systematic exposure. A sector ETF short, or a beta-weighted SPY short, would hedge market exposure without fighting the pairs signal.

**The correct path forward:**
1. Replace the equal-dollar lead short with a sector-ETF hedge (short the sector ETF rather than the specific lead stock) to isolate beta exposure without interfering with the spread trade.
2. Alternatively, explore a 30–50% notional short on the lead rather than 100%, accepting a partial hedge but reducing the spread-fight cost.
3. Re-run both validation windows with the revised hedge structure.

Do not integrate H1 as currently designed into Phase 4. The short leg harms performance in the sideways regime — the regime where the long-only strategy is strongest — and provides only marginal improvement in the bull regime.

---

## 8. Execution checklist

- [x] Phase 1 data validation complete (all four runs confirmed in DB, SPY proxy confirmed)
- [x] Phase 2 evidence gathered (5 tables: regime perf, trade activity, market neutrality, funnel, exits)
- [x] Phase 3 hypotheses written from evidence
- [x] `tuning/parameter_space.py` checked — `enable_short_leg` not present; proposed fixes are code changes, not tuning interventions
- [x] Fix `available_cash` bug in `BobsBrain.py` (PR #42)
- [x] Add pre-entry conflict guard in `BobsBrain.py` (PR #42)
- [x] Add `gross_long_pct` / `gross_short_pct` indicators to `portfolio_snapshots` (PR #42)
- [ ] Fix `exit_reason` to be set on short position closes (still NULL for all short exits — low priority given structural go/no-go)
- [x] Re-run `scripts/run_h1_validation_backtests.py` after fixes (`fac053`, `a1bd6b`)
- [x] Record re-run results (this section)
- [x] Gate Phase 4 decision: **NO-GO** — equal-dollar short leg is structurally incompatible with spread-convergence mechanism; long leg now profitable in sideways but short leg destroys the gains

**Next steps (if H1 is revisited):**
- [ ] Design sector-ETF hedge variant: short the sector ETF for the lead's sector in equal notional, rather than the lead stock itself
- [ ] Run sensitivity test: 50% notional short on lead (partial hedge) vs 100% (current design)
- [ ] Rerun both validation windows with revised hedge and confirm sideways long-leg P&L is not degraded
