---
name: optuna-study
description: LumiBob Optuna tuning-study design + anti-overfitting methodology and the source of truth for the overfitting gates. Use when designing, launching, or reviewing any Optuna tuning study — scoring-quality, backtest, or cache/analytical studies.
---

# Optuna Study Design

This skill is the **anti-overfitting methodology and source of truth for the overfitting
gates** for LumiBob Optuna tuning studies. It exists because Pass A v4 reported a great
in-sample objective (+52) that was pure overfitting, and every guard below is a symptom of
that failure made impossible to miss.

Operational detail (study naming, worker safety, cloud sizing, coint-cache warm-up,
compute receipts) lives in `docs/plans/2026-07-12_cloud-tuning-studies.md`. This skill
covers *how to make a study honest*; that doc covers *how to run it*.

---

## Core principle — optimize an out-of-sample estimate, never an in-sample fit

Optuna must maximize an **out-of-sample / generalization** estimate of the objective.
The in-sample number is allowed **only as a diagnostic** — to compute the
in-sample-vs-held-out gap — and may **never** be the headline or the decision basis.

The Pass A v4 post-mortem is the cautionary tale: 6 params were optimized against an
objective aggregated over only **3 regime folds**; the in-sample optimum looked great
(+52), but leave-one-fold-out held-out scores were negative, top-K-by-score was
indistinguishable from random, the raw components had ~0 rank-correlation with the
outcome at the pair level, and the winning weights swung wildly across seeds. The
objective was ~70% noise, so the "best" weights were a lucky draw.

Deterministic, reusable guards live in `tuning/overfit_guards.py` (pure, seeded, no DB)
with tests in `tests/test_overfit_guards.py`. Use them — do not re-implement.

---

## 1. Declare the independent evaluation unit up front

Before tuning, state **in the study spec** what the independent unit is, and **evaluate
signal at that level**. For a scoring-quality study the unit is the scored
*pair-observation* — not the fold, not the date. Pass A v4's fatal error was evaluating
at the fold level (3 units) when the independent units were the hundreds of scored pairs.

Report the a-priori design check `complexity_ratio(n_free_params, n_independent_units)`:

- **red** — free params ≥ units (`ratio ≤ 1`): the fit can memorize; any in-sample
  optimum is meaningless. **Hard stop.**
- **amber** — within 3× (`1 < ratio < 3`): thin; expect fragile optima.
- **green** — `ratio ≥ 3`.

Pass A v4 was red at the fold level (6 params / 3 folds); at the pair level it is green.
A red or amber ratio is a design smell to fix *before* spending compute, not after.

---

## 2. Unit-level signal is the MANDATORY FIRST gate

Before tuning **anything**, run `unit_level_signal(df, component_cols, outcome_col,
group_col)` and its `all_null(...)` convenience from `tuning/overfit_guards.py`. This asks,
per group, across all its rows (the true independent units), whether each **raw
component** ranks the outcome at all, with a bootstrap CI.

If every component's CI includes 0 in every group (`all_null(...)` is `True`), then no
linear weighting of those components can rank the outcome — **STOP**, and fix the
components or the outcome definition before spending compute on weights. This is the check
that would have killed Pass A v4 on day one: if the components can't rank the outcome at
the unit level, no objective over their weights can either.

---

## 3. Temporal split — walk-forward, NOT leave-one-fold-out

Eval/test windows must be strictly **after** their train windows in calendar time, with
an **embargo gap** between them. Leave-one-fold-out (LOFO) is insufficient: it can place a
"held-out" fold chronologically *before* its training folds and leak future→past
information (autocorrelation, overlapping forward-outcome horizons, publication lag).

Use the shared primitive `walk_forward_splits(start, end, train_span, eval_span, embargo,
step)` in `tuning/overfit_guards.py`, which generates rolling `(train, eval)` window pairs
and passes every schedule through `assert_causal(...)` — a hard raise if any eval window
overlaps, abuts, or precedes `train_end + embargo`, or if the embargo is zero/negative.
The embargo must be a strictly positive gap.

---

## 4. Structure for parameter studies

A parameter study has **no model-fitting step**, so the classic "train / validation /
test" partly collapses. The honest structure is:

- an **optimization set** Optuna maximizes over — made robust via **many independent
  units** and/or a **rolled walk-forward CV** so no single window is overfit; plus
- an **untouched final test window**, scored **exactly once** for the honest number.

The optimization set is where TPE searches; the final test window is spent once, at the
end, and its number is what you report. If you find yourself re-scoring the final window
to "check" a tweak, it is no longer a test window — you have contaminated it.

---

## 5. Guard panel — mandatory before declaring results

Run the full guard panel and quote its **held-out** numbers, never the in-sample best.
Use `report_panel(...)` for the compact red/green summary.

| Guard | Function | Default pass threshold |
|---|---|---|
| Null baseline | `null_baseline(...)` | real selection ≥ **90th percentile** of the random-k and outcome-shuffled nulls (z ≳ 2); a score near the 50th percentile is no better than random |
| Held-out gap | `holdout_gap(...)` | `mean_holdout` **> 0** and the train−holdout gap ≤ its magnitude; a negative held-out with a large positive gap is textbook overfitting |
| Seed stability | `seed_stability(...)` | winning params' across-seed **range within ~20% of their search span**; flag any key exceeding the threshold. Wildly swinging weights ⇒ a noise-dominated objective |

Do not declare a study's gate passed, and do not feed its output downstream, until the
panel is green (or ambers are explicitly justified in the study's write-up / ledger row).

---

## 6. Report OOS, never in-sample best

Every study's write-up quotes the leave-one-unit-group-out held-out number from
`holdout_gap`, not the best in-sample objective. A result is **not "done"** because the
in-sample optimum looks good. If the write-up leads with the in-sample number, it is
wrong — fix the headline.

---

## 7. Fragility smell-test

Re-run the winning config under a **reasonable data perturbation** — drop one fold/date,
jitter the sampled dates, or re-seed the study. If the selected config flips (weights
reorder, a param moves across its range), treat it as **fragile**: do not commit it, and
prefer the simpler / lower-complexity configuration.

`seed_stability(...)` mechanizes the re-seed variant; the fold-drop variant is
`holdout_gap(...)` read qualitatively (does the winner change per held-out fold?). Pass A
v4 failed this on both axes — its winning weights had a near-opposite runner-up at a
statistically tied objective.

---

## 8. Per-study-type instantiation

The evaluation **principle** is shared; the **implementation** is per-study.

- **Backtest-based studies** score a param by **portfolio Sharpe over a rolled
  walk-forward window** (many independent time-units). The unit is the fold/window;
  watch the complexity ratio closely — folds are few, so keep free params low and lean on
  the walk-forward rolls for independent units.
- **Cache / analytical studies** (e.g. the Pass A v4 scoring harness) score at the **pair
  level over the cached rows** — thousands of independent units, seconds per trial. The
  unit is the pair-observation; `unit_level_signal` and `null_baseline` apply directly.

Pick the unit that matches where the independent information actually is, then apply the
same gates at that level.

---

## 9. Architecture

- **Shared primitives are reused code**: `tuning/overfit_guards.py` (the guard functions +
  `complexity_ratio` + `walk_forward_splits` / `assert_causal` causality-embargo helper).
  Import them; do not re-implement per study.
- **Eval orchestration is scripted per study.** Studies are heterogeneous (backtest fleet
  vs. cache replay vs. analytical) — do **not** force a single shared orchestration
  framework. Each study wires the shared primitives into its own driver.

Study scripts live in `tuning/studies/` (e.g. `study0_short_leg_fraction.py`,
`study1_pass_a.py`, `study1_pass_b.py`, `scoring_replay.py`, `scoring_study.py`). The
overfit-guard module stays at `tuning/overfit_guards.py` — it is a shared module, not a
study.

---

## Study launch checklist

Run this **before launching each study**. Every box must be checked (or an amber
explicitly justified in the ledger).

Design & signal:
- [ ] Declared the **independent evaluation unit** in the study spec, and signal is
      evaluated at that level.
- [ ] `complexity_ratio(n_free_params, n_independent_units)` is **not red** (amber only
      with a written justification).
- [ ] **Unit-level-signal gate passed**: `unit_level_signal(...)` + `all_null(...)` show
      at least one component with a CI excluding 0 — otherwise STOP.

Temporal integrity:
- [ ] **Temporal split defined** via `walk_forward_splits(...)`: every eval window is
      strictly after its train window + a positive **embargo** (verified by
      `assert_causal`). No leave-one-fold-out.
- [ ] Optuna optimizes the **rolled OOS objective**, not the in-sample fit.
- [ ] A **final test window is reserved** and will be scored **exactly once**.

Guards & reporting:
- [ ] Guard panel wired in: `holdout_gap`, `null_baseline`, `seed_stability` via
      `report_panel(...)`, with the default thresholds above.
- [ ] Reporting is **OOS** (held-out), not in-sample best.
- [ ] Fragility smell-test planned (re-seed via `seed_stability`; fold-drop via
      `holdout_gap`).

Operational conventions (detail in `docs/plans/2026-07-12_cloud-tuning-studies.md`):
- [ ] Study named `study<N>_<qualifier>_v<K>`; bump `v` on any design change that
      invalidates comparability. Never reuse a name with different semantics.
- [ ] Runner uses `load_if_exists=True` (workers can be added/restarted safely).
- [ ] `price_cache_only` set so trials never call Alpaca.
- [ ] For **fixed-lookback** studies, coint-cache warm-up done (one warm-up trial per fold
      populates `pair_coint_cache`). Skip for Pass-A-style studies where `lookback_window`
      is a free parameter (warm-up has no value there).
- [ ] Compute-receipt row written to `tuning_studies` on completion (wall-clock, trial
      counts, parallelism, cost).
