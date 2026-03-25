---
name: backtest-agent
description: LumiBob backtesting workflow. Use when asked to run a backtest to evaluate a code change, e.g. "run a backtest for this PR", "backtest this change", "design a backtesting plan".
---

# Backtesting Agent

Four phases: understand the change → generate a plan → await approval → execute and hand off.

**Goal**: test the efficacy of a change as quickly as possible. Prefer shorter, targeted runs over long exhaustive ones. If a long test feels necessary, consider splitting into smaller runs sampled across different periods instead of a single long run.

---

## Phase 1 — Understand the Change

```bash
gh pr view <PR> --json title,body,files,additions,deletions
gh pr diff <PR>
```

Classify the change (drives all Phase 2 choices):

| Category | Examples |
|---|---|
| **Pair discovery** | Correlation logic, `min_correlation`, `lookback_window`, `min_daily_pairs`, `max_lag` |
| **Signal generation** | Z-score windows, entry/exit thresholds, holdout gate logic |
| **Position sizing / risk** | Cash allocation fraction, max positions, stop-loss logic |
| **Bug fix / data fix** | Correctness change; output should now differ from (or match) a baseline |
| **Infrastructure** | No behavioural change expected; timing or DB/cache layer only |

If intent is still unclear after reading the diff, ask up to **3** targeted questions before proceeding.

---

## Phase 2 — Generate a Backtesting Plan

Present the plan as structured text. Do not edit files yet.

### Date range by category

| Type | Duration | Use when |
|---|---|---|
| Smoke test | ~2 weeks | Infrastructure / correctness check |
| Standard validation | ~3 months | Bug fixes, signal generation changes |
| Full validation | ~6 months, spanning volatile + calm | Pair discovery or position sizing changes |
| Comparative | Two separate runs | Expected return profile change; want before/after |

### `ticker_limit` tier

- `50` — smoke test; fastest, least representative
- `100` — standard (default); good speed/coverage balance
- none — full universe; only for pair discovery changes needing broad coverage

### Signals to watch by category

- **Pair discovery** → `active_pairs`, `avg_correlation`, trade count, `candidates_found`, `candidates_buy_ready`
- **Signal generation** → win rate, avg P&L per pair, trade frequency, avg holding period
- **Position sizing / risk** → `cash_ratio`, max drawdown, total return, SPY comparison
- **Bug fix** → anomaly flags: unmatched sells, zero trades, unchanged portfolio value, `completed_at IS NULL`
- **Infrastructure** → `backtest_time_seconds`, output equivalence vs. known-good baseline

### Plan format

1. **Change summary** — what the PR does and why a backtest is warranted
2. **Classification** — category/categories
3. **Run configuration(s)** — exact dates, budget, `ticker_limit`, number of runs
4. **Signals to watch** — metrics and anomaly flags most relevant to this change
5. **Justification** — 1–2 sentences explaining the parameter choices

---

## Phase 3 — Await Approval

Stop after presenting the plan. Do not edit files or run commands until the user explicitly approves. Revise and re-present if the user requests changes.

---

## Phase 4 — Execute and Hand Off

For each run in the approved plan:

**Step 1 — Update `main.py`**

Locate the backtest block (the `backtesting_start` assignment and surrounding `BobsBrain.backtest(...)` call). Update `backtesting_start`, `backtesting_end`, `budget`, and `ticker_limit`. Do not change anything else.

**Step 2 — Run the backtest**

```bash
RUN_MODE=backtest python main.py
```

Wait for completion. If the run crashes or exits non-zero, report the error and stop.

**Step 3 — Retrieve the run ID**

```sql
SELECT run_id, started_at, completed_at
FROM backtest_runs
ORDER BY started_at DESC
LIMIT 1;
```

If `completed_at IS NULL`, flag as an anomaly — the run may have crashed silently.

**Step 4 — Hand off to evaluation**

Pass the `run_id` to the backtest evaluation workflow (`backtest-evaluation` rule). Follow that workflow Steps 1–5 in full. If multiple runs were planned, complete all before handing off and request a side-by-side comparison.
