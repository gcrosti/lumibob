# Backtesting Agent

Use when asked to run a backtest to evaluate a code change, e.g. "run a backtest for this PR", "backtest this change", "design a backtesting plan".

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
| **Pair discovery** | `TickerClusterer` / HDBSCAN, `max_daily_candidates`, `cooldown_days`, sector gate, `corr_long_window` / `corr_short_window`, composite weights |
| **Signal generation** | `zscore_window`, `entry_threshold`, `exit_threshold`, Z-score depth in scoring |
| **Position sizing / risk** | `min_position_pct`, `max_position_pct`, `target_deployed_pct`, dynamic K / portfolio construction |
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

### Run size (no `ticker_limit` in current code)

- **Shorter calendar window** (~2 weeks) — smoke / infrastructure
- **~3 months** — standard validation for most changes
- **Longer or multiple windows** — pair-discovery or regime-sensitive changes
- Tune **`max_daily_candidates`** (and dates) if you need heavier or lighter scoring load per day — there is no separate ticker cap in `main.py` today

### Signals to watch by category

- **Pair discovery** → `active_pairs`, `avg_correlation`, trade count, `candidates_found`, `candidates_buy_ready`
- **Signal generation** → win rate, avg P&L per pair, trade frequency, avg holding period
- **Position sizing / risk** → `cash_ratio`, max drawdown, total return, SPY comparison
- **Bug fix** → anomaly flags: unmatched sells, zero trades, unchanged portfolio value, `completed_at IS NULL`
- **Infrastructure** → `backtest_time_seconds`, output equivalence vs. known-good baseline

### Plan format

1. **Change summary** — what the PR does and why a backtest is warranted
2. **Classification** — category/categories
3. **Run configuration(s)** — exact dates, budget, relevant `parameters` keys (`max_daily_candidates`, correlation windows, weights, etc.), number of runs
4. **Signals to watch** — metrics and anomaly flags most relevant to this change
5. **Justification** — 1–2 sentences explaining the parameter choices

---

## Phase 3 — Await Approval

Stop after presenting the plan. Do not edit files or run commands until the user explicitly approves. Revise and re-present if the user requests changes.

---

## Phase 4 — Execute and Hand Off

For each run in the approved plan:

**Step 1 — Update `main.py`**

Locate the backtest block (the `backtesting_start` assignment and surrounding `BobsBrain.backtest(...)` call). Update `backtesting_start`, `backtesting_end`, `budget`, and the `parameters={...}` dict as needed for the experiment (e.g. `max_daily_candidates`, `corr_long_window`, weights). Do not change unrelated code.

**Step 2 — Run the backtest**

Run the backtest backgrounded so the agent does not rely on staying alive for the full duration. The `caffeinate -i` wrapper prevents macOS from sleeping while the process is running.

```bash
caffeinate -i python main.py &
```

Poll for completion by checking the DB at increasing intervals (start at 60 s, back off to 120 s):

```sql
SELECT completed_at FROM backtest_runs ORDER BY started_at DESC LIMIT 1;
```

If `completed_at IS NULL` after the process exits, flag as an anomaly — the run likely crashed.

**Step 3 — Retrieve the run ID**

```sql
SELECT run_id, started_at, completed_at
FROM backtest_runs
ORDER BY started_at DESC
LIMIT 1;
```

**Step 4 — Hand off to evaluation**

Pass the `run_id` to the backtest evaluation workflow. Follow the Backtest Evaluation steps in full. If multiple runs were planned, complete all before handing off and request a side-by-side comparison.
