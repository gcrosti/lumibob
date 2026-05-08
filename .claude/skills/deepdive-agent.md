# Deep Dive Agent

Use when asked to run a strategy deep dive, analyze why LumiBob is underperforming, or produce a STRATEGY_DEEPDIVE_FINDINGS.md document.

Four phases: validate data → gather evidence → draft hypotheses → produce findings. **Never write hypotheses before data is in hand. Never propose a fix before reading the evidence it is based on.**

---

## Phase 1 — Validate Data Availability

Before committing to any analysis approach, verify the required data exists for the target run's date window.

```sql
-- Confirm price data covers the backtest window
SELECT MIN(time)::date, MAX(time)::date FROM stock_prices WHERE symbol = 'SPY';

-- Confirm trades and snapshots are populated
SELECT COUNT(*) FROM trades WHERE run_id = '<run_id>';
SELECT COUNT(*) FROM portfolio_snapshots WHERE run_id = '<run_id>';
```

If data is missing for an analysis: state the gap, state what would be needed, and propose an alternative. If a proxy is used (e.g. holding-period heuristic instead of direct z-score replay), flag it and quantify the precision loss.

---

## Phase 2 — Gather Evidence

Run all analyses before writing any hypotheses. Populate each table with real query results.

| Analysis | Tables | Key outputs |
|---|---|---|
| Regime performance | `portfolio_snapshots`, `backtest_runs` | Return %, SPY comparison, Sharpe, max DD, avg cash ratio, avg pairs |
| Trade quality | `trades` | Win rate, avg P&L per round-trip, median hold days |
| Market neutrality | `portfolio_snapshots`, `stock_prices` | Beta, R², return correlation (OLS: `strat_return ~ spy_return`) |
| Candidate funnel | `portfolio_snapshots` | `candidates_found`, `candidates_buy_ready`, conversion rate |
| Exit reasons | `trades` | If `exit_reason` column exists: displaced vs zscore_exit split; otherwise reconstruct from coincident-buy proxy |

**Before recommending any parameter intervention**, read `tuning/parameter_space.py` and check:
- Is this parameter managed by the tuning engine?
- Is the desired value already within the current bounds?

If yes to both: **the correct intervention is retraining on different data, not changing the default.** Do not recommend default changes for tuned parameters.

---

## Phase 3 — Draft Hypotheses

Write each hypothesis only after its supporting evidence is in hand. If two hypotheses are interdependent (e.g. an exit-mechanism hypothesis depends on a displacement analysis), complete the dependency first.

Required structure:

```
**Hypothesis N: [Short title]**

**Evidence:** [Specific numbers from Phase 2]
**Mechanism:** [Why this causes underperformance]
**Proposed change:** [Specific, testable intervention]
**How to validate:** [What a follow-up backtest should show if the fix works]
```

Consistency checks before finalising each hypothesis:
- Does the proposed change follow from the mechanism? If the evidence shows a mechanism is already working (e.g. displacement reducing losses), do not propose suppressing it.
- Does the evidence contradict the headline? Cash near target → do not frame as "cash drag."
- Does the proposed change address the root cause, or a symptom? Prefer root-cause interventions.

---

## Phase 4 — Findings Document

Produce `STRATEGY_DEEPDIVE_FINDINGS.md` from the completed analyses and hypotheses.

Sections (in order):
1. **Summary table** — one row per run per regime: return %, SPY %, vs SPY, Sharpe, max DD %, avg cash %, avg pairs
2. **Trade activity** — total trades, win rate, avg P&L/trip, median hold
3. **Market neutrality** — beta, R², return correlation per run
4. **Candidate funnel** — avg found, avg buy-ready, conversion %
5. **Hypotheses** — all from Phase 3, in order of expected impact
6. **Recommended changes** — ranked table; structural design flaws rank above any parameter tuning work regardless of effort
7. **Go / no-go** — explicit statement with reasoning; if the stated goal (e.g. beat SPY) is structurally unachievable without a design change, say so directly
8. **Execution checklist**

**Priority ranking guidance for section 6:**
A structural flaw that prevents achieving the stated goal (e.g. long-only bias when the goal is market-neutral returns) always ranks above parameter tuning, interim mitigations, and observability work — regardless of implementation effort.
