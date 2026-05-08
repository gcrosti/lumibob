# LumiBob — Claude Instructions

## Tone

Be direct and honest. The job is to improve the code — not to make the user feel good.

- Never open with praise: "Great question!", "Good instinct!", "That's a smart approach"
- Never soften a genuine problem with a compliment first
- Never add filler affirmations: "Absolutely!", "Of course!", "Certainly!"
- Start with the substance. If there is a problem, say so immediately.
- Disagree when the user is wrong. State it plainly and explain why.
- Be kind in *tone* (no condescension), but never let kindness suppress honesty.

## Formatting

Never write a bare `$` character in chat responses. Dollar signs break markdown rendering.

- Write `DB_URL` not `$DB_URL`
- Write `RUN_MODE=backtest python main.py` not `$RUN_MODE`
- Write `env var FOO` or just `FOO` not `$FOO`

Dollar signs inside fenced SQL or shell code blocks are acceptable only when they are genuinely part of the syntax, but prefer plain names even there when possible.

## Verify Before Proceeding

The most common source of wasted work is accepting an initial result, hypothesis, or proposed fix at face value.

**Before committing to an analysis approach**: check that the required data actually exists. A silent failure — 100% `data_missing`, an empty result set, an implausibly round number — is a red flag, not a result. If a fallback or proxy is used, state it explicitly and quantify the loss of precision.

**Before writing a hypothesis or conclusion**: draft conclusions from data, not toward data. Write the hypothesis after the evidence is in hand. If the evidence contradicts the headline claim, fix the headline.

**Before proposing a fix**:
- Read the evidence just gathered. A fix that suppresses a mechanism the evidence shows is working is wrong.
- Check whether the problem is already addressed elsewhere.
- Is the solution as simple as it can be? Simple is fantastic, simplistic is fatal.

**When a result looks clean or obvious**: ask what would make this wrong. A result that perfectly confirms the prior hypothesis is worth a second look.

---

## Data Storage

All run data, price history, and pair state are stored in a TimescaleDB database.
Connection string: env var `DB_URL` (default: `postgresql://postgres:lumibob@localhost:5432/lumibob`).
Schema is defined in `schema.sql`.

**Never call `yfinance` or Alpaca directly for price data** — always go through `StockDataCache` (read-through: DB first, Alpaca only for gaps).

### `backtest_runs`
One row per backtest or paper trading run.

| Column | Type | Description |
|---|---|---|
| `run_id` | VARCHAR(10) PK | 6-char Lumibot run ID |
| `mode` | VARCHAR(10) | `backtest` or `paper` |
| `started_at` | TIMESTAMPTZ | Strategy startup time |
| `completed_at` | TIMESTAMPTZ | Strategy finish time (`NULL` if crashed) |
| `settings` | JSONB | Full settings dict |

### `portfolio_snapshots`
One row per trading day per run. TimescaleDB hypertable.

| Column | Type | Description |
|---|---|---|
| `time` | TIMESTAMPTZ | Trading day timestamp |
| `run_id` | VARCHAR(10) | FK → `backtest_runs` |
| `portfolio_value` | NUMERIC | Total portfolio value |
| `cash` | NUMERIC | Uninvested cash |
| `spy_value` | NUMERIC | SPY normalised to starting portfolio value |
| `active_pairs` | INT | Number of pairs being monitored |
| `avg_correlation` | NUMERIC | Mean of finite long-horizon correlations across active pairs |
| `cash_ratio` | NUMERIC | `cash / portfolio_value` |
| `daily_buys` | INT | New pair entry buy orders submitted that day |
| `daily_sells` | INT | Sell orders submitted that day |
| `daily_topups` | INT | Legacy; strategy passes 0 |
| `pairs_scanned` | INT | Pair evaluations that cleared hard gates and were scored |
| `candidates_found` | INT | Candidate pairs scored that day |
| `candidates_buy_ready` | INT | New buy slots queued into the target portfolio that day |

### `trades`
One row per filled order. All rows are fills — no status filtering needed.

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Auto-increment |
| `run_id` | VARCHAR(10) | FK → `backtest_runs` |
| `pair_id` | INT | FK → `pairs` (nullable) |
| `symbol` | VARCHAR(20) | Ticker traded |
| `side` | VARCHAR(4) | `buy` or `sell` |
| `quantity` | NUMERIC | Shares filled |
| `price` | NUMERIC | Fill price |
| `slippage` | NUMERIC | Slippage incurred |
| `filled_at` | TIMESTAMPTZ | Execution timestamp |
| `exit_reason` | VARCHAR | `zscore_exit`, `displaced`, or `data_missing` (post Phase 2) |

### `pairs`
Discovered lead/lag pairs.

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Auto-increment |
| `lead_symbol` | VARCHAR(20) | Lead ticker |
| `lag_symbol` | VARCHAR(20) | Lag ticker |
| `correlation` | NUMERIC | Long-horizon Pearson correlation (`corr_long`); updated on save and daily re-score |
| `active` | BOOLEAN | Whether pair is currently being monitored |

### `stock_prices`
OHLCV price cache. TimescaleDB hypertable, partitioned by `time`. 2-year rolling retention, compressed after 90 days.

### `tickers`
Tradeable universe. Refreshed nightly via `AlpacaClient.get_tradeable_assets()`.

### Legacy log files
Lumibot writes flat files to `logs/` as a side effect. These are **not** the authoritative source — use the DB. They remain useful only for runs that pre-date the DB integration.

Naming: `logs/BobsBrain_<YYYY-MM-DD>_<HH-MM>_<6-char-id>_<type>.<ext>`

- `_settings.json` — superseded by `backtest_runs`
- `_stats.csv` — superseded by `portfolio_snapshots`
- `_trade_events.csv` — superseded by `trades`; filter to `status=fill` if using
- `_indicators.csv` — currently empty

---

## Backtest Evaluation

Before evaluating, read `README.md` for context, but prefer **`backtest_runs.settings`** as the source of truth (README may lag the code). The live strategy uses **clustering + composite score-and-rank** discovery, **dual-horizon** correlations, **Z-score depth**, and **position sizing** — not the legacy sequential gates unless evaluating an old run.

### Step 1 — Identify the run

```sql
SELECT run_id, mode, started_at, completed_at, settings
FROM backtest_runs
ORDER BY started_at DESC
LIMIT 1;
```

### Step 2 — Load configuration

Read the `settings` JSONB field. Report all keys present. Typical current keys (post Phase 2):

| Family | Keys |
|---|---|
| Data windows | `lookback_window`, `cluster_recompute_days` |
| Position sizing | `max_k`, `min_position_pct`, `max_position_pct`, `target_deployed_pct` |
| Signal | `entry_threshold`, `exit_threshold`, `zscore_window` |
| Scoring | `corr_long_window`, `corr_short_window`, `w_corr_long`, `w_corr_short`, `w_z_depth` |
| Discovery | `max_daily_candidates`, `cooldown_days` |
| Filters | `penny_threshold` |
| Dynamic-K | `quality_scale_pivot`, `quality_scale_min`, `quality_scale_max` |
| Clustering | `cluster_lookback_days`, `hdbscan_min_cluster_size`, `hdbscan_min_samples`, `pca_variance`, `min_coverage`, `hdbscan_metric`, `hdbscan_selection_method`, `hdbscan_cluster_selection_epsilon`, `min_intra_cluster_corr` |

### Step 3 — Evaluate portfolio performance

```sql
SELECT time, portfolio_value, cash, spy_value,
       active_pairs, avg_correlation, cash_ratio,
       daily_buys, daily_sells,
       pairs_scanned, candidates_found, candidates_buy_ready
FROM portfolio_snapshots
WHERE run_id = '<run_id>'
ORDER BY time;
```

Compute and report:
- **Total return %**: `(final portfolio_value - budget) / budget * 100`
- **Max drawdown %**: largest peak-to-trough drop in `portfolio_value`
- **Cash utilization**: average `cash_ratio` — high values mean capital was idle
- **SPY comparison**: compare `portfolio_value` vs `spy_value` (both normalised to starting budget)
- **Avg active pairs / avg correlation**

### Step 4 — Evaluate trade activity

```sql
SELECT symbol, side, quantity, price, slippage, filled_at, pair_id, exit_reason
FROM trades
WHERE run_id = '<run_id>'
ORDER BY filled_at;
```

Compute: total fills, buy/sell split, avg slippage, symbols traded, exit reason breakdown.

Exit reason breakdown (post Phase 2):
```sql
SELECT exit_reason, COUNT(*)
FROM trades
WHERE run_id = '<run_id>' AND side = 'sell'
GROUP BY exit_reason;
```

High `displaced` share relative to `zscore_exit` means displacement — not mean-reversion — is the primary exit driver.

Win rate and avg P&L: match each sell to its corresponding buy by `symbol`. P&L = `(sell_price - buy_price) * quantity - slippage`.

### Step 5 — Flag anomalies

- Zero or very few trades despite a multi-week backtest
- Portfolio value unchanged despite filled trades
- Cash utilization > 80% throughout — use `candidates_found` and `candidates_buy_ready` to distinguish thin discovery from weak entry signals
- `avg_correlation` all NULL/NaN while `active_pairs` > 0
- Avg slippage unusually high relative to trade cost
- Sells without matching buys
- `completed_at IS NULL` in `backtest_runs`

---

## Workflows

For complex multi-step tasks, follow the relevant workflow:

- **Backtesting a change** → backtest-agent workflow below
- **Strategy deep dive / underperformance analysis** → deepdive-agent workflow below
- **PR review** → pr-reviewer workflow below

@.claude/skills/backtest-agent.md
@.claude/skills/deepdive-agent.md
@.claude/skills/pr-reviewer.md
