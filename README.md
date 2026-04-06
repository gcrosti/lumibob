# LumiBob

A **score-and-rank** lead/lag pairs-trading strategy built on [Lumibot](https://github.com/Lumiwealth/lumibot) and [Alpaca](https://alpaca.markets). LumiBob clusters a tradeable universe, scores pairs with a composite model (dual-horizon correlation plus Z-score depth), builds a daily **target portfolio** (top K, with displacement of weaker names), and trades the **lag** leg using Z-score spread signals. Use backtests and paper trading to validate behaviour before risking real capital.

## How It Works

Each trading day:

1. **Pair maintenance** — Existing pairs are re-scored with the same composite model (long/short correlation, Z-score depth). Positions that fall out of the target set are marked to sell; names entering the target set are marked to buy when not yet held.

2. **Pair discovery** — Tickers are grouped with **`TickerClusterer`** (HDBSCAN on PCA-reduced log-returns; Ward fallback when needed). Discovery walks **within-cluster** pairs (ordered by cached correlation), applies hard gates (e.g. penny filter, **same-sector or both-ETF** using SEC EDGAR SIC metadata), respects a **global daily scoring budget** (`max_daily_candidates`) and **pair cooldown**. Candidates receive a **composite score**; together with existing positions they are ranked into a target book sized by **dynamic K** (cash and pool quality).

3. **Execution** — Sells run first, then buys. Buy size scales between **`min_position_pct`** and **`max_position_pct`** using each pair’s **`composite_score`**, with a **deployment-gap** boost toward **`target_deployed_pct`**. If a buy exceeds available cash, the loop **continues** to smaller candidates rather than stopping.

Clustering uses the same **read-through price path** as the strategy (`StockDataCache`), with the first cluster build on the **first simulated session day** (correct `as_of` in backtests).

**Note:** `PairSimulator.py` and older sequential gates (cointegration, simulated Sharpe, watchlist) are **not** on the live discovery path today; `StockEvaluator` still exposes cointegration and legacy helpers for experiments.

## Architecture

```
main.py
└── BobsBrain (Lumibot Strategy)
      ├── DatabaseClient ──────────► PostgreSQL / TimescaleDB
      ├── AlpacaClient ───────────► Alpaca API
      ├── StockDataCache(db, alpaca)   read-through price cache
      ├── StockEvaluator               dual-horizon correlation, Z-score depth / action
      └── TickerClusterer              HDBSCAN / Ward clustering on returns
```

| Module | Responsibility |
|---|---|
| `BobsBrain.py` | Strategy — scoring, target portfolio, orders, daily snapshots |
| `AlpacaClient.py` | Thin wrapper around `alpaca-py` for assets and OHLCV |
| `DatabaseClient.py` | All PostgreSQL access — tickers, prices, metadata, pairs, runs, trades, snapshots |
| `StockDataCache.py` | Read-through cache: DB first, Alpaca for gaps |
| `StockEvaluator.py` | `get_correlation_dual`, Z-score spread and depth, legacy cointegration/correlation helpers |
| `TickerClusterer.py` | Movement-similarity clusters and within-cluster pair ordering |
| `PairSimulator.py` | Optional offline pair simulation / grid search (not wired into current discovery) |

## Prerequisites

- Python 3.10+
- Docker (for TimescaleDB)
- An [Alpaca](https://alpaca.markets) account (paper-trading keys are fine)

## Quick Start

```bash
# 1. Clone and install dependencies
git clone <repo-url> && cd LumiBob
pip install -r requirements.txt

# 2. Start TimescaleDB
docker compose up -d

# 3. Apply the schema
psql postgresql://postgres:lumibob@localhost:5432/lumibob -f schema.sql

# 4. Configure environment
cp .env.example .env
# Edit .env and add your Alpaca API key + secret

# 5. Run a backtest
python main.py
```

## Configuration

Environment variables (from `.env`):

| Variable | Description | Default |
|---|---|---|
| `ALPACA_API_KEY` | Alpaca API key | *(required)* |
| `ALPACA_API_SECRET` | Alpaca API secret | *(required)* |
| `ALPACA_IS_PAPER` | Use Alpaca paper trading endpoint | `true` |
| `DB_URL` | PostgreSQL connection string | `postgresql://postgres:lumibob@localhost:5432/lumibob` |
| `RUN_MODE` | `backtest` or `paper` | `backtest` |

Strategy parameters are passed in **`main.py`** via `parameters={...}` (same keys for paper and backtest unless you diverge). Defaults below match **`BobsBrain.initialize()`** when a key is omitted.

| Parameter | Description | Default |
|---|---|---|
| `lookback_window` | Calendar days of price history loaded for scoring (must cover `corr_long_window` in trading bars) | `130` |
| `cluster_recompute_days` | Minimum days between cluster recomputes; `None` = cache for the run | `None` |
| `min_position_pct` | Min fraction of portfolio for a new position (low composite score) | `0.03` |
| `max_position_pct` | Max fraction of portfolio for a new position (high composite score) | `0.20` |
| `target_deployed_pct` | Target deployed fraction; gap boosts per-buy allocation | `0.60` |
| `entry_threshold` | Z-score magnitude treated as “entry” for spread | `2.0` |
| `exit_threshold` | Z-score band for spread exit / shallow signal | `0.5` |
| `zscore_window` | Rolling window (bars) for spread Z-score | `20` |
| `corr_long_window` | Long-horizon correlation window (bars, log-returns) | `90` |
| `corr_short_window` | Short-horizon correlation window (bars) | `20` |
| `w_corr_long` | Composite weight on long correlation | `0.3` |
| `w_corr_short` | Composite weight on short correlation | `0.5` |
| `w_z_depth` | Composite weight on Z-score depth | `0.2` |
| `max_daily_candidates` | Max new candidate pairs scored per day (global budget) | `200` |
| `cooldown_days` | Days before the same unordered pair is scored again | `7` |
| `replacement_threshold` | Min composite-score edge for a candidate to displace a held name | `0.05` |

These are persisted on each run in **`backtest_runs.settings`** (JSONB). The default **`main.py`** backtest passes position sizing, correlation windows, weights, and discovery budgets; keys such as **`lookback_window`**, **`entry_threshold`**, **`exit_threshold`**, **`zscore_window`**, and **`cluster_recompute_days`** are omitted there and therefore use the defaults in the table above.

## Execution Modes

### Backtest

Uses Lumibot’s `YahooDataBacktesting` engine. Default window in **`main.py`**: 2024-01-02 through 2024-03-26, budget 10,000 (see `BobsBrain.backtest(...)`).

```bash
RUN_MODE=backtest python main.py
```

### Paper Trading

Connects to Alpaca paper trading and runs against live data.

```bash
RUN_MODE=paper python main.py
```

## Database

PostgreSQL with [TimescaleDB](https://www.timescale.com/) for time-series prices. Use **`docker-compose.yml`** for a local instance.

### Tables

| Table | Purpose |
|---|---|
| `tickers` | Tradeable universe (Alpaca) |
| `stock_prices` | OHLCV cache (hypertable; compression/retention per schema) |
| `ticker_metadata` | Sector / ETF flags (SEC EDGAR SIC path); supports sector gate |
| `backtest_runs` | Run metadata and **`settings` JSONB** |
| `pairs` | Lead/lag pairs; **`correlation`** stores long-horizon **`corr_long`** |
| `portfolio_snapshots` | Daily portfolio, cash, SPY line, discovery funnel columns |
| `trades` | Fills with optional slippage |
| `failed_tickers` | Symbols skipped after bad price fetches |

### Schema Management

Apply **`schema.sql`** on a fresh database. **`DatabaseClient`** runs idempotent **`ALTER … IF NOT EXISTS`** migrations at startup for incremental columns/tables.

## Tests

```bash
python -m pytest tests/
```

## License

*TBD*
