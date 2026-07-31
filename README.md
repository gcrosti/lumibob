# LumiBob

A **score-and-rank** lead/lag pairs-trading strategy built on [Lumibot](https://github.com/Lumiwealth/lumibot) and [Alpaca](https://alpaca.markets). LumiBob clusters a tradeable universe, scores pairs with a composite model (dual-horizon correlation plus Z-score depth), builds a daily **target portfolio** (top K, with displacement of weaker names), and trades the **lag** leg using Z-score spread signals. Use backtests and paper trading to validate behaviour before risking real capital.

## How It Works

Each trading day:

1. **Pair maintenance** — Existing pairs are re-scored with the same composite model (long/short correlation, Z-score depth). Positions that fall out of the target set are marked to sell (`displaced`); names entering the target set are marked to buy when not yet held. Sells where the Z-score reverted below `exit_threshold` are tagged `zscore_exit`; sells where price data was unavailable are tagged `data_missing`.

2. **Pair discovery** — Tickers are grouped with **`TickerClusterer`** (HDBSCAN on a **correlation-distance matrix** — `1 − Pearson ρ` — with **sector pre-partitioning**: ETFs, each known SIC sector, and unknown-sector tickers each cluster independently). A per-cluster sanity gate (`min_intra_cluster_corr`) dissolves loosely correlated clusters. Discovery walks **within-cluster** pairs, applies a penny-stock filter, respects a **global daily scoring budget** (`max_daily_candidates`) and **pair cooldown**. Candidates receive a **composite score**; together with existing positions they are ranked into a target book sized by **dynamic K** (`k_target = round(max_k * quality_scale)`, where `max_k` is a hard ceiling).

3. **Execution** — Sells run first, then buys. Buy size scales between **`min_position_pct`** and **`max_position_pct`** using each pair's **`composite_score`**, with a **deployment-gap** boost toward **`target_deployed_pct`**. If a buy exceeds available cash, the loop **continues** to smaller candidates rather than stopping.

Clustering uses the same **read-through price path** as the strategy (`StockDataCache`), with the first cluster build on the **first simulated session day** (correct `as_of` in backtests).

**Note:** `PairSimulator.py` and older sequential gates (cointegration, simulated Sharpe, watchlist) are **not** on the live discovery path today; `StockEvaluator` still exposes cointegration and legacy helpers for experiments.

## Architecture

```
main.py
└── BobsBrain (Lumibot Strategy)
      ├── DatabaseClient ──────────► PostgreSQL / TimescaleDB
      ├── AlpacaClient ───────────► Alpaca API
      ├── StockDataCache(db, alpaca)   read-through price cache (with retry)
      ├── StockEvaluator               dual-horizon correlation, Z-score depth / action
      └── TickerClusterer              correlation-distance HDBSCAN, sector pre-partition

tuning/
  parameter_space.py   26 tunable parameters across 3 tiers; suggest() for Optuna
  objective.py         BacktestObjective — Optuna callable; Sharpe-based scoring
  walk_forward.py      WalkForward fold generator (train + holdout windows)
  regime_detector.py   Market regime classifier (calm_bull / vol_shock / sideways / trend_bull)
  battery.py           Five-regime calibration test harness (Phase 3)
  studies/
    tier2_slow.py       Phase 1 proof study (Tier 2, single window)
    phase3_battery.py   Phase 3 five-regime battery vs baseline
    phase4_coarse.py    Phase 4 regime-conditioned Tier 3 tuning (12-fold walk-forward)

scripts/
  prewarm_cache.py          Pre-warm stock_prices for historical regime windows
  refresh_ticker_metadata.py  Re-fetch SEC EDGAR SIC metadata for all universe tickers
  after_battery.sh          Post-battery summary and artefact archiving
  watch_and_cutover.sh      Cut over active_parameters when gate criterion is met

migrations/
  001_add_exit_reason.sql   ALTER TABLE migration for exit_reason column
  002_coint_cache.sql       Add pair_coint_cache table and coint_pvalue/halflife_days columns to pairs
  003_short_leg.sql         trades.leg (long vs short fills) and pairs.lead_short_qty (H1)
  004_failed_tickers_windowed.sql  Scope failed_tickers by (window_start, window_end); replaces symbol-only PK
```

| Module | Responsibility |
|---|---|
| `BobsBrain.py` | Strategy — scoring, target portfolio, orders, daily snapshots |
| `AlpacaClient.py` | Thin wrapper around `alpaca-py` for assets and OHLCV |
| `DatabaseClient.py` | All PostgreSQL access — tickers, prices, metadata, pairs, runs, trades, snapshots |
| `StockDataCache.py` | Read-through cache: DB first, Alpaca for gaps (with exponential-backoff retry) |
| `StockEvaluator.py` | `get_correlation_dual`, Z-score spread and depth, legacy cointegration/correlation helpers |
| `TickerClusterer.py` | Correlation-distance HDBSCAN with sector pre-partition and sanity gate |
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

To apply the schema to an existing database that pre-dates Phase 1:

```bash
psql postgresql://postgres:lumibob@localhost:5432/lumibob -f migrations/001_add_exit_reason.sql
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

Strategy parameters are passed in **`main.py`** via `STRATEGY_PARAMETERS` (same keys for paper and backtest unless you diverge). All keys are persisted on each run in **`backtest_runs.settings`** (JSONB). Defaults below match **`BobsBrain.initialize()`** when a key is omitted.

### Data windows

| Parameter | Description | Default |
|---|---|---|
| `lookback_window` | Calendar days of price history loaded for scoring | `130` |
| `cluster_recompute_days` | Min days between cluster recomputes; `None` = cache for the run | `None` |

### Position sizing

| Parameter | Description | Default |
|---|---|---|
| `max_k` | Hard ceiling on target portfolio size | `20` |
| `min_position_pct` | Min fraction of portfolio for a new position (low score) | `0.03` |
| `max_position_pct` | Max fraction of portfolio for a new position (high score) | `0.20` |
| `target_deployed_pct` | Target deployed fraction; gap boosts per-buy allocation | `0.60` |
| `short_leg_fraction` | Fraction [0.0, 1.0] of long notional to short the lead stock. 0.0 = long-only; 1.0 = full dollar-neutral hedge. Replaces deprecated `enable_short_leg`. | `0.0` |

### Signal

| Parameter | Description | Default |
|---|---|---|
| `entry_threshold` | Z-score magnitude for spread entry | `2.0` |
| `exit_threshold` | Z-score band for spread exit | `0.5` |
| `zscore_window` | Rolling window (bars) for spread Z-score | `20` |

### Scoring (composite score)

| Parameter | Description | Default |
|---|---|---|
| `corr_long_window` | Long-horizon correlation window (bars) | `90` |
| `corr_short_window` | Short-horizon correlation window (bars) | `20` |
| `w_corr_long` | Composite weight on long correlation | `0.3` |
| `w_corr_short` | Composite weight on short correlation | `0.5` |
| `w_z_depth` | Composite weight on Z-score depth | `0.2` |
| `max_halflife_days` | Half-life ceiling for the persisted `score_halflife` observability column | `60` |

The composite is these three components only. Cointegration and half-life were removed from the score as dead weight for ranking (PR #50 / Pass A v4); their per-pair component scores (`score_coint`, `score_halflife`) and inputs (`coint_pvalue`, `halflife_days`) are still computed and persisted for post-hoc analysis. All `w_*` weights are normalised to sum to 1.0 by the tuning pipeline. When setting them manually (e.g. in `main.py`), pass pre-normalised values or call `tuning.parameter_space.normalize_weights()` before use.

### Discovery

| Parameter | Description | Default |
|---|---|---|
| `max_daily_candidates` | Max new candidate pairs scored per day | `200` |
| `cooldown_days` | Days before the same unordered pair is scored again | `7` |

### Filters

| Parameter | Description | Default |
|---|---|---|
| `penny_threshold` | Min price for penny-stock filter | `5.0` |

### Dynamic-K quality scale

| Parameter | Description | Default |
|---|---|---|
| `quality_scale_pivot` | Pool-correlation pivot for quality multiplier | `0.7` |
| `quality_scale_min` | Floor on quality_scale multiplier | `0.5` |
| `quality_scale_max` | Ceiling on quality_scale multiplier (must be ≤ 1.0) | `1.0` |

### Clustering / HDBSCAN

| Parameter | Description | Default |
|---|---|---|
| `cluster_lookback_days` | Days of price history used to build clusters | `126` |
| `hdbscan_min_cluster_size` | Min tickers to form a cluster | `5` |
| `hdbscan_min_samples` | HDBSCAN density parameter | `2` |
| `pca_variance` | PCA variance retained (Ward fallback path only) | `0.95` |
| `min_coverage` | Min non-NaN bar fraction to keep a ticker | `0.5` |
| `hdbscan_metric` | Distance metric (`'precomputed'` = correlation distance) | `'precomputed'` |
| `hdbscan_selection_method` | Cluster selection strategy | `'eom'` |
| `hdbscan_cluster_selection_epsilon` | Merge distance threshold | `0.0` |
| `min_intra_cluster_corr` | Dissolve clusters below this median intra-correlation | `0.3` |

## Execution Modes

### Backtest

Uses Lumibot's `YahooDataBacktesting` engine. Default window in **`main.py`**: 2024-01-02 through 2024-03-26, budget 10,000.

```bash
RUN_MODE=backtest python main.py
```

### Paper Trading

Connects to Alpaca paper trading and runs against live data.

```bash
RUN_MODE=paper python main.py
```

## Parameter Tuning Engine

The `tuning/` module implements a multi-phase Optuna-based tuning engine. See **`TUNING_ENGINE_PLAN.md`** for the full design.

**Phases completed:**

| Phase | What | Status |
|---|---|---|
| Phase 0 | Expose all magic numbers as tunable parameters | ✓ Done |
| Phase 1 | Optuna proof study (Tier 2, single window); gate: best-trial beats baseline | ✓ PASS (+36% score) |
| Phase 2 | HDBSCAN overhaul, sector pre-partition, K-ballooning fix | ✓ Done |
| Phase 3 | Five-regime battery (calm_bull, vol_shock, sideways, trend_bull, mixed) | ✓ FAIL (1/3) — see Phase 3.5 |
| Phase 3.5 | Strategy deep dive; H5 identified: only 12% of pairs are cointegrated | ✓ Done — see `STRATEGY_DEEPDIVE_FINDINGS.md` |
| Phase 4 (coarse) | Regime-conditioned Tier 3 tuning, 12-fold walk-forward | 🔲 Awaiting cointegration gate |

**Pre-Phase-4 gate:** A cointegration filter (Engle-Granger ADF) must be added and validated before launching the 600-trial Phase 4 study. See `STRATEGY_DEEPDIVE_FINDINGS.md` §H5.

To run the Phase 1 proof study:

```bash
caffeinate -i python -m tuning.studies.tier2_slow
```

To run the Phase 3 battery:

```bash
python -m tuning.studies.phase3_battery
```

To preview Phase 4 fold plan without running:

```bash
python -m tuning.studies.phase4_coarse --list-folds
```

## Database

PostgreSQL with [TimescaleDB](https://www.timescale.com/) for time-series prices. Use **`docker-compose.yml`** for a local instance.

### Tables

| Table | Purpose |
|---|---|
| `tickers` | Tradeable universe (Alpaca) |
| `stock_prices` | OHLCV cache (hypertable; compression/retention per schema) |
| `ticker_metadata` | Sector / ETF flags (SEC EDGAR SIC path); used by `TickerClusterer` sector pre-partition |
| `backtest_runs` | Run metadata and **`settings` JSONB** (all 26 strategy parameters) |
| `pairs` | Lead/lag pairs; **`correlation`** stores long-horizon **`corr_long`** |
| `portfolio_snapshots` | Daily portfolio, cash, SPY line, discovery funnel columns |
| `trades` | Fills with optional slippage; **`exit_reason`** distinguishes `zscore_exit` / `displaced` / `data_missing`; **`leg`** distinguishes lag long vs lead short leg |
| `failed_tickers` | Symbols skipped after bad price fetches |
| `tuning_studies` | Optuna study metadata, fold windows, holdout metrics |
| `active_parameters` | Current best-trial parameters used by live strategy |

### Schema Management

Apply **`schema.sql`** on a fresh database. For existing databases, apply incremental migrations:

```bash
psql postgresql://postgres:lumibob@localhost:5432/lumibob -f migrations/001_add_exit_reason.sql
psql postgresql://postgres:lumibob@localhost:5432/lumibob -f migrations/002_coint_cache.sql
psql postgresql://postgres:lumibob@localhost:5432/lumibob -f migrations/003_short_leg.sql
psql postgresql://postgres:lumibob@localhost:5432/lumibob -f migrations/004_failed_tickers_windowed.sql
```

**`DatabaseClient`** also runs idempotent **`ALTER … IF NOT EXISTS`** migrations at startup for incremental columns/tables.

## Tests

```bash
python -m pytest tests/
```

## License

*TBD*
