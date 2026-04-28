-- LumiBob database schema
-- Apply once against a running TimescaleDB instance:
--   psql $DB_URL -f schema.sql

-- ---------------------------------------------------------------------------
-- Tickers universe
-- One row per symbol, refreshed nightly via AlpacaClient.get_tradeable_assets().
-- Replaces the Nasdaq FTP HTTP call that previously happened on every
-- before_market_opens() invocation.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tickers (
    symbol       VARCHAR(20) PRIMARY KEY,
    exchange     VARCHAR(10),
    last_updated DATE NOT NULL
);

-- ---------------------------------------------------------------------------
-- OHLCV price cache
-- One row per symbol per trading day. TimescaleDB hypertable partitions the
-- data by time so a 60-day range scan touches only 2 chunks regardless of
-- total history size.
--
-- Retention: 2 years rolling (auto-drop via TimescaleDB policy).
-- Compression: after 90 days (~8x reduction).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stock_prices (
    time    TIMESTAMPTZ  NOT NULL,
    symbol  VARCHAR(20)  NOT NULL,
    open    NUMERIC,
    high    NUMERIC,
    low     NUMERIC,
    close   NUMERIC      NOT NULL,
    volume  BIGINT
);

SELECT create_hypertable('stock_prices', 'time', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS stock_prices_symbol_time_idx
    ON stock_prices (symbol, time DESC);

ALTER TABLE stock_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('stock_prices', INTERVAL '90 days',
    if_not_exists => TRUE);

SELECT add_retention_policy('stock_prices', INTERVAL '2 years',
    if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- Discovered lead/lag pairs
-- Replaces pairs/pair_history.json. Each row carries the full pair config so
-- historical queries can answer "which pairs had the highest correlation?" or
-- "how did pairs discovered in November perform vs. December?"
--
-- run_id scopes each pair to the run that discovered it, mirroring the
-- pattern used by trades and portfolio_snapshots. This prevents active pairs
-- from one run polluting pair discovery and loading in subsequent runs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pairs (
    id            SERIAL PRIMARY KEY,
    run_id        VARCHAR(10) REFERENCES backtest_runs(run_id),
    lead_symbol   VARCHAR(20) NOT NULL,
    lag_symbol    VARCHAR(20) NOT NULL,
    lag_days      INT         NOT NULL DEFAULT 1,
    short_ma      INT         NOT NULL DEFAULT 2,
    long_ma       INT         NOT NULL DEFAULT 5,
    -- Long-horizon Pearson correlation (BobsBrain ``corr_long``)
    correlation   NUMERIC,
    simulated_return DOUBLE PRECISION,
    initial_cost  NUMERIC,
    discovered_at DATE        NOT NULL,
    last_updated  DATE,
    active        BOOLEAN     NOT NULL DEFAULT TRUE
);

-- One active configuration per (run, lead, lag, lag_days) triple.
-- Partial index: only enforced when run_id is set so legacy NULL rows
-- (created before this column existed) are left untouched.
CREATE UNIQUE INDEX IF NOT EXISTS pairs_run_lead_lag_days_idx
    ON pairs (run_id, lead_symbol, lag_symbol, lag_days)
    WHERE run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS pairs_lag_symbol_active_idx
    ON pairs (lag_symbol, active);

-- ---------------------------------------------------------------------------
-- Backtest / paper run metadata
-- Every other table foreign-keys here so all data for a specific run can be
-- isolated in a single query. Replaces the Lumibot-generated _settings.json.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id       VARCHAR(10) PRIMARY KEY,
    mode         VARCHAR(10) NOT NULL CHECK (mode IN ('backtest', 'paper')),
    started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    settings     JSONB
);

-- ---------------------------------------------------------------------------
-- Portfolio snapshots + all strategy indicators
-- One row per trading day per run. Replaces _stats.csv and _indicators.csv.
-- Columns mirror all self.add_line() calls in BobsBrain.on_trading_iteration().
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    run_id          VARCHAR(10) NOT NULL REFERENCES backtest_runs(run_id),
    portfolio_value NUMERIC,
    cash            NUMERIC,
    -- indicators (mirrors all six self.add_line() calls in BobsBrain)
    spy_value       NUMERIC,      -- SPY normalised to starting portfolio value
    active_pairs    INT,          -- pairs being monitored that day
    avg_correlation NUMERIC,      -- mean Pearson correlation across active pairs
    cash_ratio      NUMERIC,      -- cash / portfolio_value
    daily_buys           INT,     -- new pair entry buy orders (excludes top-ups)
    daily_sells          INT,     -- sell orders submitted
    daily_topups         INT,     -- top-up buy orders submitted
    pairs_scanned        INT,     -- combinations evaluated through quality gates
    candidates_found     INT,     -- passed all quality gates (penny/corr/coint/sim)
    candidates_buy_ready INT      -- subset of candidates_found with a buy signal
);

SELECT create_hypertable('portfolio_snapshots', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS portfolio_snapshots_run_id_idx
    ON portfolio_snapshots (run_id, time DESC);

-- ---------------------------------------------------------------------------
-- Individual trade fills
-- One row per filled order. Foreign-keyed to both backtest_runs and pairs so
-- per-pair P&L can be computed across all runs. Replaces _trade_events.csv.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    id          SERIAL PRIMARY KEY,
    run_id      VARCHAR(10) NOT NULL REFERENCES backtest_runs(run_id),
    pair_id     INT         REFERENCES pairs(id),
    symbol      VARCHAR(20) NOT NULL,
    side        VARCHAR(4)  NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity    NUMERIC     NOT NULL,
    price       NUMERIC     NOT NULL,
    slippage    NUMERIC     NOT NULL DEFAULT 0,
    filled_at   TIMESTAMPTZ NOT NULL,
    exit_reason VARCHAR(20) CHECK (exit_reason IN ('zscore_exit', 'displaced', 'data_missing'))
);

CREATE INDEX IF NOT EXISTS trades_run_id_symbol_idx
    ON trades (run_id, symbol, filled_at DESC);

-- ---------------------------------------------------------------------------
-- Unfetchable symbols
-- Symbols for which Alpaca returned no price data are recorded here so the
-- fetch attempt is skipped on every subsequent day, eliminating repeated
-- API calls and log noise for delisted or invalid tickers.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS failed_tickers (
    symbol     VARCHAR(20) PRIMARY KEY,
    reason     TEXT,
    failed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Tuning engine tables (Phase 1+)
-- ---------------------------------------------------------------------------

-- One row per Optuna study run.  The compute receipt columns let us plot
-- marginal Sharpe gain per trial and make data-driven scaling decisions.
CREATE TABLE IF NOT EXISTS tuning_studies (
    study_id            SERIAL PRIMARY KEY,
    study_name          VARCHAR(100) NOT NULL,
    tier                INT NOT NULL,           -- 1, 2, or 3
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    train_start         DATE NOT NULL,
    train_end           DATE NOT NULL,
    holdout_start       DATE,                   -- NULL for single-window proof studies
    holdout_end         DATE,
    best_params         JSONB,
    best_objective      NUMERIC,
    holdout_metrics     JSONB,                  -- scored after study completes
    -- compute receipt
    n_trials_completed  INT,
    n_trials_pruned     INT,
    wall_clock_seconds  INT,
    parallel_jobs       INT DEFAULT 1,
    machine             VARCHAR(100)
);

-- Active parameter set for live / paper trading.
-- BobsBrain.initialize() will add one line to read from here when Phase 5
-- is implemented.  For now this table is populated manually or by the
-- tuner after a study completes.
CREATE TABLE IF NOT EXISTS active_parameters (
    effective_date      DATE PRIMARY KEY,
    regime_label        INT,                    -- NULL until Phase 4 regime detector
    params              JSONB NOT NULL,
    source_study_id     INT REFERENCES tuning_studies(study_id)
);
