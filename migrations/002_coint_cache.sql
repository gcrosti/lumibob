-- Migration 002: cointegration cache and pair coint columns
-- Idempotent — safe to run multiple times.

-- Cross-run cache for cointegration test results.
-- Keyed by (lead, lag, lookback_window, window_end_date) so results are
-- shared across battery runs covering the same date range.
CREATE TABLE IF NOT EXISTS pair_coint_cache (
    lead_symbol     VARCHAR(20)       NOT NULL,
    lag_symbol      VARCHAR(20)       NOT NULL,
    lookback_window INT               NOT NULL,
    window_end_date DATE              NOT NULL,
    coint_pvalue    DOUBLE PRECISION  NOT NULL,
    halflife_days   DOUBLE PRECISION,
    computed_at     TIMESTAMP         NOT NULL DEFAULT NOW(),
    PRIMARY KEY (lead_symbol, lag_symbol, lookback_window, window_end_date)
);

-- Add cointegration columns to pairs for post-hoc analysis.
ALTER TABLE pairs ADD COLUMN IF NOT EXISTS coint_pvalue    DOUBLE PRECISION;
ALTER TABLE pairs ADD COLUMN IF NOT EXISTS halflife_days   DOUBLE PRECISION;
