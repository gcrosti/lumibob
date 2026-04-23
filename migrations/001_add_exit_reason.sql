-- Migration 001: add exit_reason to trades
-- Apply against an existing database:
--   psql DB_URL -f migrations/001_add_exit_reason.sql
--
-- exit_reason values:
--   zscore_exit  — spread reverted below exit_threshold
--   displaced    — pair was crowded out of the top-K target portfolio by a higher-scoring candidate
--   data_missing — price data unavailable at sell time; exit reason could not be determined

ALTER TABLE trades
    ADD COLUMN IF NOT EXISTS exit_reason VARCHAR(20)
    CHECK (exit_reason IN ('zscore_exit', 'displaced', 'data_missing'));
