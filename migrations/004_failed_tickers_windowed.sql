-- Migration 004: scope failed_tickers by fetch window.
--
-- Before this change, a symbol was globally blacklisted the first time Alpaca
-- returned no data for it, regardless of the date window that was requested.
-- A symbol that has no data for a 2022 backtest (e.g. IPO'd in 2023) was
-- incorrectly excluded from 2023 backtests too, shrinking the clustering
-- universe over time and degrading pair discovery.
--
-- After this migration each row records the (window_start, window_end) for
-- which the fetch failed.  A symbol is only skipped when the requested window
-- overlaps with a stored failure window.
--
-- Existing rows (which have no window info) are assigned a sentinel range
-- (1970-01-01, 1970-01-01) so the NOT NULL constraint can be applied; they
-- will not match any real backtest window and are effectively inert.
-- Run "DELETE FROM failed_tickers;" after migrating to start fresh.

ALTER TABLE failed_tickers
    ADD COLUMN IF NOT EXISTS window_start DATE,
    ADD COLUMN IF NOT EXISTS window_end   DATE;

-- Drop the symbol-only primary key before adding the composite one.
ALTER TABLE failed_tickers DROP CONSTRAINT IF EXISTS failed_tickers_pkey;

-- Fill NULLs from pre-migration rows with a sentinel so NOT NULL can be set.
UPDATE failed_tickers
SET window_start = '1970-01-01',
    window_end   = '1970-01-01'
WHERE window_start IS NULL OR window_end IS NULL;

ALTER TABLE failed_tickers ALTER COLUMN window_start SET NOT NULL;
ALTER TABLE failed_tickers ALTER COLUMN window_end   SET NOT NULL;

ALTER TABLE failed_tickers ADD PRIMARY KEY (symbol, window_start, window_end);
