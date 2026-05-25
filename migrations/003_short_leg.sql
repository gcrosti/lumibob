-- Migration 003: short leg (H1) — trade leg tagging + persisted short qty on pairs
-- Idempotent — safe to run multiple times.

ALTER TABLE trades ADD COLUMN IF NOT EXISTS leg VARCHAR(5) NOT NULL DEFAULT 'long';

ALTER TABLE pairs ADD COLUMN IF NOT EXISTS lead_short_qty NUMERIC;

-- Enforce leg values when constraint is not already present (fresh installs use schema.sql).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'trades_leg_check'
    ) THEN
        ALTER TABLE trades ADD CONSTRAINT trades_leg_check CHECK (leg IN ('long', 'short'));
    END IF;
END $$;
