"""
Standalone price-data pre-warmer for Phase 3 historical regime windows.

Fetches OHLCV data from Alpaca for a given date range and stores it in the
stock_prices / stock_ohlcv tables so that subsequent BobsBrain backtests
find a warm cache.

Key difference from StockDataCache.warm_cache():
  - Accepts any historical date range (not just "last N days from now").
  - Does NOT touch the failed_tickers table, so tickers that simply didn't
    exist yet in the historical window are not permanently blacklisted.
  - Batches requests to respect Alpaca's rate limits.

Usage
-----
    # Pre-warm a named regime (adds lookback buffer automatically):
    python scripts/prewarm_cache.py --regime sideways_2022

    # Pre-warm a custom date range:
    python scripts/prewarm_cache.py --start 2021-09-01 --end 2022-12-31

    # Pre-warm all cold regimes:
    python scripts/prewarm_cache.py --all-cold

    # Dry-run: show what would be fetched without calling Alpaca:
    python scripts/prewarm_cache.py --regime sideways_2022 --dry-run

Available regime names:
    calm_bull_2017   vol_shock_2020   sideways_2022
    trend_bull_2023  mixed_2024

Env:
    DB_URL              (default: postgresql://postgres:lumibob@localhost:5432/lumibob)
    ALPACA_API_KEY      required
    ALPACA_API_SECRET   required
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Allow running as __main__ from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from AlpacaClient import AlpacaClient
from DatabaseClient import DatabaseClient
from tuning.battery import REGIMES, Regime, check_warmth

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')
ALPACA_KEY    = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET = os.getenv('ALPACA_API_SECRET', '')

# How many calendar days of lookback to add before the regime start
# (must cover BobsBrain's lookback_window default of 130 calendar days plus buffer).
LOOKBACK_BUFFER_DAYS = 150

# Batch size for Alpaca requests (symbols per call).
SYMBOL_BATCH = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_all_symbols(db: DatabaseClient) -> list[str]:
    """Return all symbols in the tickers table."""
    import psycopg2
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT symbol FROM tickers ORDER BY symbol')
            return [r[0] for r in cur.fetchall()]


def _count_cached_days(db: DatabaseClient, start: datetime, end: datetime) -> int:
    """Return the number of distinct trading days already in stock_prices for the range."""
    import psycopg2
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(DISTINCT time::date)
                FROM stock_prices
                WHERE time >= %s AND time <= %s
                """,
                (start, end),
            )
            return cur.fetchone()[0] or 0


def prewarm(
    start: date,
    end: date,
    dry_run: bool = False,
) -> None:
    """
    Fetch price data for *all tickers* over [start - LOOKBACK_BUFFER_DAYS, end]
    and upsert into the stock_prices table.

    Does NOT modify failed_tickers — missing data for a historical date range
    is expected (tickers that didn't exist yet) and should not be permanent.
    """
    fetch_start = start - timedelta(days=LOOKBACK_BUFFER_DAYS)
    fetch_end = end

    start_dt = datetime.combine(fetch_start, datetime.min.time())
    end_dt   = datetime.combine(fetch_end, datetime.min.time())

    print(f'\n[prewarm] Date range: {fetch_start} → {fetch_end}')

    db = DatabaseClient(DB_URL)
    alpaca = AlpacaClient(ALPACA_KEY, ALPACA_SECRET)

    symbols = _get_all_symbols(db)
    print(f'[prewarm] Universe: {len(symbols)} symbols')

    cached_before = _count_cached_days(db, start_dt, end_dt)
    print(f'[prewarm] Cached trading days before: {cached_before}')

    if dry_run:
        print('[prewarm] Dry-run — no Alpaca calls will be made.')
        return

    print(f'[prewarm] Fetching in batches of {SYMBOL_BATCH}...')
    total_batches = (len(symbols) + SYMBOL_BATCH - 1) // SYMBOL_BATCH
    records_written = 0

    for i in range(0, len(symbols), SYMBOL_BATCH):
        batch = symbols[i:i + SYMBOL_BATCH]
        batch_num = i // SYMBOL_BATCH + 1
        print(f'[prewarm] Batch {batch_num}/{total_batches} ({len(batch)} symbols)...')

        try:
            # Use get_historical_bars which returns a close-price DataFrame.
            # We then store via upsert_prices (close prices in stock_prices).
            bars = alpaca.get_historical_bars(batch, start_dt, end_dt)
            if not bars.empty:
                db.upsert_prices(bars)
                records_written += bars.shape[0] * bars.shape[1]
                print(f'           → {bars.shape[1]} symbols, {bars.shape[0]} days')
            else:
                print(f'           → no data returned')
        except Exception as exc:
            print(f'[prewarm] Batch {batch_num} failed: {exc}')
            continue

    cached_after = _count_cached_days(db, start_dt, end_dt)
    print(f'\n[prewarm] Done. Cached trading days after: {cached_after} (+{cached_after - cached_before})')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _regime_by_name(name: str) -> Regime:
    for r in REGIMES:
        if r.name == name:
            return r
    valid = ', '.join(r.name for r in REGIMES)
    raise ValueError(f'Unknown regime "{name}". Valid: {valid}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pre-warm StockDataCache for Phase 3 historical regime windows.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--regime', metavar='NAME',
        help='Named regime to pre-warm (e.g. sideways_2022).',
    )
    group.add_argument(
        '--all-cold',
        action='store_true',
        help='Pre-warm all regimes whose data is not yet in the DB.',
    )
    group.add_argument(
        '--start', metavar='YYYY-MM-DD',
        help='Custom start date (also requires --end).',
    )
    parser.add_argument(
        '--end', metavar='YYYY-MM-DD',
        help='Custom end date (required with --start).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fetched without calling Alpaca.',
    )
    args = parser.parse_args()

    if args.regime:
        regime = _regime_by_name(args.regime)
        is_warm, note = check_warmth(regime)
        if is_warm:
            print(f'[prewarm] {regime.name} is already warm: {note}')
            print('[prewarm] Nothing to do.')
            return
        print(f'[prewarm] {regime.name}: {note}')
        prewarm(regime.start, regime.end, dry_run=args.dry_run)

    elif args.all_cold:
        cold = [(r, note) for r in REGIMES for _warm, note in [check_warmth(r)] if not _warm]
        # Rebuild list properly
        cold = []
        for r in REGIMES:
            is_warm, note = check_warmth(r)
            if not is_warm:
                cold.append((r, note))

        if not cold:
            print('[prewarm] All regimes are already warm.')
            return

        print(f'[prewarm] {len(cold)} cold regimes to pre-warm:')
        for r, note in cold:
            print(f'  {r.name:<25}  {note}')

        for r, note in cold:
            print(f'\n[prewarm] === {r.name} ===')
            prewarm(r.start, r.end, dry_run=args.dry_run)

    elif args.start:
        if not args.end:
            parser.error('--end is required with --start')
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)
        prewarm(start, end, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
