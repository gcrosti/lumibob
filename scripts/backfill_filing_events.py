"""
Backfill (and incrementally refresh) the filing_events table from SEC EDGAR.

Maps the tradeable universe to CIKs via company_tickers.json, then pulls
item-coded 8-K filings per company through the submissions API. Unmapped
symbols are almost always ETFs/CEFs (funds file under trust CIKs) — they are
counted and reported, not treated as errors. Resumable: already-stored
filings are skipped by the (symbol, accession) primary key, and --skip-fresh
avoids re-fetching symbols already refreshed recently.

Usage:
    python -m scripts.backfill_filing_events                # full backfill since 2021-01-01
    python -m scripts.backfill_filing_events --since 2026-07-01   # incremental (cron-able)

Env:
    DB_URL              (default: postgresql://lumibob@localhost:5432/lumibob)
    EDGAR_USER_AGENT    optional override for the SEC-required contact UA
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from DatabaseClient import DatabaseClient
from EdgarClient import EdgarClient

DB_URL = os.getenv('DB_URL', 'postgresql://lumibob@localhost:5432/lumibob')
DEFAULT_SINCE = '2021-01-01'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default=DEFAULT_SINCE,
                    help='earliest acceptance date to fetch (YYYY-MM-DD)')
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='restrict to these symbols (default: whole tickers table)')
    ap.add_argument('--skip-fresh-hours', type=float, default=0.0,
                    help='skip symbols whose newest fetched_at is younger than this')
    args = ap.parse_args()
    since = date.fromisoformat(args.since)

    db = DatabaseClient(DB_URL)
    db.migrate_filing_events()
    edgar = EdgarClient()

    symbols = args.symbols or db.get_tickers()
    cik_map = edgar.load_ticker_cik_map()

    fresh: set[str] = set()
    if args.skip_fresh_hours > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=args.skip_fresh_hours)
        with db._conn() as conn:  # noqa: SLF001 — script-local convenience
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT symbol FROM filing_events GROUP BY symbol '
                    'HAVING MAX(fetched_at) >= %s', (cutoff,))
                fresh = {r[0] for r in cur.fetchall()}

    mapped, unmapped, inserted, errors = 0, 0, 0, 0
    for i, sym in enumerate(sorted(symbols)):
        cik = cik_map.get(EdgarClient.normalize_ticker(sym))
        if cik is None:
            unmapped += 1
            continue
        mapped += 1
        if sym in fresh:
            continue
        try:
            filings = edgar.fetch_filings(cik, since=since)
        except Exception as exc:  # noqa: BLE001 — keep the sweep going
            errors += 1
            print(f'  {sym}: fetch failed ({exc})')
            continue
        inserted += db.upsert_filing_events(
            [dict(symbol=sym, cik=cik, **f) for f in filings])
        if (i + 1) % 250 == 0:
            print(f'  {i + 1}/{len(symbols)} symbols | mapped {mapped} | '
                  f'inserted {inserted}')

    print(f'done: {len(symbols)} symbols | mapped {mapped} | unmapped {unmapped} '
          f'(mostly funds) | new filings {inserted} | fetch errors {errors}')


if __name__ == '__main__':
    main()
