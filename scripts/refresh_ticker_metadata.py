"""
Standalone SEC EDGAR metadata refresh for all tickers in the `tickers` table.

Run once before Phase 3 (and nightly thereafter) to maximise sector coverage.

Key improvements over the inline BobsBrain fetch:
  - Uses company_tickers.json (all SEC registrants) instead of the exchange-
    only file, giving meaningfully broader CIK coverage.
  - Retries every ticker that currently has a NULL sector (not just missing
    rows), so stale placeholder records get a second chance.
  - Removes dotted-symbol artifacts (preferred shares, warrants, etc.) that
    slipped into ticker_metadata before the '.' filter was added to
    AlpacaClient.get_tradeable_assets().
  - Prints a before/after coverage summary.

Usage:
    python scripts/refresh_ticker_metadata.py [--dry-run]

Env:
    DB_URL  (default: postgresql://postgres:lumibob@localhost:5432/lumibob)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras
import requests

# ---------------------------------------------------------------------------
# Path setup so we can import BobsBrain._sic_to_sector without importing the
# full strategy class (which would trigger lumibot imports).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from BobsBrain import _sic_to_sector  # noqa: E402

DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')
EDGAR_HEADERS = {'User-Agent': 'LumiBob research@lumibob.local'}
RATE_LIMIT_SLEEP = 0.11   # ~9 req/s — SEC fair-access guideline
BATCH_SIZE = 200          # upsert rows in batches to avoid huge transactions


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn():
    return psycopg2.connect(DB_URL)


def coverage_report(label: str) -> dict:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                               AS total,
                    COUNT(sector)                                          AS has_sector,
                    COUNT(CASE WHEN is_etf THEN 1 END)                    AS etf_count
                FROM ticker_metadata
            """)
            total, has_sector, etf_count = cur.fetchone()
            cur.execute('SELECT COUNT(*) FROM tickers')
            universe = cur.fetchone()[0]
    pct = has_sector / total * 100 if total else 0
    print(f"\n[coverage:{label}]")
    print(f"  ticker_metadata rows : {total}")
    print(f"  universe (tickers)   : {universe}")
    print(f"  with sector          : {has_sector} ({pct:.1f}%)")
    print(f"  marked ETF           : {etf_count}")
    return {'total': total, 'has_sector': has_sector, 'universe': universe}


def fetch_symbols_needing_refresh() -> list[str]:
    """
    Return every symbol from the `tickers` table that either:
      - has no row in ticker_metadata at all, OR
      - has a NULL sector and is not an ETF (stale placeholder).

    Also removes dotted-symbol artifacts (e.g. ABR.PRD) that pre-date
    the '.' exclusion filter in AlpacaClient.get_tradeable_assets().
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            # Remove dotted-symbol artifacts that shouldn't be in the table
            cur.execute("""
                DELETE FROM ticker_metadata
                WHERE symbol LIKE '%.%'
            """)
            deleted = cur.rowcount
            if deleted:
                print(f"[refresh] Removed {deleted} dotted-symbol artifact rows "
                      "(preferred shares / warrants — filtered by AlpacaClient).")

            cur.execute("""
                SELECT t.symbol
                FROM tickers t
                LEFT JOIN ticker_metadata m ON t.symbol = m.symbol
                WHERE m.symbol IS NULL          -- no metadata row yet
                   OR (m.sector IS NULL AND NOT m.is_etf)  -- stale placeholder
                ORDER BY t.symbol
            """)
            symbols = [row[0] for row in cur.fetchall()]
    return symbols


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------

def build_cik_map() -> dict[str, int]:
    """
    Download the broader company_tickers.json (all SEC registrants) and return
    a {TICKER_UPPER: cik} dict.  Falls back to company_tickers_exchange.json
    when the broader file is unavailable.
    """
    urls = [
        'https://www.sec.gov/files/company_tickers.json',
        'https://www.sec.gov/files/company_tickers_exchange.json',
    ]
    for url in urls:
        try:
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # company_tickers.json is {index: {cik_str, ticker, title}, ...}
            # company_tickers_exchange.json is {fields:[...], data:[[...],...]}
            if 'fields' in data:
                fields = data['fields']
                rows = data['data']
                df_data = [dict(zip(fields, r)) for r in rows]
                cik_map = {
                    str(r['ticker']).upper(): int(r['cik'])
                    for r in df_data
                    if r.get('ticker')
                }
            else:
                cik_map = {
                    str(v['ticker']).upper(): int(v['cik_str'])
                    for v in data.values()
                    if isinstance(v, dict) and v.get('ticker')
                }
            print(f"[edgar] Loaded CIK map from {url}: {len(cik_map):,} entries")
            return cik_map
        except Exception as exc:
            print(f"[edgar] {url} failed: {exc}")
    raise RuntimeError("Could not download EDGAR CIK map from any URL.")


def fetch_sic(cik: int) -> int | None:
    """Fetch SIC code for a single CIK from the submissions endpoint."""
    cik_padded = str(int(cik)).zfill(10)
    url = f'https://data.sec.gov/submissions/CIK{cik_padded}.json'
    try:
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=6)
        if r.status_code == 200:
            return r.json().get('sic')
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main refresh logic
# ---------------------------------------------------------------------------

def refresh(dry_run: bool = False) -> None:
    print("=" * 60)
    print("SEC EDGAR ticker metadata refresh")
    print("=" * 60)

    before = coverage_report("before")

    symbols = fetch_symbols_needing_refresh()
    print(f"\n[refresh] Symbols to process: {len(symbols)}")

    if not symbols:
        print("[refresh] Nothing to do — all tickers have sector data.")
        return

    print("[edgar] Downloading CIK map...")
    cik_map = build_cik_map()

    fetched_at = datetime.now(timezone.utc)
    matched_records: list[dict] = []
    unmatched: list[str] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        cik = cik_map.get(symbol.upper())
        if cik is None:
            unmatched.append(symbol)
        else:
            sic = fetch_sic(cik)
            sector = _sic_to_sector(sic)
            matched_records.append({
                'symbol':     symbol,
                'sic_code':   int(sic) if sic else None,
                'sic_sector': sector,
                'is_etf':     False,
                'fetched_at': fetched_at,
            })
            time.sleep(RATE_LIMIT_SLEEP)

        if i % 100 == 0 or i == total:
            pct = i / total * 100
            print(f"[edgar] {i}/{total} ({pct:.1f}%) — matched: {len(matched_records)}, "
                  f"unmatched: {len(unmatched)}")

        # Flush matched records in batches
        if len(matched_records) >= BATCH_SIZE and not dry_run:
            _upsert_sec_batch(matched_records)
            matched_records = []

    # Final flush of matched records
    if matched_records and not dry_run:
        _upsert_sec_batch(matched_records)

    # Store placeholders for unmatched so we don't retry every run
    if unmatched and not dry_run:
        _upsert_placeholder_batch(unmatched, fetched_at)

    print(f"\n[refresh] Done. Matched: {len(matched_records) + (BATCH_SIZE - len(matched_records) % BATCH_SIZE) % BATCH_SIZE}, "
          f"unmatched (placeholders stored): {len(unmatched)}")

    after = coverage_report("after")
    delta = after['has_sector'] - before['has_sector']
    print(f"\n[refresh] Sector coverage delta: +{delta} tickers")


def _upsert_sec_batch(records: list[dict]) -> None:
    rows = [
        (
            r['symbol'],
            r['sic_sector'],
            r['is_etf'],
            r['fetched_at'],
            r['sic_code'],
            r['sic_sector'],
            'sec_edgar',
        )
        for r in records
    ]
    sql = """
        INSERT INTO ticker_metadata (symbol, sector, is_etf, fetched_at, sic_code, sic_sector, source)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE
            SET sector     = EXCLUDED.sector,
                is_etf     = EXCLUDED.is_etf,
                fetched_at = EXCLUDED.fetched_at,
                sic_code   = EXCLUDED.sic_code,
                sic_sector = EXCLUDED.sic_sector,
                source     = EXCLUDED.source
    """
    with _conn() as conn:
        psycopg2.extras.execute_values(conn.cursor(), sql, rows)


def _upsert_placeholder_batch(symbols: list[str], fetched_at: datetime) -> None:
    rows = [(s, None, False, fetched_at) for s in symbols]
    sql = """
        INSERT INTO ticker_metadata (symbol, sector, is_etf, fetched_at)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE
            SET fetched_at = EXCLUDED.fetched_at
    """
    with _conn() as conn:
        psycopg2.extras.execute_values(conn.cursor(), sql, rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refresh SEC EDGAR ticker metadata.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Fetch and report without writing to the DB.',
    )
    args = parser.parse_args()
    refresh(dry_run=args.dry_run)
