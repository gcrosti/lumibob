"""
Ticker metadata refresh: sector (SIC) + instrument class, for all tickers.

Run nightly. Populates `ticker_metadata`, which `TickerClusterer` uses to
pre-partition the universe (ETFs and each sector cluster separately).

2026-08-01 rewrite — the previous version had four defects that between them
disabled the ETF partition in live clustering (see
`docs/deepdives/2026-08-01_disasters-surviving-the-event-gate.md` §4):

  1. `is_etf` was HARDCODED False on every SEC upsert, so this script actively
     clobbered any correct ETF flag written by another source. QQQ carried
     source='sec_edgar', is_etf=False for exactly this reason.
  2. The re-fetch query skipped rows already flagged `is_etf`, so a symbol
     could never be re-classified, while genuine funds were retried against
     EDGAR forever (they have no CIK and never resolve).
  3. It DELETEd every dotted symbol as an "artifact". Those are preferred
     shares — the best-performing cohort in the replay pool (+96 to +272 bps
     mean). Deleting their metadata drops them into the unknown-sector
     partition.
  4. `sector` mixed two taxonomies: SIC-derived strings from this script and
     yfinance's own strings from another path, so e.g. "Financial Services"
     and "Finance, Insurance & Real Estate" partitioned separately.

Instrument classification: a symbol absent from SEC's company_tickers.json is
almost certainly a fund (ETFs/CEFs file under trust CIKs, not their trading
symbol). That is the primary signal and it is free. `--verify-etf` adds an
optional yfinance `quoteType` pass to confirm a sample.

Usage:
    python scripts/refresh_ticker_metadata.py [--dry-run] [--full] [--verify-etf N]

Env:
    DB_URL  (default: postgresql://lumibob@localhost:5432/lumibob)
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from BobsBrain import _sic_to_sector  # noqa: E402

DB_URL = os.getenv('DB_URL', 'postgresql://lumibob@localhost:5432/lumibob')
EDGAR_HEADERS = {'User-Agent': os.getenv(
    'EDGAR_USER_AGENT', 'LumiBob research (contact: gcrosti@gmail.com)')}
RATE_LIMIT_SLEEP = 0.11   # ~9 req/s — SEC fair-access guideline
BATCH_SIZE = 200
STALE_DAYS = 30           # re-resolve rows older than this on a --full run


def _conn():
    return psycopg2.connect(DB_URL)


def coverage_report(label: str) -> dict:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*),
                       COUNT(sector),
                       COUNT(CASE WHEN is_etf THEN 1 END),
                       COUNT(sic_code)
                FROM ticker_metadata
            """)
            total, has_sector, etf, has_sic = cur.fetchone()
            cur.execute('SELECT COUNT(*) FROM tickers')
            universe = cur.fetchone()[0]
    print(f"\n[coverage:{label}]")
    print(f"  universe (tickers)   : {universe}")
    print(f"  ticker_metadata rows : {total}")
    print(f"  with sector          : {has_sector} ({has_sector / max(total,1) * 100:.1f}%)")
    print(f"  with SIC code        : {has_sic} ({has_sic / max(total,1) * 100:.1f}%)")
    print(f"  marked ETF/fund      : {etf} ({etf / max(total,1) * 100:.1f}%)")
    return {'total': total, 'has_sector': has_sector, 'etf': etf}


def symbols_to_process(full: bool) -> list[str]:
    """
    Symbols needing (re)classification.

    Default: rows missing entirely, or never successfully classified — i.e. no
    SIC code AND not yet identified as a fund.  `--full` additionally re-resolves
    anything older than STALE_DAYS.

    Unlike the previous version this does NOT delete dotted symbols (preferred
    shares are tradeable and well-performing) and does NOT skip rows flagged
    is_etf (a wrong flag must be correctable).
    """
    sql = """
        SELECT t.symbol
        FROM tickers t
        LEFT JOIN ticker_metadata m ON t.symbol = m.symbol
        WHERE m.symbol IS NULL
           OR (m.sic_code IS NULL AND NOT m.is_etf)
    """
    if full:
        sql += f"   OR m.fetched_at < NOW() - INTERVAL '{STALE_DAYS} days'"
    sql += ' ORDER BY t.symbol'
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]


def build_cik_map() -> dict[str, int]:
    """{TICKER: cik} for every SEC registrant. Absence implies a fund."""
    urls = ['https://www.sec.gov/files/company_tickers.json',
            'https://www.sec.gov/files/company_tickers_exchange.json']
    for url in urls:
        try:
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if 'fields' in data:
                rows = [dict(zip(data['fields'], r)) for r in data['data']]
                cik_map = {str(r['ticker']).upper(): int(r['cik'])
                           for r in rows if r.get('ticker')}
            else:
                cik_map = {str(v['ticker']).upper(): int(v['cik_str'])
                           for v in data.values()
                           if isinstance(v, dict) and v.get('ticker')}
            print(f"[edgar] CIK map from {url}: {len(cik_map):,} entries")
            return cik_map
        except Exception as exc:
            print(f"[edgar] {url} failed: {exc}")
    raise RuntimeError('Could not download EDGAR CIK map.')


def normalize_symbol(symbol: str) -> str:
    """Broker tickers use '.'/'/' where EDGAR uses '-' (BRK.B -> BRK-B)."""
    return symbol.upper().replace('.', '-').replace('/', '-')


def fetch_sic(cik: int) -> int | None:
    url = f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json'
    try:
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=10)
        if r.status_code == 200:
            sic = r.json().get('sic')
            return int(sic) if sic else None
    except Exception:
        pass
    return None


def refresh(dry_run: bool = False, full: bool = False, verify_etf: int = 0) -> None:
    print('=' * 60)
    print('Ticker metadata refresh (sector + instrument class)')
    print('=' * 60)
    before = coverage_report('before')

    symbols = symbols_to_process(full)
    print(f"\n[refresh] symbols to process: {len(symbols)}")
    if not symbols:
        print('[refresh] nothing to do.')
        return

    cik_map = build_cik_map()
    fetched_at = datetime.now(timezone.utc)
    batch: list[tuple] = []
    n_sec, n_fund, n_nosic = 0, 0, 0

    for i, symbol in enumerate(symbols, start=1):
        cik = cik_map.get(normalize_symbol(symbol))
        if cik is None:
            # No SEC registrant under this ticker -> fund (ETF/CEF/trust).
            batch.append((symbol, None, True, fetched_at, None, None, 'fund_inferred'))
            n_fund += 1
        else:
            sic = fetch_sic(cik)
            if sic is None:
                # Registered with SEC but carrying no SIC code.  Operating
                # companies essentially always have one; this bucket samples as
                # closed-end funds, BDCs and trusts (QQQ, MXF, ASA, ADX, PHK...)
                # — the old-style ETFs that register under their own ticker and
                # so are missed by the no-CIK test above.  Classifying these as
                # funds may occasionally mis-slot an odd operating company, but
                # the cost is only which cluster partition it lands in.
                n_nosic += 1
                batch.append((symbol, None, True, fetched_at, None, None,
                              'fund_inferred_nosic'))
            else:
                n_sec += 1
                sector = _sic_to_sector(sic)
                batch.append((symbol, sector, False, fetched_at, sic, sector,
                              'sec_edgar'))
            time.sleep(RATE_LIMIT_SLEEP)

        if len(batch) >= BATCH_SIZE and not dry_run:
            _upsert(batch)
            batch = []
        if i % 250 == 0 or i == len(symbols):
            print(f"[refresh] {i}/{len(symbols)} ({i / len(symbols) * 100:.1f}%) — "
                  f"sec {n_sec}, funds {n_fund}, cik-but-no-sic {n_nosic}")

    if batch and not dry_run:
        _upsert(batch)

    if verify_etf and not dry_run:
        _verify_etf_sample(verify_etf)

    after = coverage_report('after')
    print(f"\n[refresh] sector delta +{after['has_sector'] - before['has_sector']}, "
          f"ETF/fund delta +{after['etf'] - before['etf']}")


def _upsert(rows: list[tuple]) -> None:
    """Upsert. NB: is_etf and sector are now written from real signals, never
    hardcoded, so a wrong prior value is corrected rather than preserved."""
    sql = """
        INSERT INTO ticker_metadata
            (symbol, sector, is_etf, fetched_at, sic_code, sic_sector, source)
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


def _verify_etf_sample(n: int) -> None:
    """Spot-check the fund inference against yfinance quoteType."""
    try:
        import yfinance as yf
    except ImportError:
        print('[verify] yfinance not installed; skipping')
        return
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT symbol, is_etf FROM ticker_metadata '
                        'ORDER BY random() LIMIT %s', (n,))
            sample = cur.fetchall()
    agree = checked = 0
    for symbol, is_etf in sample:
        try:
            qt = yf.Ticker(symbol).info.get('quoteType', '')
        except Exception:
            continue
        if not qt:
            continue
        checked += 1
        agree += int(bool(is_etf) == (qt.upper() in ('ETF', 'MUTUALFUND', 'CLOSEDEND')))
        time.sleep(0.3)
    if checked:
        print(f"\n[verify] fund inference agrees with yfinance quoteType on "
              f"{agree}/{checked} ({agree / checked * 100:.0f}%)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Refresh ticker metadata.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--full', action='store_true',
                    help=f're-resolve rows older than {STALE_DAYS} days')
    ap.add_argument('--verify-etf', type=int, default=0, metavar='N',
                    help='spot-check N symbols against yfinance quoteType')
    args = ap.parse_args()
    refresh(dry_run=args.dry_run, full=args.full, verify_etf=args.verify_etf)
