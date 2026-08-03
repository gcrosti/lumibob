"""
Backfill (and refresh) daily closed-end-fund NAVs into nav_prices.

Fetches each fund's official daily NAV via its free Nasdaq mirror symbol
(X<ticker>X convention, e.g. VKQ -> XVKQX) through yfinance, and stores it
under the fund's own trading symbol. Funds without a working mirror symbol
are reported as uncovered — never silently proxied (CLAUDE.md rule).

Note: this is NAV data, not tradeable price data — the StockDataCache/yfinance
price rule does not apply.

Usage:
    python -m scripts.backfill_nav_prices --symbols VKQ VMO MXE
    python -m scripts.backfill_nav_prices --symbols-file funds.txt --start 2021-01-01

Env:
    DB_URL   (default: postgresql://lumibob@localhost:5432/lumibob)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf

from DatabaseClient import DatabaseClient

DB_URL = os.getenv('DB_URL', 'postgresql://lumibob@localhost:5432/lumibob')
DEFAULT_START = '2021-01-01'
SLEEP_S = 0.4


def mirror_symbol(symbol: str) -> str:
    return f'X{symbol.upper()}X'


def fetch_nav(symbol: str, start: str) -> list[dict]:
    hist = yf.Ticker(mirror_symbol(symbol)).history(start=start, auto_adjust=False)
    if hist is None or hist.empty:
        return []
    return [
        dict(symbol=symbol, day=idx.date(), nav=float(row.Close))
        for idx, row in hist.iterrows()
        if row.Close and row.Close > 0
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=None)
    ap.add_argument('--symbols-file', default=None,
                    help='newline-separated fund symbols')
    ap.add_argument('--start', default=DEFAULT_START)
    args = ap.parse_args()

    symbols = list(args.symbols or [])
    if args.symbols_file:
        symbols += Path(args.symbols_file).read_text().split()
    if not symbols:
        ap.error('provide --symbols or --symbols-file')
    symbols = sorted(set(s.upper() for s in symbols))

    db = DatabaseClient(DB_URL)
    db.migrate_nav_prices()

    covered, uncovered, written = [], [], 0
    for i, sym in enumerate(symbols):
        try:
            rows = fetch_nav(sym, args.start)
        except Exception:
            rows = []
        if rows:
            written += db.upsert_nav_prices(rows)
            covered.append(sym)
        else:
            uncovered.append(sym)
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(symbols)}')
        time.sleep(SLEEP_S)

    print(f'done: {len(covered)}/{len(symbols)} funds covered | {written} NAV rows')
    if uncovered:
        print(f'UNCOVERED (no mirror symbol / no data): {uncovered}')


if __name__ == '__main__':
    main()
