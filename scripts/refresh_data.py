"""
Nightly data refresh: update the tradeable universe and warm the price cache.

Intended to run as a cron job on the DB instance so the price cache stays
current and BobsBrain always finds warm data without hitting Alpaca at runtime.

Usage:
    python -m scripts.refresh_data

Env:
    DB_URL              (default: postgresql://lumibob@localhost:5432/lumibob)
    ALPACA_API_KEY      required
    ALPACA_API_SECRET   required
    ALPACA_IS_PAPER     (default: true)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from AlpacaClient import AlpacaClient
from DatabaseClient import DatabaseClient
from StockDataCache import StockDataCache

DB_URL = os.getenv('DB_URL', 'postgresql://lumibob@localhost:5432/lumibob')
WARM_DAYS = 150


def main() -> None:
    api_key = os.environ['ALPACA_API_KEY']
    secret_key = os.environ['ALPACA_API_SECRET']
    paper = os.getenv('ALPACA_IS_PAPER', 'true').lower() == 'true'

    db = DatabaseClient(DB_URL)
    alpaca = AlpacaClient(api_key=api_key, secret_key=secret_key, paper=paper, mode='backtest')
    cache = StockDataCache(db, alpaca)

    print('[refresh] Fetching tradeable universe from Alpaca...')
    tickers = alpaca.get_tradeable_assets()
    db.upsert_tickers(tickers, 'ALPACA')
    print(f'[refresh] Upserted {len(tickers)} tickers.')

    print(f'[refresh] Warming price cache for last {WARM_DAYS} days...')
    cache.warm_cache(tickers, days=WARM_DAYS)
    print('[refresh] Done.')


if __name__ == '__main__':
    main()
