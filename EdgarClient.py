"""
Thin client for SEC EDGAR's free JSON APIs.

Provides the two things the filing-events pipeline needs (plan WS2a,
docs/plans/2026-07-31_composite-score-overhaul.md):

  * a ticker -> CIK map from company_tickers.json, and
  * per-company filing histories from the submissions API, item-coded for 8-Ks,
    with acceptance timestamps so events are point-in-time honest (a filing
    "exists" only from its acceptance timestamp forward).

SEC fair-access rules: identify yourself via User-Agent and stay under
10 requests/second. `_get` enforces both; keep it that way.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timezone

import requests

_UA = os.getenv(
    'EDGAR_USER_AGENT',
    'LumiBob research (contact: gcrosti@gmail.com)',
)
_MIN_INTERVAL_S = 0.12          # ~8 req/s, under the SEC's 10 req/s ceiling
_TICKER_MAP_URL = 'https://www.sec.gov/files/company_tickers.json'
_SUBMISSIONS_URL = 'https://data.sec.gov/submissions/{name}'

# 6-K is the foreign-private-issuer analogue of the 8-K (Shell, Eni, Canadian
# miners, ...). It carries no item codes, so its rows have items = '' — noisier
# (some issuers file weekly buyback 6-Ks), but without it foreign legs are
# invisible to event studies.
DEFAULT_FORMS = ('8-K', '8-K/A', '6-K')


class EdgarClient:
    def __init__(self, session: requests.Session | None = None):
        self._session = session or requests.Session()
        self._session.headers['User-Agent'] = _UA
        self._last_request_ts = 0.0

    # ------------------------------------------------------------------ http

    def _get(self, url: str) -> dict:
        wait = _MIN_INTERVAL_S - (time.monotonic() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        resp = self._session.get(url, timeout=30)
        self._last_request_ts = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------- ticker -> CIK

    @staticmethod
    def normalize_ticker(symbol: str) -> str:
        """EDGAR tickers use '-' where brokers use '.' or '/' (BRK.B -> BRK-B)."""
        return symbol.upper().replace('.', '-').replace('/', '-')

    def load_ticker_cik_map(self) -> dict[str, int]:
        """Ticker -> CIK for every registrant SEC lists. Funds and trusts are
        largely absent — an unmapped symbol usually means an ETF/CEF, and the
        caller should report coverage rather than treat absence as an error."""
        raw = self._get(_TICKER_MAP_URL)
        return {
            self.normalize_ticker(row['ticker']): int(row['cik_str'])
            for row in raw.values()
        }

    # ------------------------------------------------------------- filings

    def fetch_filings(
        self,
        cik: int,
        since: date,
        forms: tuple[str, ...] = DEFAULT_FORMS,
    ) -> list[dict]:
        """
        All filings of the given forms accepted on/after *since*.

        Returns dicts with: accession, form, items (comma-separated 8-K item
        codes, '' for other forms), filed_at (timezone-aware acceptance
        timestamp). Walks the submissions API's archived pages only as far
        back as *since* requires.
        """
        root = self._get(_SUBMISSIONS_URL.format(name=f'CIK{cik:010d}.json'))
        out = self._extract(root['filings']['recent'], since, forms)

        # Archived chunks are newest-first; each advertises its date range.
        for chunk in root['filings'].get('files', []):
            if date.fromisoformat(chunk['filingTo']) < since:
                continue
            page = self._get(_SUBMISSIONS_URL.format(name=chunk['name']))
            out.extend(self._extract(page, since, forms))
        return out

    @staticmethod
    def _extract(block: dict, since: date, forms: tuple[str, ...]) -> list[dict]:
        rows = []
        accepted = block.get('acceptanceDateTime', [])
        items_col = block.get('items', [])
        for i, form in enumerate(block.get('form', [])):
            if form not in forms:
                continue
            ts_raw = accepted[i] if i < len(accepted) else None
            if not ts_raw:
                continue
            filed_at = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
            if filed_at.astimezone(timezone.utc).date() < since:
                continue
            rows.append(dict(
                accession=block['accessionNumber'][i],
                form=form,
                items=items_col[i] if i < len(items_col) else '',
                filed_at=filed_at,
            ))
        return rows
