"""Unit tests for EdgarClient (pure logic only — no network)."""
import os
import sys
from datetime import date, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from EdgarClient import DEFAULT_FORMS, EdgarClient


class TestNormalizeTicker:
    def test_dots_and_slashes_become_dashes(self):
        assert EdgarClient.normalize_ticker('BRK.B') == 'BRK-B'
        assert EdgarClient.normalize_ticker('brk/b') == 'BRK-B'

    def test_plain_symbol_uppercased(self):
        assert EdgarClient.normalize_ticker('rgld') == 'RGLD'


class TestDefaultForms:
    def test_includes_foreign_private_issuer_form(self):
        # Without 6-K, foreign legs (SHEL, E, BMA, ...) are invisible to
        # event studies — see the 2026-07-31 deepdive.
        assert '6-K' in DEFAULT_FORMS
        assert '8-K' in DEFAULT_FORMS


def _block(**overrides):
    """A minimal submissions-API filing block, newest-first like EDGAR's."""
    block = dict(
        accessionNumber=['0001-24-000002', '0001-23-000001'],
        form=['8-K', '10-Q'],
        acceptanceDateTime=['2024-02-22T13:00:13.000Z', '2023-05-01T09:30:00.000Z'],
        items=['2.02,9.01', ''],
    )
    block.update(overrides)
    return block


class TestExtract:
    def test_filters_by_form(self):
        rows = EdgarClient._extract(_block(), since=date(2020, 1, 1),
                                    forms=('8-K',))
        assert [r['form'] for r in rows] == ['8-K']
        assert rows[0]['items'] == '2.02,9.01'
        assert rows[0]['accession'] == '0001-24-000002'

    def test_filters_by_since_date(self):
        rows = EdgarClient._extract(
            _block(form=['8-K', '8-K']),
            since=date(2024, 1, 1), forms=('8-K',))
        assert len(rows) == 1
        assert rows[0]['filed_at'].year == 2024

    def test_acceptance_timestamp_is_timezone_aware(self):
        rows = EdgarClient._extract(_block(), since=date(2020, 1, 1),
                                    forms=('8-K',))
        assert rows[0]['filed_at'].tzinfo is not None
        assert rows[0]['filed_at'].astimezone(timezone.utc).hour == 13

    def test_missing_acceptance_timestamp_row_skipped(self):
        rows = EdgarClient._extract(
            _block(acceptanceDateTime=['', '2023-05-01T09:30:00.000Z']),
            since=date(2020, 1, 1), forms=('8-K', '10-Q'))
        assert [r['form'] for r in rows] == ['10-Q']

    def test_missing_items_column_yields_empty_items(self):
        block = _block()
        del block['items']
        rows = EdgarClient._extract(block, since=date(2020, 1, 1), forms=('8-K',))
        assert rows[0]['items'] == ''
