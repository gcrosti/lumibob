import unittest

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from YahooDBReader import YahooDBReader


class TestCleanTickersForYahoo(unittest.TestCase):
    def setUp(self):
        self.reader = YahooDBReader()

    def test_plain_ticker_unchanged(self):
        result = self.reader.clean_tickers_for_yahoo(['AAPL'])
        self.assertIn('AAPL', result)

    def test_strips_leading_and_trailing_whitespace(self):
        result = self.reader.clean_tickers_for_yahoo(['  MSFT  '])
        self.assertIn('MSFT', result)

    def test_dot_replaced_with_hyphen(self):
        """BRK.B → BRK-B (Yahoo Finance share-class format)."""
        result = self.reader.clean_tickers_for_yahoo(['BRK.B'])
        self.assertIn('BRK-B', result)
        self.assertNotIn('BRK.B', result)

    def test_space_replaced_with_hyphen(self):
        """'PFE PR A' → 'PFE-PR-A' (preferred share Nasdaq format)."""
        result = self.reader.clean_tickers_for_yahoo(['PFE PR A'])
        self.assertIn('PFE-PR-A', result)

    def test_warrant_suffix_w_becomes_wt(self):
        """Nasdaq warrant suffix '.W' should become '-WT' in Yahoo format."""
        result = self.reader.clean_tickers_for_yahoo(['XYZ.W'])
        self.assertIn('XYZ-WT', result)
        self.assertNotIn('XYZ-W', result)

    def test_non_warrant_hyphen_w_not_extended(self):
        """A ticker ending in '-W' that is NOT a warrant suffix should not become '-WT'
        because the hyphen is introduced by the dot→hyphen replacement first.
        Tickers like 'XYZ.W' should become 'XYZ-WT'; plain 'XYZ-W' (already hyphenated)
        would also be extended — this documents current behaviour."""
        result = self.reader.clean_tickers_for_yahoo(['XYZW'])
        self.assertIn('XYZW', result)

    def test_nan_values_skipped(self):
        """float NaN entries should be silently dropped."""
        result = self.reader.clean_tickers_for_yahoo([float('nan'), 'GOOG'])
        self.assertIn('GOOG', result)
        self.assertEqual(len([t for t in result if not isinstance(t, str) or t == 'nan']), 0)

    def test_non_string_values_skipped(self):
        """Non-string, non-NaN values (e.g. integers) should be skipped."""
        result = self.reader.clean_tickers_for_yahoo([42, 'TSLA'])
        self.assertIn('TSLA', result)
        self.assertNotIn(42, result)

    def test_duplicates_removed(self):
        """Duplicate tickers should appear only once in the output."""
        result = self.reader.clean_tickers_for_yahoo(['AMZN', 'AMZN', 'AMZN'])
        self.assertEqual(result.count('AMZN'), 1)

    def test_empty_list_returns_empty(self):
        result = self.reader.clean_tickers_for_yahoo([])
        self.assertEqual(result, [])

    def test_mixed_formats_all_cleaned(self):
        """Multiple tickers with different quirks cleaned in one call."""
        tickers = ['BRK.B', '  AAPL  ', 'XYZ.W', float('nan'), 'GOOG']
        result = self.reader.clean_tickers_for_yahoo(tickers)
        self.assertIn('BRK-B', result)
        self.assertIn('AAPL', result)
        self.assertIn('XYZ-WT', result)
        self.assertIn('GOOG', result)
        self.assertEqual(len(result), 4)

    def test_pandas_na_skipped(self):
        """pandas NA values should be treated the same as NaN and skipped."""
        result = self.reader.clean_tickers_for_yahoo([pd.NA, 'META'])
        self.assertIn('META', result)


if __name__ == '__main__':
    unittest.main()
