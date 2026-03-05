import yfinance as yf
import pandas as pd

class YahooDBReader:
    """
    reads data from Yahoo Finance
    """

    def get_exchange_tickers(self):
        """Fetches official ticker lists from Nasdaq FTP."""
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
        other_url = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

        nasdaq_df = pd.read_csv(nasdaq_url, sep='|')[:-1]
        other_df = pd.read_csv(other_url, sep='|')[:-1]

        # 'N' is NYSE, 'G/S/Q' are various Nasdaq tiers
        nyse = other_df[other_df['Exchange'] == 'N']['ACT Symbol'].tolist()
        nasdaq = nasdaq_df['Symbol'].tolist()

        return list(set(nyse + nasdaq))  # Combine and remove duplicates

    def clean_tickers_for_yahoo(self, ticker_list):
        """
        Translates Nasdaq FTP ticker formats into Yahoo Finance formats.
        Drops tickers that cannot be mapped to a valid Yahoo Finance symbol.
        """
        cleaned_list = []
        for ticker in ticker_list:

            if pd.isna(ticker) or not isinstance(ticker, str):
                continue

            # 1. Remove any extra whitespace
            t = ticker.strip()

            # 2. Drop tickers containing '$' — Nasdaq uses these for certain
            #    special/test symbols that have no valid Yahoo Finance equivalent
            if '$' in t:
                continue

            # 3. Handle Share Classes/Preferreds:
            # Nasdaq often uses dots (BRK.B) or spaces (PFE PR A)
            # Yahoo uses hyphens (BRK-B, PFE-PA)
            t = t.replace(' ', '-')
            t = t.replace('.', '-')

            # 4. Specific fix for Warrants (Nasdaq '.W' -> Yahoo '-WT')
            if t.endswith('-W'):
                t = t + 'T'

            cleaned_list.append(t)

        return list(set(cleaned_list))  # Ensure uniqueness

    def get_all_stocks(self, start_date, end_date, limit=None):
        print("Fetching ticker lists...")
        raw_tickers = self.get_exchange_tickers()  # Assuming your previous FTP function

        # Clean tickers for Yahoo format
        cleaned_tickers = self.clean_tickers_for_yahoo(raw_tickers)

        if limit:
            cleaned_tickers = cleaned_tickers[:limit]

        print(f"Downloading data for {len(cleaned_tickers)} stocks...")

        # Fetch data
        data = yf.download(
            tickers=cleaned_tickers,
            start=start_date,
            end=end_date,
            group_by='column',  # Group by price type (Open, Close, etc.)
            threads=True
        )

        if 'Close' in data.columns:
            # Extract the 'Close' cross-section.
            # This turns the MultiIndex into a simple DataFrame where
            # columns = Tickers and rows = Dates.
            close_data = data['Close']

            # Remove columns that are entirely NaN (stocks with no data found)
            close_data = close_data.dropna(axis=1, how='all')

            return close_data

        return pd.DataFrame()  # Return empty if no data


    def get_stocks(self, stock_tickers, start_date, end_date):
        """
        returns stock data from specific tickers during the given timeframe
        """
        return yf.download(tickers=stock_tickers, start=start_date, end=end_date)['Close']


