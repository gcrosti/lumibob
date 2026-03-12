"""
AlpacaClient — boundary between LumiBob and the Alpaca API.

Wraps the alpaca-py SDK and exposes only what the strategy needs:
  - a tradeable asset universe (replaces the Nasdaq FTP HTTP call)
  - historical daily price bars (replaces yf.download())
  - paper/live order submission

The public interface deliberately mirrors YahooDBReader's shape so that
BobsBrain's pair-discovery code requires minimal changes. Used in both
backtest mode (pair discovery data) and paper mode (pair discovery + orders).

In backtest mode, submit_order() is a no-op since order execution is handled
by Lumibot's YahooDataBacktesting simulation engine.
"""

from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus, OrderSide, TimeInForce
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest


class AlpacaClient:
    """
    Boundary between LumiBob and the outside world.

    Instantiate once and pass to StockDataCache. The same instance can be
    used for both data fetching and order submission — the trading client
    is only exercised when submit_order() is called (paper/live mode).
    """

    # Exchanges to include in the tradeable universe
    _TARGET_EXCHANGES = {AssetExchange.NYSE, AssetExchange.NASDAQ}

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        mode: Literal["backtest", "paper"] = "backtest",
    ):
        self._mode = mode
        self._data_client = StockHistoricalDataClient(api_key, secret_key)
        self._trading_client = TradingClient(api_key, secret_key, paper=paper)

    # ------------------------------------------------------------------
    # Asset universe
    # ------------------------------------------------------------------

    def get_tradeable_assets(self) -> list[str]:
        """
        Return active, tradeable NYSE and Nasdaq common-share symbols.

        Filters applied on top of the Alpaca-active + tradable baseline:
        - Symbols containing '.' are non-common-share instruments on Alpaca
          (warrants, preferred series, rights, units, e.g. ACHR.WS, BRK.B).
        - Assets whose name contains 'ETF' are exchange-traded funds, which
          don't exhibit the lead/lag equity dynamics the strategy targets.

        Results should be stored in the tickers table via DatabaseClient and
        refreshed nightly rather than called on every before_market_opens().
        After changing these filters, clear the tickers table so the cache is
        rebuilt on the next run (see DatabaseClient.clear_tickers).
        """
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        assets = self._trading_client.get_all_assets(request)
        return [
            a.symbol
            for a in assets
            if a.exchange in self._TARGET_EXCHANGES
            and a.tradable
            and '.' not in a.symbol
            and 'ETF' not in (a.name or '').upper()
        ]

    # ------------------------------------------------------------------
    # Historical price data
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for a list of symbols over a date range.

        Returns a DataFrame with a timezone-naive DatetimeIndex and symbol
        columns containing close prices — matching the shape that
        yf.download() produces so callers need no conversion.

        Empty symbols or date ranges that return no data are silently dropped.
        """
        if not symbols:
            return pd.DataFrame()

        start_utc = _ensure_utc(start)
        end_utc = _ensure_utc(end)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_utc,
            end=end_utc,
            adjustment="all",
        )
        bars = self._data_client.get_stock_bars(request)
        df = bars.df  # MultiIndex: (symbol, timestamp)

        if df.empty:
            return pd.DataFrame()

        # Pivot to: DatetimeIndex (timezone-naive) × symbol columns, values=close
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return df.pivot_table(
            index="timestamp", columns="symbol", values="close", aggfunc="last"
        )

    def get_ohlcv_records(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """
        Like get_historical_bars() but returns a flat list of dicts with full
        OHLCV data, suitable for DatabaseClient.upsert_ohlcv().
        """
        if not symbols:
            return []

        start_utc = _ensure_utc(start)
        end_utc = _ensure_utc(end)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_utc,
            end=end_utc,
            adjustment="all",
        )
        bars = self._data_client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return []

        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        records = []
        for _, row in df.iterrows():
            records.append({
                "time":   row["timestamp"],
                "symbol": row["symbol"],
                "open":   row.get("open"),
                "high":   row.get("high"),
                "low":    row.get("low"),
                "close":  row["close"],
                "volume": row.get("volume"),
            })
        return records

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(self, symbol: str, qty: int, side: str) -> None:
        """
        Submit a market order via the Alpaca trading client.

        In backtest mode this is a no-op — order execution is handled by
        Lumibot's simulation engine (BobsBrain.create_order / submit_order).
        In paper mode this sends the order to Alpaca's paper trading endpoint.
        """
        if self._mode == "backtest":
            return

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        self._trading_client.submit_order(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_utc(dt: datetime) -> datetime:
    """Return dt with UTC timezone, attaching it if naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
