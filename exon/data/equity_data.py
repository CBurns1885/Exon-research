"""Equities market data pipeline — fetch, normalise, cache, and serve OHLCV data.

Designed for US equities via the Alpaca API. Handles market hours,
corporate actions awareness, and multi-timeframe data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# US market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16

# Annualisation constants for equities
TRADING_DAYS_PER_YEAR = 252
BARS_PER_DAY = {"1Min": 390, "5Min": 78, "15Min": 26, "30Min": 13, "1Hour": 7, "1Day": 1}


@dataclass
class EquityDataPipeline:
    """Fetches and caches OHLCV data for US equities via Alpaca."""

    client: object  # AlpacaClient instance
    cache_dir: Path = field(default_factory=lambda: Path("data"))

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_bars(
        self,
        symbols: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for multiple symbols into a wide close-price DataFrame.

        Parameters
        ----------
        symbols : list of ticker symbols, e.g. ["AAPL", "MSFT", "GOOGL"]
        start : start date
        end : end date (defaults to now)
        timeframe : Alpaca timeframe string

        Returns
        -------
        DataFrame with DatetimeIndex and one column per symbol (close prices)
        """
        try:
            raw = self.client.get_bars(symbols, str(start), str(end) if end else None, timeframe)
        except Exception as e:
            logger.error("Failed to fetch bars: %s", e)
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        return self._to_wide(raw, field="close")

    def fetch_ohlcv(
        self,
        symbols: list[str],
        start: str,
        end: str | None = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch full OHLCV data (multi-index: symbol × timestamp)."""
        try:
            raw = self.client.get_bars(symbols, start, end, timeframe)
        except Exception as e:
            logger.error("Failed to fetch OHLCV: %s", e)
            return pd.DataFrame()
        return raw

    def fetch_universe(
        self,
        symbols: list[str],
        start: str,
        end: str | None = None,
        timeframe: str = "1Day",
        data_field: str = "close",
    ) -> pd.DataFrame:
        """Fetch a single price field for multiple symbols into a wide DataFrame.

        Returns DataFrame with DatetimeIndex and one column per symbol.
        """
        try:
            raw = self.client.get_bars(symbols, start, end, timeframe)
        except Exception as e:
            logger.error("Failed to fetch universe data: %s", e)
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        return self._to_wide(raw, field=data_field)

    def get_returns(
        self,
        prices: pd.DataFrame,
        method: str = "log",
    ) -> pd.DataFrame:
        """Compute returns from a price DataFrame."""
        if method == "log":
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    def get_volume(
        self,
        symbols: list[str],
        start: str,
        end: str | None = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch volume data as a wide DataFrame."""
        try:
            raw = self.client.get_bars(symbols, start, end, timeframe)
        except Exception as e:
            logger.error("Failed to fetch volume: %s", e)
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        return self._to_wide(raw, field="volume")

    # -- Caching -----------------------------------------------------------

    def save(self, df: pd.DataFrame, name: str) -> Path:
        path = self.cache_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.info("Saved %s (%d rows)", path, len(df))
        return path

    def load(self, name: str) -> pd.DataFrame:
        path = self.cache_dir / f"{name}.parquet"
        return pd.read_parquet(path)

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _to_wide(bars_df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
        """Convert Alpaca multi-index (symbol, timestamp) bars to wide format."""
        if bars_df.empty:
            return pd.DataFrame()

        # Alpaca returns MultiIndex (symbol, timestamp)
        if isinstance(bars_df.index, pd.MultiIndex):
            wide = bars_df[field].unstack(level=0)
        else:
            # Single symbol — just return the column
            wide = bars_df[[field]]

        wide = wide.sort_index().ffill().dropna(how="all")
        return wide
