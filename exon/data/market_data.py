"""High-level market data pipeline: fetch, normalise, cache, and serve OHLCV data."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .coinbase_client import CoinbaseClient

logger = logging.getLogger(__name__)

GRANULARITY_SECONDS = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600,
    "TWO_HOUR": 7200,
    "SIX_HOUR": 21600,
    "ONE_DAY": 86400,
}

MAX_CANDLES_PER_REQUEST = 300


@dataclass
class MarketDataPipeline:
    """Fetches and caches OHLCV data from Coinbase."""

    client: CoinbaseClient
    cache_dir: Path = field(default_factory=lambda: Path("data"))

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public API --------------------------------------------------------

    def fetch_candles(
        self,
        product_id: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        granularity: str = "ONE_HOUR",
    ) -> pd.DataFrame:
        """Fetch OHLCV candles, paginating automatically.

        Parameters
        ----------
        product_id : e.g. "BTC-USD"
        start, end : anything pandas.Timestamp can parse
        granularity : Coinbase granularity string
        """
        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())
        step = GRANULARITY_SECONDS[granularity]

        all_candles: list[dict] = []
        cursor = start_ts

        while cursor < end_ts:
            batch_end = min(cursor + step * MAX_CANDLES_PER_REQUEST, end_ts)
            try:
                candles = self.client.get_candles(product_id, cursor, batch_end, granularity)
                all_candles.extend(candles)
            except Exception:
                logger.warning("Failed batch %s %d->%d, skipping", product_id, cursor, batch_end)
            cursor = batch_end

        if not all_candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        return self._normalise(all_candles, product_id)

    def fetch_universe(
        self,
        symbols: list[str],
        start: str,
        end: str,
        granularity: str = "ONE_HOUR",
        field: str = "close",
    ) -> pd.DataFrame:
        """Fetch a single price field for multiple symbols into a wide DataFrame.

        Returns DataFrame with DatetimeIndex and one column per symbol.
        """
        frames: dict[str, pd.Series] = {}
        for sym in symbols:
            logger.info("Fetching %s", sym)
            df = self.fetch_candles(sym, start, end, granularity)
            if not df.empty:
                frames[sym] = df[field]
        if not frames:
            return pd.DataFrame()
        combined = pd.DataFrame(frames).sort_index()
        combined = combined.ffill().dropna(how="all")
        return combined

    def get_returns(
        self,
        prices: pd.DataFrame,
        method: str = "log",
    ) -> pd.DataFrame:
        """Compute returns from a price DataFrame."""
        if method == "log":
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    # -- caching -----------------------------------------------------------

    def save(self, df: pd.DataFrame, name: str) -> Path:
        path = self.cache_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.info("Saved %s (%d rows)", path, len(df))
        return path

    def load(self, name: str) -> pd.DataFrame:
        path = self.cache_dir / f"{name}.parquet"
        return pd.read_parquet(path)

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _normalise(raw_candles: list[dict], product_id: str) -> pd.DataFrame:
        rows = []
        for c in raw_candles:
            rows.append(
                {
                    "timestamp": int(c["start"]),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c["volume"]),
                }
            )
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df.attrs["product_id"] = product_id
        return df
