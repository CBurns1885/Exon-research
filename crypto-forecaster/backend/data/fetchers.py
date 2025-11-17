"""Data fetchers for cryptocurrency prices and market data."""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from bs4 import BeautifulSoup
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoinGeckoFetcher:
    """Fetch data from CoinGecko API (free, no API key required)."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize CoinGecko fetcher."""
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["X-Cg-Pro-Api-Key"] = api_key

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with retry logic."""
        url = f"{self.BASE_URL}/{endpoint}"

        for attempt in range(3):
            try:
                response = requests.get(url, params=params, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def get_current_price(self, symbol: str) -> Dict:
        """Get current price and market data for a cryptocurrency."""
        try:
            coin_id = config.get_coingecko_id(symbol)
            if not coin_id:
                logger.error(f"No CoinGecko ID for {symbol}")
                return {}

            data = self._make_request(
                f"coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "true",
                    "community_data": "false",
                    "developer_data": "false",
                }
            )

            market_data = data.get("market_data", {})

            return {
                "symbol": symbol,
                "name": data.get("name", ""),
                "current_price": market_data.get("current_price", {}).get("usd"),
                "market_cap": market_data.get("market_cap", {}).get("usd"),
                "market_cap_rank": data.get("market_cap_rank"),
                "total_volume": market_data.get("total_volume", {}).get("usd"),
                "price_change_24h": market_data.get("price_change_24h"),
                "price_change_percentage_24h": market_data.get("price_change_percentage_24h"),
                "price_change_percentage_7d": market_data.get("price_change_percentage_7d"),
                "price_change_percentage_30d": market_data.get("price_change_percentage_30d"),
                "ath": market_data.get("ath", {}).get("usd"),
                "ath_date": market_data.get("ath_date", {}).get("usd"),
                "atl": market_data.get("atl", {}).get("usd"),
                "atl_date": market_data.get("atl_date", {}).get("usd"),
            }
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return {}

    def get_historical_prices(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Get historical OHLC data."""
        try:
            coin_id = config.get_coingecko_id(symbol)
            if not coin_id:
                logger.error(f"No CoinGecko ID for {symbol}")
                return pd.DataFrame()

            data = self._make_request(
                f"coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": days}
            )

            if not data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["date"] = df["timestamp"].dt.date
            df["symbol"] = symbol

            # Add volume (approximate based on market activity)
            # CoinGecko OHLC doesn't include volume, so we'll fetch it separately
            df["volume"] = 0.0

            return df

        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {e}")
            return pd.DataFrame()

    def get_historical_market_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Get historical market data including prices and volume."""
        try:
            coin_id = config.get_coingecko_id(symbol)
            if not coin_id:
                return pd.DataFrame()

            data = self._make_request(
                f"coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": days, "interval": "daily"}
            )

            if not data:
                return pd.DataFrame()

            # Extract prices, market caps, and volumes
            prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
            volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])

            # Merge data
            df = prices.merge(market_caps, on="timestamp").merge(volumes, on="timestamp")

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["date"] = df["timestamp"].dt.date
            df["symbol"] = symbol

            # Create OHLC (using price as close, approximate open/high/low)
            df["close"] = df["price"]
            df["open"] = df["close"].shift(1).fillna(df["close"])
            df["high"] = df["close"] * 1.02  # Approximate
            df["low"] = df["close"] * 0.98   # Approximate

            return df[["symbol", "timestamp", "date", "open", "high", "low", "close", "volume", "market_cap"]]

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()


class CoinbaseFetcher:
    """Fetch data from Coinbase (using public endpoints)."""

    BASE_URL = "https://api.coinbase.com/v2"

    def get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price from Coinbase."""
        try:
            pair = config.get_coinbase_pair(symbol)
            url = f"{self.BASE_URL}/prices/{pair}/spot"

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            return float(data["data"]["amount"])

        except Exception as e:
            logger.error(f"Error fetching Coinbase spot price for {symbol}: {e}")
            return None


class WebScraperFetcher:
    """Web scraper for cryptocurrency data as backup."""

    def scrape_coinmarketcap_price(self, symbol: str) -> Optional[Dict]:
        """Scrape price from CoinMarketCap."""
        try:
            crypto_info = config.get_crypto_info(symbol)
            crypto_name = crypto_info.get("name", "").lower()

            # CoinMarketCap URL
            url = f"https://coinmarketcap.com/currencies/{crypto_name}/"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Try to find price (this is fragile and may need updates)
            price_element = soup.find("span", {"class": "sc-65e7f566-0"})

            if price_element:
                price_text = price_element.text.strip()
                # Remove $ and commas
                price = float(price_text.replace("$", "").replace(",", ""))

                return {
                    "symbol": symbol,
                    "price": price,
                    "source": "coinmarketcap"
                }

        except Exception as e:
            logger.error(f"Error scraping price for {symbol}: {e}")

        return None


class CryptoDataFetcher:
    """Main data fetcher that combines multiple sources."""

    def __init__(self):
        """Initialize all fetchers."""
        self.coingecko = CoinGeckoFetcher(api_key=config.COINGECKO_API_KEY)
        self.coinbase = CoinbaseFetcher()
        self.scraper = WebScraperFetcher()

    def get_current_prices_all(self) -> Dict[str, Dict]:
        """Get current prices for all tracked cryptocurrencies."""
        results = {}

        for symbol in config.get_crypto_symbols():
            logger.info(f"Fetching current price for {symbol}...")

            # Try CoinGecko first (most reliable)
            data = self.coingecko.get_current_price(symbol)

            if data and data.get("current_price"):
                results[symbol] = data
            else:
                # Fallback to Coinbase
                coinbase_price = self.coinbase.get_spot_price(symbol)
                if coinbase_price:
                    results[symbol] = {
                        "symbol": symbol,
                        "current_price": coinbase_price,
                        "source": "coinbase"
                    }

            # Rate limiting
            time.sleep(1.5)

        return results

    def get_historical_data_all(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """Get historical data for all cryptocurrencies."""
        results = {}

        for symbol in config.get_crypto_symbols():
            logger.info(f"Fetching historical data for {symbol}...")

            df = self.coingecko.get_historical_market_data(symbol, days=days)

            if not df.empty:
                results[symbol] = df
            else:
                logger.warning(f"No historical data for {symbol}")

            # Rate limiting
            time.sleep(2)

        return results

    def get_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Get historical data for a single cryptocurrency."""
        return self.coingecko.get_historical_market_data(symbol, days=days)
