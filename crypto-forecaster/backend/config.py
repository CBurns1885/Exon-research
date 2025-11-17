"""Configuration for Crypto Forecasting System."""
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


class CryptoConfig:
    """Configuration for cryptocurrency forecasting."""

    # API Configuration
    API_HOST = os.getenv("CRYPTO_API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("CRYPTO_API_PORT", 8001))

    # Database
    DATABASE_URL = os.getenv("CRYPTO_DATABASE_URL", "sqlite:///./crypto_forecaster.db")

    # API Keys
    COINBASE_API_KEY = os.getenv("COINBASE_API_KEY", "")
    COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET", "")
    COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")

    # Top 10 Cryptocurrencies by Market Cap (non-stablecoins)
    CRYPTOCURRENCIES = {
        "BTC": {
            "name": "Bitcoin",
            "symbol": "BTC",
            "emoji": "🪙",
            "color": "#F7931A",
            "coingecko_id": "bitcoin",
            "coinbase_pair": "BTC-USD"
        },
        "ETH": {
            "name": "Ethereum",
            "symbol": "ETH",
            "emoji": "💎",
            "color": "#627EEA",
            "coingecko_id": "ethereum",
            "coinbase_pair": "ETH-USD"
        },
        "BNB": {
            "name": "BNB",
            "symbol": "BNB",
            "emoji": "⚡",
            "color": "#F3BA2F",
            "coingecko_id": "binancecoin",
            "coinbase_pair": "BNB-USD"
        },
        "SOL": {
            "name": "Solana",
            "symbol": "SOL",
            "emoji": "☀️",
            "color": "#00FFA3",
            "coingecko_id": "solana",
            "coinbase_pair": "SOL-USD"
        },
        "XRP": {
            "name": "XRP",
            "symbol": "XRP",
            "emoji": "💧",
            "color": "#23292F",
            "coingecko_id": "ripple",
            "coinbase_pair": "XRP-USD"
        },
        "ADA": {
            "name": "Cardano",
            "symbol": "ADA",
            "emoji": "🔷",
            "color": "#0033AD",
            "coingecko_id": "cardano",
            "coinbase_pair": "ADA-USD"
        },
        "DOGE": {
            "name": "Dogecoin",
            "symbol": "DOGE",
            "emoji": "🐕",
            "color": "#C2A633",
            "coingecko_id": "dogecoin",
            "coinbase_pair": "DOGE-USD"
        },
        "MATIC": {
            "name": "Polygon",
            "symbol": "MATIC",
            "emoji": "🟣",
            "color": "#8247E5",
            "coingecko_id": "matic-network",
            "coinbase_pair": "MATIC-USD"
        },
        "DOT": {
            "name": "Polkadot",
            "symbol": "DOT",
            "emoji": "⚫",
            "color": "#E6007A",
            "coingecko_id": "polkadot",
            "coinbase_pair": "DOT-USD"
        },
        "LINK": {
            "name": "Chainlink",
            "symbol": "LINK",
            "emoji": "🔗",
            "color": "#2A5ADA",
            "coingecko_id": "chainlink",
            "coinbase_pair": "LINK-USD"
        }
    }

    # Forecasting parameters
    FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", 30))
    HISTORICAL_DAYS = int(os.getenv("HISTORICAL_DAYS", 365))  # 1 year

    # Data refresh intervals (in seconds)
    PRICE_UPDATE_INTERVAL = int(os.getenv("PRICE_UPDATE_INTERVAL", 300))  # 5 minutes
    FORECAST_UPDATE_INTERVAL = int(os.getenv("FORECAST_UPDATE_INTERVAL", 3600))  # 1 hour

    # Technical indicators
    TECHNICAL_INDICATORS = [
        "sma_7",    # 7-day Simple Moving Average
        "sma_30",   # 30-day Simple Moving Average
        "ema_12",   # 12-day Exponential Moving Average
        "ema_26",   # 26-day Exponential Moving Average
        "rsi",      # Relative Strength Index
        "macd",     # Moving Average Convergence Divergence
        "bollinger_high",  # Bollinger Band High
        "bollinger_low",   # Bollinger Band Low
        "volume_sma",      # Volume Simple Moving Average
    ]

    @classmethod
    def get_crypto_symbols(cls) -> List[str]:
        """Get list of cryptocurrency symbols."""
        return list(cls.CRYPTOCURRENCIES.keys())

    @classmethod
    def get_crypto_info(cls, symbol: str) -> Dict:
        """Get cryptocurrency information."""
        return cls.CRYPTOCURRENCIES.get(symbol.upper(), {})

    @classmethod
    def get_crypto_name(cls, symbol: str) -> str:
        """Get cryptocurrency name."""
        return cls.CRYPTOCURRENCIES.get(symbol.upper(), {}).get("name", "Unknown")

    @classmethod
    def get_coinbase_pair(cls, symbol: str) -> str:
        """Get Coinbase trading pair."""
        return cls.CRYPTOCURRENCIES.get(symbol.upper(), {}).get("coinbase_pair", f"{symbol}-USD")

    @classmethod
    def get_coingecko_id(cls, symbol: str) -> str:
        """Get CoinGecko ID."""
        return cls.CRYPTOCURRENCIES.get(symbol.upper(), {}).get("coingecko_id", "")


config = CryptoConfig()
