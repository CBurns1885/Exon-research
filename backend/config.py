"""Configuration management for the African Economy Model."""
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

class Config:
    """Application configuration."""

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./african_economy.db")

    # Countries
    COUNTRIES = {
        "KE": {
            "name": "Kenya",
            "currency": "KES",
            "region": "East Africa",
            "emoji": "🇰🇪"
        },
        "NG": {
            "name": "Nigeria",
            "currency": "NGN",
            "region": "West Africa",
            "emoji": "🇳🇬"
        },
        "GH": {
            "name": "Ghana",
            "currency": "GHS",
            "region": "West Africa",
            "emoji": "🇬🇭"
        },
        "RW": {
            "name": "Rwanda",
            "currency": "RWF",
            "region": "East Africa",
            "emoji": "🇷🇼"
        },
        "EG": {
            "name": "Egypt",
            "currency": "EGP",
            "region": "North Africa",
            "emoji": "🇪🇬"
        }
    }

    # Forecasting parameters
    FORECAST_MONTHS = int(os.getenv("FORECAST_MONTHS", 6))
    HISTORICAL_YEARS = int(os.getenv("HISTORICAL_YEARS", 5))

    # Economic indicators
    INDICATORS = [
        "interest_rate",
        "fx_rate",
        "capital_inflow",
        "capital_outflow",
        "imports",
        "exports",
        "gdp_growth"
    ]

    @classmethod
    def get_country_codes(cls) -> List[str]:
        """Get list of country codes."""
        return list(cls.COUNTRIES.keys())

    @classmethod
    def get_country_name(cls, code: str) -> str:
        """Get country name from code."""
        return cls.COUNTRIES.get(code, {}).get("name", "Unknown")

    @classmethod
    def get_currency(cls, code: str) -> str:
        """Get currency code for country."""
        return cls.COUNTRIES.get(code, {}).get("currency", "USD")

config = Config()
