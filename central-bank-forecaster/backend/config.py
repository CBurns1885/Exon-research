"""Configuration for Central Bank Rate Forecaster."""
import os
from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime

load_dotenv()


class CentralBankConfig:
    """Configuration for central bank forecasting."""

    # API Configuration
    API_HOST = os.getenv("CB_API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("CB_API_PORT", 8002))

    # Database
    DATABASE_URL = os.getenv("CB_DATABASE_URL", "sqlite:///./central_banks.db")

    # LLM API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # Which LLM to use: "openai", "anthropic", or "local"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")

    # Economic data APIs
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # St. Louis Fed FRED API

    # Central Banks Configuration
    CENTRAL_BANKS = {
        "FED": {
            "name": "Federal Reserve",
            "country": "United States",
            "currency": "USD",
            "emoji": "🇺🇸",
            "website": "https://www.federalreserve.gov",
            "speeches_url": "https://www.federalreserve.gov/newsevents/speeches.htm",
            "current_chair": "Jerome Powell",
            "meeting_frequency": "8 times per year",
            "target_inflation": 2.0,
            "calendar_url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        },
        "ECB": {
            "name": "European Central Bank",
            "country": "Eurozone",
            "currency": "EUR",
            "emoji": "🇪🇺",
            "website": "https://www.ecb.europa.eu",
            "speeches_url": "https://www.ecb.europa.eu/press/key/html/index.en.html",
            "current_chair": "Christine Lagarde",
            "meeting_frequency": "6 weeks",
            "target_inflation": 2.0,
            "calendar_url": "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
        },
        "BOE": {
            "name": "Bank of England",
            "country": "United Kingdom",
            "currency": "GBP",
            "emoji": "🇬🇧",
            "website": "https://www.bankofengland.co.uk",
            "speeches_url": "https://www.bankofengland.co.uk/news/speeches",
            "current_chair": "Andrew Bailey",
            "meeting_frequency": "8 times per year",
            "target_inflation": 2.0,
            "calendar_url": "https://www.bankofengland.co.uk/monetary-policy/mpc-meeting-calendar"
        },
        "BOJ": {
            "name": "Bank of Japan",
            "country": "Japan",
            "currency": "JPY",
            "emoji": "🇯🇵",
            "website": "https://www.boj.or.jp/en",
            "speeches_url": "https://www.boj.or.jp/en/about/press/index.htm",
            "current_chair": "Kazuo Ueda",
            "meeting_frequency": "8 times per year",
            "target_inflation": 2.0,
            "calendar_url": "https://www.boj.or.jp/en/mopo/mpmdeci/index.htm"
        },
        "BOC": {
            "name": "Bank of Canada",
            "country": "Canada",
            "currency": "CAD",
            "emoji": "🇨🇦",
            "website": "https://www.bankofcanada.ca",
            "speeches_url": "https://www.bankofcanada.ca/media-type/speeches-and-appearances/",
            "current_chair": "Tiff Macklem",
            "meeting_frequency": "8 times per year",
            "target_inflation": 2.0,
            "calendar_url": "https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/"
        },
        "RBA": {
            "name": "Reserve Bank of Australia",
            "country": "Australia",
            "currency": "AUD",
            "emoji": "🇦🇺",
            "website": "https://www.rba.gov.au",
            "speeches_url": "https://www.rba.gov.au/speeches/",
            "current_chair": "Michele Bullock",
            "meeting_frequency": "11 times per year",
            "target_inflation": 2.5,
            "calendar_url": "https://www.rba.gov.au/monetary-policy/rba-board-minutes/"
        },
        "SNB": {
            "name": "Swiss National Bank",
            "country": "Switzerland",
            "currency": "CHF",
            "emoji": "🇨🇭",
            "website": "https://www.snb.ch",
            "speeches_url": "https://www.snb.ch/en/the-snb/communication/speeches",
            "current_chair": "Thomas Jordan",
            "meeting_frequency": "4 times per year",
            "target_inflation": 2.0,
            "calendar_url": "https://www.snb.ch/en/the-snb/mandates-goals/monetary-policy"
        },
        "RBNZ": {
            "name": "Reserve Bank of New Zealand",
            "country": "New Zealand",
            "currency": "NZD",
            "emoji": "🇳🇿",
            "website": "https://www.rbnz.govt.nz",
            "speeches_url": "https://www.rbnz.govt.nz/hub/news?category=Speech",
            "current_chair": "Adrian Orr",
            "meeting_frequency": "7 times per year",
            "target_inflation": 2.0,
            "calendar_url": "https://www.rbnz.govt.nz/monetary-policy/official-cash-rate-decisions"
        }
    }

    # Forecasting parameters
    FORECAST_MONTHS = [3, 6, 12]  # Forecast horizons in months

    # Scraping settings
    SCRAPE_INTERVAL_HOURS = 24  # How often to scrape for new speeches
    MAX_SPEECHES_PER_SCRAPE = 20

    # NLP settings
    SENTIMENT_LABELS = ["very_dovish", "dovish", "neutral", "hawkish", "very_hawkish"]

    # Economic indicators to track
    ECONOMIC_INDICATORS = {
        "inflation": ["CPI", "PCE", "Core_CPI", "Core_PCE"],
        "employment": ["Unemployment_Rate", "Nonfarm_Payrolls", "Labor_Force_Participation"],
        "growth": ["GDP_Growth", "GDP_QoQ", "GDP_YoY"],
        "sentiment": ["Consumer_Confidence", "Manufacturing_PMI", "Services_PMI"],
    }

    @classmethod
    def get_bank_codes(cls) -> List[str]:
        """Get list of central bank codes."""
        return list(cls.CENTRAL_BANKS.keys())

    @classmethod
    def get_bank_info(cls, code: str) -> Dict:
        """Get central bank information."""
        return cls.CENTRAL_BANKS.get(code.upper(), {})

    @classmethod
    def get_bank_name(cls, code: str) -> str:
        """Get central bank name."""
        return cls.CENTRAL_BANKS.get(code.upper(), {}).get("name", "Unknown")


config = CentralBankConfig()
