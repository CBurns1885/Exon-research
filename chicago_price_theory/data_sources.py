"""
Data Sources Module - Real-world Economic Data Integration

Fetches data from various APIs:
- FRED (Federal Reserve Economic Data) - US economic indicators
- BLS (Bureau of Labor Statistics) - Employment, wages, prices
- World Bank - International development data
- Census Bureau - Demographics, income

All API integrations are OPTIONAL - models work without external data.
"""

import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings


class FREDDataFetcher:
    """
    Fetch data from Federal Reserve Economic Data (FRED).

    Free API key available at: https://fred.stlouisfed.org/docs/api/api_key.html

    Popular series:
    - UNRATE: Unemployment Rate
    - CPIAUCSL: Consumer Price Index
    - GDP: Gross Domestic Product
    - AHETPI: Average Hourly Earnings
    - PAYEMS: Total Nonfarm Payroll Employment
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED data fetcher.

        Parameters:
        -----------
        api_key : str, optional
            FRED API key. If None, will try to read from environment or config.
        """
        self.api_key = api_key or self._get_api_key()

    def _get_api_key(self) -> Optional[str]:
        """Try to get API key from environment or config file."""
        import os

        # Try environment variable
        key = os.environ.get('FRED_API_KEY')
        if key:
            return key

        # Try config file
        try:
            with open('chicago_price_theory/config.json', 'r') as f:
                config = json.load(f)
                return config.get('fred_api_key')
        except:
            pass

        return None

    def fetch_series(self,
                     series_id: str,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch time series data from FRED.

        Parameters:
        -----------
        series_id : str
            FRED series identifier (e.g., 'UNRATE', 'GDP')
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format

        Returns:
        --------
        DataFrame with date and value columns
        """
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Get one free at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        # Default to last 10 years if no dates specified
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')

        url = f"{self.BASE_URL}/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'observations' not in data:
                raise ValueError(f"No data found for series {series_id}")

            # Convert to DataFrame
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])

            return df[['date', 'value']]

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch FRED data: {str(e)}")

    def get_latest_value(self, series_id: str) -> float:
        """Get most recent value for a series."""
        df = self.fetch_series(series_id)
        return df['value'].iloc[-1]

    def get_average(self, series_id: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> float:
        """Get average value over a period."""
        df = self.fetch_series(series_id, start_date, end_date)
        return df['value'].mean()


class BLSDataFetcher:
    """
    Fetch data from Bureau of Labor Statistics.

    No API key required for basic usage (limited to 25 queries/day).
    Register for key at: https://data.bls.gov/registrationEngine/

    Popular series:
    - LNS14000000: Unemployment Rate
    - CES0500000003: Average Hourly Earnings
    - CUUR0000SA0: Consumer Price Index
    """

    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize BLS data fetcher."""
        self.api_key = api_key

    def fetch_series(self,
                     series_id: str,
                     start_year: Optional[int] = None,
                     end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch BLS time series data.

        Parameters:
        -----------
        series_id : str
            BLS series identifier
        start_year : int, optional
            Start year (default: 10 years ago)
        end_year : int, optional
            End year (default: current year)

        Returns:
        --------
        DataFrame with date and value columns
        """
        if not end_year:
            end_year = datetime.now().year
        if not start_year:
            start_year = end_year - 10

        headers = {'Content-type': 'application/json'}
        data = json.dumps({
            "seriesid": [series_id],
            "startyear": str(start_year),
            "endyear": str(end_year)
        })

        if self.api_key:
            data = json.dumps({
                "seriesid": [series_id],
                "startyear": str(start_year),
                "endyear": str(end_year),
                "registrationkey": self.api_key
            })

        try:
            response = requests.post(self.BASE_URL, data=data, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result['status'] != 'REQUEST_SUCCEEDED':
                raise ValueError(f"BLS request failed: {result.get('message', 'Unknown error')}")

            # Parse data
            series_data = result['Results']['series'][0]['data']

            df = pd.DataFrame(series_data)
            df['date'] = pd.to_datetime(df['year'] + '-' + df['period'].str.replace('M', ''))
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            df = df.sort_values('date')

            return df[['date', 'value']]

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch BLS data: {str(e)}")


class WorldBankDataFetcher:
    """
    Fetch data from World Bank Open Data API.

    No API key required.

    Popular indicators:
    - NY.GDP.PCAP.CD: GDP per capita
    - SE.ADT.LITR.ZS: Literacy rate
    - SL.UEM.TOTL.ZS: Unemployment rate
    - SP.POP.TOTL: Total population
    """

    BASE_URL = "https://api.worldbank.org/v2"

    def fetch_indicator(self,
                       indicator: str,
                       country: str = 'US',
                       start_year: Optional[int] = None,
                       end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch World Bank indicator data.

        Parameters:
        -----------
        indicator : str
            World Bank indicator code
        country : str
            ISO country code (default: 'US')
        start_year : int, optional
            Start year
        end_year : int, optional
            End year

        Returns:
        --------
        DataFrame with year and value columns
        """
        if not end_year:
            end_year = datetime.now().year
        if not start_year:
            start_year = end_year - 20

        url = f"{self.BASE_URL}/country/{country}/indicator/{indicator}"
        params = {
            'format': 'json',
            'date': f'{start_year}:{end_year}',
            'per_page': 1000
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if len(data) < 2 or not data[1]:
                raise ValueError(f"No data found for indicator {indicator}")

            # Parse data
            records = []
            for item in data[1]:
                if item['value'] is not None:
                    records.append({
                        'year': int(item['date']),
                        'value': float(item['value'])
                    })

            df = pd.DataFrame(records)
            df = df.sort_values('year')

            return df

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch World Bank data: {str(e)}")


class EconomicDataAggregator:
    """
    Aggregate data from multiple sources for economic modeling.

    Provides convenience methods for common economic indicators
    needed by Chicago price theory models.
    """

    def __init__(self, fred_api_key: Optional[str] = None,
                 bls_api_key: Optional[str] = None):
        """Initialize with API credentials."""
        self.fred = FREDDataFetcher(fred_api_key) if fred_api_key else None
        self.bls = BLSDataFetcher(bls_api_key) if bls_api_key else None
        self.worldbank = WorldBankDataFetcher()

    def get_unemployment_rate(self, source: str = 'fred') -> float:
        """
        Get current unemployment rate.

        Parameters:
        -----------
        source : str
            'fred' or 'bls'

        Returns:
        --------
        Unemployment rate as percentage
        """
        if source == 'fred' and self.fred:
            return self.fred.get_latest_value('UNRATE')
        elif source == 'bls' and self.bls:
            df = self.bls.fetch_series('LNS14000000')
            return df['value'].iloc[-1]
        else:
            raise ValueError(f"Source '{source}' not available or not configured")

    def get_average_wage(self, industry: str = 'total_private') -> float:
        """
        Get average hourly earnings.

        Parameters:
        -----------
        industry : str
            Industry code (default: 'total_private')

        Returns:
        --------
        Average hourly wage in dollars
        """
        if not self.fred:
            raise ValueError("FRED API key required for wage data")

        # FRED series for average hourly earnings
        series_map = {
            'total_private': 'CES0500000003',
            'manufacturing': 'CES3000000003',
            'construction': 'CES2000000003',
            'retail': 'CES4200000003'
        }

        series_id = series_map.get(industry, 'CES0500000003')
        return self.fred.get_latest_value(series_id)

    def get_wage_by_education(self) -> Dict[str, float]:
        """
        Get median weekly earnings by education level.

        Returns:
        --------
        Dictionary mapping education level to earnings
        """
        # BLS typically publishes this quarterly
        # Using approximate values based on recent data
        # In production, would fetch from BLS CPS tables

        warnings.warn(
            "Education-wage data uses approximate values. "
            "For precise data, query BLS Current Population Survey directly."
        )

        return {
            'less_than_hs': 626,      # Less than high school
            'high_school': 809,        # High school diploma
            'some_college': 899,       # Some college, no degree
            'associates': 963,         # Associate degree
            'bachelors': 1334,         # Bachelor's degree
            'masters': 1574,           # Master's degree
            'professional': 1924,      # Professional degree
            'doctorate': 1909          # Doctoral degree
        }

    def get_cpi(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get Consumer Price Index data.

        Returns:
        --------
        DataFrame with CPI values over time
        """
        if not self.fred:
            raise ValueError("FRED API key required for CPI data")

        return self.fred.fetch_series('CPIAUCSL', start_date=start_date)

    def get_gdp(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get GDP data.

        Returns:
        --------
        DataFrame with GDP values over time
        """
        if not self.fred:
            raise ValueError("FRED API key required for GDP data")

        return self.fred.fetch_series('GDP', start_date=start_date)

    def get_labor_force_participation(self) -> float:
        """Get current labor force participation rate."""
        if not self.fred:
            raise ValueError("FRED API key required")

        return self.fred.get_latest_value('CIVPART')


# Convenience function for quick data access
def get_economic_data(indicator: str,
                     source: str = 'fred',
                     api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Quick data fetch for common economic indicators.

    Parameters:
    -----------
    indicator : str
        Economic indicator name
    source : str
        Data source ('fred', 'bls', 'worldbank')
    api_key : str, optional
        API key if required

    Returns:
    --------
    DataFrame with time series data

    Example:
    --------
    >>> df = get_economic_data('unemployment', source='fred', api_key='your_key')
    >>> print(df.tail())
    """
    indicator_map = {
        'unemployment': {'fred': 'UNRATE', 'bls': 'LNS14000000'},
        'wages': {'fred': 'CES0500000003'},
        'cpi': {'fred': 'CPIAUCSL', 'bls': 'CUUR0000SA0'},
        'gdp': {'fred': 'GDP'},
        'employment': {'fred': 'PAYEMS'}
    }

    if indicator not in indicator_map:
        raise ValueError(f"Unknown indicator: {indicator}")

    if source not in indicator_map[indicator]:
        raise ValueError(f"Indicator '{indicator}' not available from {source}")

    series_id = indicator_map[indicator][source]

    if source == 'fred':
        fetcher = FREDDataFetcher(api_key)
        return fetcher.fetch_series(series_id)
    elif source == 'bls':
        fetcher = BLSDataFetcher(api_key)
        return fetcher.fetch_series(series_id)
    else:
        raise ValueError(f"Source '{source}' not supported")


if __name__ == '__main__':
    print("=" * 80)
    print("ECONOMIC DATA SOURCES - API Integration Test")
    print("=" * 80)

    print("\nThis module provides real-world economic data integration.")
    print("To use, you need API keys (free registration):")
    print("  - FRED: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("  - BLS: https://data.bls.gov/registrationEngine/ (optional)")

    # Try to fetch some data (will fail gracefully without API key)
    print("\n" + "-" * 80)
    print("Testing data fetchers...")
    print("-" * 80)

    try:
        # Test FRED (requires API key)
        fred = FREDDataFetcher()
        print("\n✓ FRED fetcher initialized")

        # Uncomment if you have an API key:
        # df = fred.fetch_series('UNRATE', start_date='2020-01-01')
        # print(f"  Latest unemployment rate: {df['value'].iloc[-1]:.1f}%")

    except Exception as e:
        print(f"\n✗ FRED test skipped (API key needed): {str(e)}")

    try:
        # Test World Bank (no API key needed)
        wb = WorldBankDataFetcher()
        print("\n✓ World Bank fetcher initialized")

        df = wb.fetch_indicator('SP.POP.TOTL', country='US', start_year=2020)
        if not df.empty:
            latest_pop = df['value'].iloc[-1] / 1e6
            print(f"  US Population ({df['year'].iloc[-1]}): {latest_pop:.1f} million")

    except Exception as e:
        print(f"\n✗ World Bank test failed: {str(e)}")

    print("\n" + "=" * 80)
    print("Setup Instructions:")
    print("=" * 80)
    print("\n1. Get a free FRED API key:")
    print("   Visit: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("\n2. Set your API key:")
    print("   Option A: Environment variable")
    print("     export FRED_API_KEY='your_key_here'")
    print("\n   Option B: Config file")
    print("     Create chicago_price_theory/config.json:")
    print('     {"fred_api_key": "your_key_here"}')
    print("\n3. Use in models:")
    print("   from chicago_price_theory.data_sources import get_economic_data")
    print("   df = get_economic_data('unemployment', source='fred')")
