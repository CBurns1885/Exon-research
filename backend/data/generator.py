"""Generate synthetic economic data for African frontier markets."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from backend.config import config


class EconomicDataGenerator:
    """Generate realistic synthetic economic data."""

    def __init__(self, years: int = 5):
        """Initialize generator with historical period."""
        self.years = years
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * years)

        # Base parameters for each country (realistic values)
        self.country_params = {
            "KE": {  # Kenya
                "interest_rate": {"base": 10.5, "volatility": 1.5, "trend": 0.0},
                "fx_rate": {"base": 140, "volatility": 5, "trend": 2.0},  # KES/USD
                "capital_inflow": {"base": 150, "volatility": 30, "trend": 5.0},  # Million USD/month
                "capital_outflow": {"base": 100, "volatility": 20, "trend": 3.0},
                "imports": {"base": 1800, "volatility": 200, "trend": 50.0},  # Million USD/month
                "exports": {"base": 600, "volatility": 80, "trend": 20.0},
                "gdp_growth": {"base": 5.5, "volatility": 1.0, "trend": 0.0},  # Quarterly %
            },
            "NG": {  # Nigeria
                "interest_rate": {"base": 18.5, "volatility": 2.0, "trend": 0.5},
                "fx_rate": {"base": 750, "volatility": 50, "trend": 20.0},  # NGN/USD
                "capital_inflow": {"base": 500, "volatility": 100, "trend": 10.0},
                "capital_outflow": {"base": 400, "volatility": 80, "trend": 15.0},
                "imports": {"base": 4500, "volatility": 400, "trend": 100.0},
                "exports": {"base": 3800, "volatility": 500, "trend": 80.0},  # Oil dependent
                "gdp_growth": {"base": 3.2, "volatility": 1.5, "trend": 0.0},
            },
            "GH": {  # Ghana
                "interest_rate": {"base": 28.0, "volatility": 3.0, "trend": 1.0},
                "fx_rate": {"base": 12, "volatility": 1.5, "trend": 0.8},  # GHS/USD
                "capital_inflow": {"base": 180, "volatility": 40, "trend": 8.0},
                "capital_outflow": {"base": 130, "volatility": 30, "trend": 5.0},
                "imports": {"base": 1200, "volatility": 150, "trend": 40.0},
                "exports": {"base": 1400, "volatility": 180, "trend": 50.0},  # Gold, cocoa
                "gdp_growth": {"base": 4.8, "volatility": 1.2, "trend": 0.0},
            },
            "RW": {  # Rwanda
                "interest_rate": {"base": 7.5, "volatility": 1.0, "trend": 0.0},
                "fx_rate": {"base": 1050, "volatility": 30, "trend": 10.0},  # RWF/USD
                "capital_inflow": {"base": 100, "volatility": 25, "trend": 8.0},
                "capital_outflow": {"base": 60, "volatility": 15, "trend": 3.0},
                "imports": {"base": 350, "volatility": 50, "trend": 15.0},
                "exports": {"base": 120, "volatility": 20, "trend": 8.0},
                "gdp_growth": {"base": 7.5, "volatility": 1.5, "trend": 0.0},
            },
            "EG": {  # Egypt
                "interest_rate": {"base": 19.25, "volatility": 2.5, "trend": 0.3},
                "fx_rate": {"base": 31, "volatility": 3, "trend": 1.5},  # EGP/USD
                "capital_inflow": {"base": 800, "volatility": 150, "trend": 20.0},
                "capital_outflow": {"base": 600, "volatility": 120, "trend": 15.0},
                "imports": {"base": 6000, "volatility": 500, "trend": 150.0},
                "exports": {"base": 3000, "volatility": 300, "trend": 80.0},
                "gdp_growth": {"base": 4.5, "volatility": 1.0, "trend": 0.0},
            }
        }

    def generate_time_series(
        self,
        base: float,
        volatility: float,
        trend: float,
        dates: pd.DatetimeIndex,
        seasonal: bool = True
    ) -> np.ndarray:
        """
        Generate a time series with trend, seasonality, and noise.

        Args:
            base: Base value
            volatility: Standard deviation of random noise
            trend: Monthly trend (added linearly)
            dates: Date index
            seasonal: Whether to add seasonal component

        Returns:
            Array of values
        """
        n = len(dates)

        # Linear trend
        trend_component = np.linspace(0, trend * n / 12, n)

        # Seasonal component (12-month cycle)
        seasonal_component = 0
        if seasonal:
            seasonal_amplitude = base * 0.05  # 5% seasonal variation
            seasonal_component = seasonal_amplitude * np.sin(
                2 * np.pi * np.arange(n) / 12
            )

        # Random walk component
        random_walk = np.cumsum(np.random.randn(n) * volatility * 0.3)

        # White noise
        noise = np.random.randn(n) * volatility

        # Combine components
        values = base + trend_component + seasonal_component + random_walk + noise

        # Ensure positive values for most indicators
        values = np.maximum(values, base * 0.5)

        return values

    def generate_country_data(self, country_code: str) -> pd.DataFrame:
        """
        Generate all economic indicators for a country.

        Args:
            country_code: Two-letter country code

        Returns:
            DataFrame with all indicators
        """
        if country_code not in self.country_params:
            raise ValueError(f"Unknown country code: {country_code}")

        params = self.country_params[country_code]

        # Generate monthly dates
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')

        data = []

        for indicator, ind_params in params.items():
            # Different frequencies for different indicators
            if indicator == "gdp_growth":
                # Quarterly data for GDP
                ind_dates = pd.date_range(
                    start=self.start_date, end=self.end_date, freq='QS'
                )
            elif indicator == "interest_rate":
                # Interest rates change less frequently
                ind_dates = dates[::1]  # Monthly but less volatile
            else:
                ind_dates = dates

            values = self.generate_time_series(
                base=ind_params["base"],
                volatility=ind_params["volatility"],
                trend=ind_params["trend"],
                dates=ind_dates,
                seasonal=(indicator in ["imports", "exports", "capital_inflow"])
            )

            for date, value in zip(ind_dates, values):
                data.append({
                    "country_code": country_code,
                    "date": date,
                    "indicator": indicator,
                    "value": value
                })

        return pd.DataFrame(data)

    def generate_all_countries(self) -> pd.DataFrame:
        """Generate data for all countries."""
        all_data = []

        for country_code in config.get_country_codes():
            country_data = self.generate_country_data(country_code)
            all_data.append(country_data)

        return pd.concat(all_data, ignore_index=True)

    def save_to_database(self, db_session):
        """Save generated data to database."""
        from backend.data.database import EconomicData

        df = self.generate_all_countries()

        # Clear existing data
        db_session.query(EconomicData).delete()

        # Insert new data
        for _, row in df.iterrows():
            record = EconomicData(
                country_code=row["country_code"],
                date=row["date"].date(),
                indicator=row["indicator"],
                value=float(row["value"])
            )
            db_session.add(record)

        db_session.commit()

        return len(df)
