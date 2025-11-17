"""Forecasting models for economic indicators."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

from backend.config import config


class EconomicForecaster:
    """Forecasts economic indicators using multiple models."""

    def __init__(self, forecast_months: int = 6):
        """Initialize forecaster."""
        self.forecast_months = forecast_months

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for forecasting."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df

    def forecast_arima(
        self,
        data: pd.Series,
        steps: int,
        order: Tuple[int, int, int] = (1, 1, 1)
    ) -> Dict[str, np.ndarray]:
        """
        Forecast using ARIMA model.

        Args:
            data: Historical time series
            steps: Number of periods to forecast
            order: ARIMA order (p, d, q)

        Returns:
            Dictionary with forecast, lower_bound, upper_bound
        """
        try:
            model = ARIMA(data, order=order)
            fitted = model.fit()

            # Forecast
            forecast_result = fitted.forecast(steps=steps)
            forecast_values = forecast_result

            # Get prediction intervals
            predictions = fitted.get_forecast(steps=steps)
            conf_int = predictions.conf_int(alpha=0.05)

            return {
                "forecast": np.array(forecast_values),
                "lower_bound": np.array(conf_int.iloc[:, 0]),
                "upper_bound": np.array(conf_int.iloc[:, 1])
            }
        except Exception as e:
            print(f"ARIMA failed: {e}, using fallback")
            return self._fallback_forecast(data, steps)

    def forecast_prophet(
        self,
        dates: pd.Series,
        values: pd.Series,
        steps: int
    ) -> Dict[str, np.ndarray]:
        """
        Forecast using Facebook Prophet.

        Args:
            dates: Date series
            values: Value series
            steps: Number of periods to forecast

        Returns:
            Dictionary with forecast, lower_bound, upper_bound
        """
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': dates,
                'y': values
            })

            # Initialize and fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95
            )
            model.fit(df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=steps, freq='MS')
            forecast = model.predict(future)

            # Extract forecasts (only future values)
            forecast_values = forecast.tail(steps)

            return {
                "forecast": np.array(forecast_values['yhat']),
                "lower_bound": np.array(forecast_values['yhat_lower']),
                "upper_bound": np.array(forecast_values['yhat_upper'])
            }
        except Exception as e:
            print(f"Prophet failed: {e}, using fallback")
            return self._fallback_forecast(values, steps)

    def forecast_exponential_smoothing(
        self,
        data: pd.Series,
        steps: int,
        seasonal_periods: int = 12
    ) -> Dict[str, np.ndarray]:
        """
        Forecast using Exponential Smoothing.

        Args:
            data: Historical time series
            steps: Number of periods to forecast
            seasonal_periods: Seasonal period length

        Returns:
            Dictionary with forecast, lower_bound, upper_bound
        """
        try:
            if len(data) < 2 * seasonal_periods:
                seasonal = None
            else:
                seasonal = 'add'

            model = ExponentialSmoothing(
                data,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None,
                trend='add'
            )
            fitted = model.fit()

            # Forecast
            forecast_values = fitted.forecast(steps=steps)

            # Estimate confidence intervals (simple approach)
            residuals = fitted.fittedvalues - data[:len(fitted.fittedvalues)]
            std_error = np.std(residuals)
            margin = 1.96 * std_error  # 95% confidence

            return {
                "forecast": np.array(forecast_values),
                "lower_bound": np.array(forecast_values - margin),
                "upper_bound": np.array(forecast_values + margin)
            }
        except Exception as e:
            print(f"Exponential Smoothing failed: {e}, using fallback")
            return self._fallback_forecast(data, steps)

    def _fallback_forecast(
        self,
        data: pd.Series,
        steps: int
    ) -> Dict[str, np.ndarray]:
        """
        Simple fallback forecast using trend and average.

        Args:
            data: Historical time series
            steps: Number of periods to forecast

        Returns:
            Dictionary with forecast, lower_bound, upper_bound
        """
        # Calculate simple linear trend
        x = np.arange(len(data))
        y = data.values

        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        # Forecast
        future_x = np.arange(len(data), len(data) + steps)
        forecast_values = slope * future_x + intercept

        # Simple confidence interval based on historical std
        std_error = np.std(y)
        margin = 1.96 * std_error

        return {
            "forecast": forecast_values,
            "lower_bound": forecast_values - margin,
            "upper_bound": forecast_values + margin
        }

    def forecast_indicator(
        self,
        country_code: str,
        indicator: str,
        data: pd.DataFrame,
        model_type: str = "prophet"
    ) -> pd.DataFrame:
        """
        Forecast a specific indicator for a country.

        Args:
            country_code: Country code
            indicator: Indicator name
            data: Historical data DataFrame
            model_type: Model to use (prophet, arima, exponential)

        Returns:
            DataFrame with forecasts
        """
        # Filter data
        mask = (data['country_code'] == country_code) & (data['indicator'] == indicator)
        indicator_data = data[mask].copy()

        if len(indicator_data) < 10:
            print(f"Insufficient data for {country_code} - {indicator}")
            return pd.DataFrame()

        indicator_data = self.prepare_data(indicator_data)

        # Generate future dates
        last_date = indicator_data['date'].max()

        # Determine frequency
        if indicator == "gdp_growth":
            freq = 'QS'
            steps = (self.forecast_months // 3) + 1
        else:
            freq = 'MS'
            steps = self.forecast_months

        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps,
            freq=freq
        )

        # Select model and forecast
        if model_type == "prophet":
            result = self.forecast_prophet(
                indicator_data['date'],
                indicator_data['value'],
                steps
            )
        elif model_type == "arima":
            result = self.forecast_arima(
                indicator_data['value'],
                steps
            )
        else:  # exponential
            result = self.forecast_exponential_smoothing(
                indicator_data['value'],
                steps
            )

        # Create forecast DataFrame
        forecasts = []
        for i, date in enumerate(future_dates):
            forecasts.append({
                'country_code': country_code,
                'date': date,
                'indicator': indicator,
                'forecast_value': result['forecast'][i],
                'lower_bound': result['lower_bound'][i],
                'upper_bound': result['upper_bound'][i],
                'model_type': model_type
            })

        return pd.DataFrame(forecasts)

    def forecast_all(
        self,
        data: pd.DataFrame,
        model_type: str = "prophet"
    ) -> pd.DataFrame:
        """
        Forecast all indicators for all countries.

        Args:
            data: Historical data DataFrame
            model_type: Model to use

        Returns:
            DataFrame with all forecasts
        """
        all_forecasts = []

        countries = config.get_country_codes()
        indicators = config.INDICATORS

        print(f"\nGenerating forecasts using {model_type} model...")

        for country in countries:
            country_name = config.get_country_name(country)
            print(f"\nForecasting {country_name} ({country})...")

            for indicator in indicators:
                try:
                    forecast = self.forecast_indicator(
                        country, indicator, data, model_type
                    )
                    if not forecast.empty:
                        all_forecasts.append(forecast)
                        print(f"  ✓ {indicator}")
                except Exception as e:
                    print(f"  ✗ {indicator}: {e}")

        if not all_forecasts:
            return pd.DataFrame()

        return pd.concat(all_forecasts, ignore_index=True)

    def save_forecasts(self, forecasts: pd.DataFrame, db_session):
        """Save forecasts to database."""
        from backend.data.database import Forecast

        # Clear existing forecasts
        db_session.query(Forecast).delete()

        # Insert new forecasts
        for _, row in forecasts.iterrows():
            record = Forecast(
                country_code=row['country_code'],
                date=row['date'].date(),
                indicator=row['indicator'],
                forecast_value=float(row['forecast_value']),
                lower_bound=float(row['lower_bound']) if pd.notna(row['lower_bound']) else None,
                upper_bound=float(row['upper_bound']) if pd.notna(row['upper_bound']) else None,
                model_type=row['model_type']
            )
            db_session.add(record)

        db_session.commit()

        return len(forecasts)
