"""Cryptocurrency price forecasting models."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import config

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoPriceForecaster:
    """Forecast cryptocurrency prices using multiple models."""

    def __init__(self, forecast_days: int = 30):
        """Initialize forecaster."""
        self.forecast_days = forecast_days

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for forecasting."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna(subset=['close'])
        return df

    def forecast_prophet(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Forecast using Facebook Prophet model.

        Args:
            df: Historical price data
            symbol: Cryptocurrency symbol

        Returns:
            DataFrame with forecasts
        """
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['date']),
                'y': df['close']
            })

            # Initialize Prophet model with optimized parameters for crypto
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,  # More flexible to capture crypto volatility
                seasonality_prior_scale=10.0,
                interval_width=0.95
            )

            # Add custom seasonalities for crypto (monthly cycles)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            # Fit model
            model.fit(prophet_df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=self.forecast_days, freq='D')

            # Predict
            forecast = model.predict(future)

            # Extract forecast for future dates only
            forecast_future = forecast.tail(self.forecast_days).copy()

            # Create result dataframe
            result = pd.DataFrame({
                'symbol': symbol,
                'date': forecast_future['ds'].dt.date,
                'predicted_price': forecast_future['yhat'],
                'lower_bound': forecast_future['yhat_lower'],
                'upper_bound': forecast_future['yhat_upper'],
                'model_type': 'prophet'
            })

            # Ensure predictions are positive (crypto prices can't be negative)
            result['predicted_price'] = result['predicted_price'].clip(lower=0)
            result['lower_bound'] = result['lower_bound'].clip(lower=0)
            result['upper_bound'] = result['upper_bound'].clip(lower=0)

            return result

        except Exception as e:
            logger.error(f"Prophet forecast failed for {symbol}: {e}")
            return self._fallback_forecast(df, symbol)

    def forecast_arima(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Forecast using ARIMA model.

        Args:
            df: Historical price data
            symbol: Cryptocurrency symbol

        Returns:
            DataFrame with forecasts
        """
        try:
            prices = df['close'].values

            # Fit ARIMA model (p=5, d=1, q=0 works well for crypto)
            model = ARIMA(prices, order=(5, 1, 0))
            fitted = model.fit()

            # Forecast
            forecast_result = fitted.forecast(steps=self.forecast_days)

            # Get confidence intervals
            predictions = fitted.get_forecast(steps=self.forecast_days)
            conf_int = predictions.conf_int(alpha=0.05)

            # Create future dates
            last_date = pd.to_datetime(df['date'].max())
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=self.forecast_days,
                freq='D'
            )

            # Create result dataframe
            result = pd.DataFrame({
                'symbol': symbol,
                'date': future_dates.date,
                'predicted_price': forecast_result,
                'lower_bound': conf_int.iloc[:, 0],
                'upper_bound': conf_int.iloc[:, 1],
                'model_type': 'arima'
            })

            # Ensure positive prices
            result['predicted_price'] = result['predicted_price'].clip(lower=0)
            result['lower_bound'] = result['lower_bound'].clip(lower=0)
            result['upper_bound'] = result['upper_bound'].clip(lower=0)

            return result

        except Exception as e:
            logger.error(f"ARIMA forecast failed for {symbol}: {e}")
            return self._fallback_forecast(df, symbol)

    def _fallback_forecast(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Simple fallback forecast using linear trend.

        Args:
            df: Historical price data
            symbol: Cryptocurrency symbol

        Returns:
            DataFrame with forecasts
        """
        try:
            # Calculate simple linear trend
            x = np.arange(len(df))
            y = df['close'].values

            # Fit linear regression
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs

            # Generate future predictions
            last_date = pd.to_datetime(df['date'].max())
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=self.forecast_days,
                freq='D'
            )

            future_x = np.arange(len(df), len(df) + self.forecast_days)
            predictions = slope * future_x + intercept

            # Calculate confidence intervals based on historical volatility
            historical_std = df['close'].std()
            margin = 1.96 * historical_std

            result = pd.DataFrame({
                'symbol': symbol,
                'date': future_dates.date,
                'predicted_price': predictions,
                'lower_bound': predictions - margin,
                'upper_bound': predictions + margin,
                'model_type': 'linear_fallback'
            })

            # Ensure positive prices
            result['predicted_price'] = result['predicted_price'].clip(lower=0)
            result['lower_bound'] = result['lower_bound'].clip(lower=0)
            result['upper_bound'] = result['upper_bound'].clip(lower=0)

            return result

        except Exception as e:
            logger.error(f"Fallback forecast failed for {symbol}: {e}")
            return pd.DataFrame()

    def forecast_crypto(self, df: pd.DataFrame, symbol: str, model_type: str = 'prophet') -> pd.DataFrame:
        """
        Forecast cryptocurrency price.

        Args:
            df: Historical price data
            symbol: Cryptocurrency symbol
            model_type: Model to use ('prophet', 'arima', or 'ensemble')

        Returns:
            DataFrame with forecasts
        """
        df = self.prepare_data(df)

        if len(df) < 30:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} days")
            return pd.DataFrame()

        if model_type == 'prophet':
            return self.forecast_prophet(df, symbol)
        elif model_type == 'arima':
            return self.forecast_arima(df, symbol)
        elif model_type == 'ensemble':
            # Combine Prophet and ARIMA
            prophet_forecast = self.forecast_prophet(df, symbol)
            arima_forecast = self.forecast_arima(df, symbol)

            if not prophet_forecast.empty and not arima_forecast.empty:
                # Average the predictions
                ensemble = prophet_forecast.copy()
                ensemble['predicted_price'] = (
                    prophet_forecast['predicted_price'] + arima_forecast['predicted_price']
                ) / 2
                ensemble['lower_bound'] = np.minimum(
                    prophet_forecast['lower_bound'],
                    arima_forecast['lower_bound']
                )
                ensemble['upper_bound'] = np.maximum(
                    prophet_forecast['upper_bound'],
                    arima_forecast['upper_bound']
                )
                ensemble['model_type'] = 'ensemble'
                return ensemble

            return prophet_forecast if not prophet_forecast.empty else arima_forecast
        else:
            return self.forecast_prophet(df, symbol)

    def forecast_all_cryptos(
        self,
        historical_data: Dict[str, pd.DataFrame],
        model_type: str = 'prophet'
    ) -> Dict[str, pd.DataFrame]:
        """
        Forecast all cryptocurrencies.

        Args:
            historical_data: Dictionary of symbol -> historical data DataFrame
            model_type: Model to use for forecasting

        Returns:
            Dictionary of symbol -> forecast DataFrame
        """
        forecasts = {}

        for symbol, df in historical_data.items():
            logger.info(f"Generating {model_type} forecast for {symbol}...")

            try:
                forecast = self.forecast_crypto(df, symbol, model_type)
                if not forecast.empty:
                    forecasts[symbol] = forecast
                    logger.info(f"✓ {symbol}: {len(forecast)} days forecasted")
                else:
                    logger.warning(f"✗ {symbol}: No forecast generated")

            except Exception as e:
                logger.error(f"✗ {symbol}: {e}")

        return forecasts

    def save_forecasts(self, forecasts: Dict[str, pd.DataFrame], db_session):
        """Save forecasts to database."""
        from backend.data.database import CryptoForecast

        # Clear existing forecasts
        db_session.query(CryptoForecast).delete()

        count = 0
        for symbol, forecast_df in forecasts.items():
            for _, row in forecast_df.iterrows():
                record = CryptoForecast(
                    symbol=row['symbol'],
                    date=row['date'],
                    predicted_price=float(row['predicted_price']),
                    lower_bound=float(row['lower_bound']) if pd.notna(row['lower_bound']) else None,
                    upper_bound=float(row['upper_bound']) if pd.notna(row['upper_bound']) else None,
                    model_type=row['model_type']
                )
                db_session.add(record)
                count += 1

        db_session.commit()
        return count
