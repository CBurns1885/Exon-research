"""Interest rate forecasting models combining NLP sentiment with economic data."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateForecaster:
    """Forecast central bank interest rates using sentiment + economic data."""

    def __init__(self):
        """Initialize forecaster."""
        self.models = {}

    def prepare_features(
        self,
        rates_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        indicators_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare features for forecasting by combining rates, sentiment, and economic data.

        Args:
            rates_df: Historical interest rates
            sentiment_df: Speech sentiment data
            indicators_df: Economic indicators

        Returns:
            DataFrame with combined features
        """
        # Ensure dates are datetime
        rates_df['date'] = pd.to_datetime(rates_df['date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        indicators_df['date'] = pd.to_datetime(indicators_df['date'])

        # Aggregate sentiment by month
        sentiment_monthly = sentiment_df.groupby(
            [pd.Grouper(key='date', freq='MS')]
        ).agg({
            'sentiment_score': 'mean',
            'confidence': 'mean'
        }).reset_index()

        # Pivot indicators
        indicators_pivot = indicators_df.pivot_table(
            index='date',
            columns='indicator_type',
            values='value',
            aggfunc='mean'
        ).reset_index()

        # Merge all data
        features = rates_df.copy()
        features = features.merge(sentiment_monthly, on='date', how='left')
        features = features.merge(indicators_pivot, on='date', how='left')

        # Fill missing sentiment with 0 (neutral)
        features['sentiment_score'] = features['sentiment_score'].fillna(0)
        features['confidence'] = features['confidence'].fillna(0.5)

        # Forward fill economic indicators
        for col in indicators_pivot.columns:
            if col != 'date':
                features[col] = features[col].fillna(method='ffill')

        # Create lag features
        features['rate_lag_1'] = features['rate'].shift(1)
        features['rate_lag_3'] = features['rate'].shift(3)
        features['rate_change'] = features['rate'] - features['rate_lag_1']

        # Sentiment momentum
        features['sentiment_lag_1'] = features['sentiment_score'].shift(1)
        features['sentiment_change'] = features['sentiment_score'] - features['sentiment_lag_1']

        return features.dropna()

    def forecast_with_prophet(
        self,
        historical_rates: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        months_ahead: int = 6
    ) -> pd.DataFrame:
        """
        Forecast rates using Prophet with sentiment as regressor.

        Args:
            historical_rates: Historical rate data
            sentiment_df: Sentiment scores
            months_ahead: Forecast horizon

        Returns:
            DataFrame with forecasts
        """
        try:
            # Prepare data for Prophet
            df = historical_rates[['date', 'rate']].copy()
            df.columns = ['ds', 'y']

            # Add sentiment as regressor
            if not sentiment_df.empty:
                sentiment_monthly = sentiment_df.groupby(
                    pd.Grouper(key='date', freq='MS')
                )['sentiment_score'].mean().reset_index()
                sentiment_monthly.columns = ['ds', 'sentiment']

                df = df.merge(sentiment_monthly, on='ds', how='left')
                df['sentiment'] = df['sentiment'].fillna(0)

            # Initialize Prophet
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=1.0,
                interval_width=0.95,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True
            )

            if 'sentiment' in df.columns:
                model.add_regressor('sentiment')

            # Fit model
            model.fit(df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=months_ahead, freq='MS')

            # Add sentiment for future (assume neutral)
            if 'sentiment' in df.columns:
                future['sentiment'] = 0

            # Predict
            forecast = model.predict(future)

            # Extract forecasts
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(months_ahead)
            forecast_df.columns = ['date', 'predicted_rate', 'lower_bound', 'upper_bound']

            return forecast_df

        except Exception as e:
            logger.error(f"Prophet forecasting failed: {e}")
            return pd.DataFrame()

    def forecast_rate_change_probability(
        self,
        features: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Predict probability of rate hike, hold, or cut.

        Args:
            features: Feature dataframe with current conditions

        Returns:
            Dictionary with probabilities
        """
        try:
            # Prepare data
            X = features.drop(['date', 'rate', 'change', 'decision_type'], axis=1, errors='ignore')

            # For demo, use simple rule-based logic
            # In production, train ML classifier on historical data

            latest = features.iloc[-1]

            sentiment = latest.get('sentiment_score', 0)
            rate_current = latest.get('rate', 5.0)
            inflation = latest.get('CPI', 2.0)
            target = 2.0

            # Simple heuristic
            inflation_gap = inflation - target
            combined_signal = sentiment + (inflation_gap / 5.0)

            # Convert to probabilities
            if combined_signal > 0.3:
                prob_hike = 0.6
                prob_hold = 0.3
                prob_cut = 0.1
            elif combined_signal < -0.3:
                prob_hike = 0.1
                prob_hold = 0.3
                prob_cut = 0.6
            else:
                prob_hike = 0.2
                prob_hold = 0.6
                prob_cut = 0.2

            return {
                "prob_hike": prob_hike,
                "prob_hold": prob_hold,
                "prob_cut": prob_cut,
            }

        except Exception as e:
            logger.error(f"Probability forecasting failed: {e}")
            return {"prob_hike": 0.33, "prob_hold": 0.34, "prob_cut": 0.33}

    def forecast_bank_rates(
        self,
        bank_code: str,
        historical_rates: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        forecast_horizons: List[int] = None
    ) -> List[Dict]:
        """
        Generate rate forecasts for a central bank.

        Args:
            bank_code: Central bank code
            historical_rates: Historical rate decisions
            sentiment_df: Speech sentiment data
            indicators_df: Economic indicators
            forecast_horizons: Months to forecast [3, 6, 12]

        Returns:
            List of forecast dictionaries
        """
        forecast_horizons = forecast_horizons or config.FORECAST_MONTHS

        forecasts = []

        try:
            # Prepare features
            features = self.prepare_features(historical_rates, sentiment_df, indicators_df)

            if features.empty:
                logger.warning(f"No features available for {bank_code}")
                return []

            # Get current rate
            current_rate = historical_rates.iloc[-1]['rate']
            current_date = historical_rates.iloc[-1]['date']

            # Calculate probabilities
            probs = self.forecast_rate_change_probability(features)

            # Generate forecasts for each horizon
            for months in forecast_horizons:
                target_date = current_date + timedelta(days=30 * months)

                # Use Prophet for point forecast
                prophet_forecast = self.forecast_with_prophet(
                    historical_rates,
                    sentiment_df,
                    months_ahead=months
                )

                if not prophet_forecast.empty:
                    predicted_rate = prophet_forecast.iloc[-1]['predicted_rate']
                    lower_bound = prophet_forecast.iloc[-1]['lower_bound']
                    upper_bound = prophet_forecast.iloc[-1]['upper_bound']
                else:
                    # Fallback: use current rate with slight adjustment based on sentiment
                    avg_sentiment = sentiment_df['sentiment_score'].mean()
                    predicted_rate = current_rate + (avg_sentiment * 0.25 * months / 12)
                    lower_bound = predicted_rate - 0.5
                    upper_bound = predicted_rate + 0.5

                # Ensure non-negative rates (except BoJ)
                if bank_code != "BOJ":
                    predicted_rate = max(0, predicted_rate)
                    lower_bound = max(0, lower_bound)
                    upper_bound = max(0, upper_bound)

                forecast = {
                    "bank_code": bank_code,
                    "forecast_date": datetime.now().date(),
                    "target_date": target_date.date() if hasattr(target_date, 'date') else target_date,
                    "predicted_rate": round(predicted_rate, 2),
                    "lower_bound": round(lower_bound, 2),
                    "upper_bound": round(upper_bound, 2),
                    "prob_hike": round(probs['prob_hike'], 2),
                    "prob_hold": round(probs['prob_hold'], 2),
                    "prob_cut": round(probs['prob_cut'], 2),
                    "model_type": "prophet_with_sentiment",
                    "confidence": 0.75,
                    "sentiment_weight": 0.3,
                    "economic_weight": 0.7,
                }

                forecasts.append(forecast)

        except Exception as e:
            logger.error(f"Forecasting failed for {bank_code}: {e}")

        return forecasts

    def save_forecasts(self, forecasts: List[Dict], db_session):
        """Save forecasts to database."""
        from backend.data.database import RateForecast

        # Clear old forecasts for same bank and forecast date
        for forecast in forecasts:
            db_session.query(RateForecast).filter(
                RateForecast.bank_code == forecast['bank_code'],
                RateForecast.forecast_date == forecast['forecast_date']
            ).delete()

        # Insert new forecasts
        for forecast in forecasts:
            record = RateForecast(**forecast)
            db_session.add(record)

        db_session.commit()
        return len(forecasts)
