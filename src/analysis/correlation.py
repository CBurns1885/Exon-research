"""Correlation analysis between Twitter sentiment and Fed decisions"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from scipy import stats

from ..utils import setup_logger, Database


class CorrelationAnalyzer:
    """Analyze correlation between Twitter sentiment and Federal Reserve decisions"""

    def __init__(self, db: Database):
        """
        Initialize correlation analyzer

        Args:
            db: Database instance
        """
        self.db = db
        self.logger = setup_logger("correlation_analyzer")

    def prepare_sentiment_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Prepare sentiment data as a time series

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with daily sentiment scores
        """
        sentiment_data = self.db.get_sentiment_by_date_range(start_date, end_date)

        if not sentiment_data:
            self.logger.warning("No sentiment data found for date range")
            return pd.DataFrame()

        df = pd.DataFrame(sentiment_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        return df

    def calculate_sentiment_trend(
        self,
        sentiment_df: pd.DataFrame,
        window: int = 7
    ) -> pd.DataFrame:
        """
        Calculate rolling sentiment trends

        Args:
            sentiment_df: DataFrame with sentiment data
            window: Rolling window size in days

        Returns:
            DataFrame with trend calculations
        """
        if sentiment_df.empty:
            return sentiment_df

        df = sentiment_df.copy()

        # Calculate rolling averages
        df['sentiment_ma'] = df['avg_sentiment'].rolling(window=window, min_periods=1).mean()

        # Calculate momentum (rate of change)
        df['sentiment_momentum'] = df['avg_sentiment'].diff(window)

        # Calculate volatility
        df['sentiment_volatility'] = df['avg_sentiment'].rolling(window=window, min_periods=1).std()

        return df

    def analyze_pre_meeting_sentiment(
        self,
        meeting_date: str,
        lookback_days: int = 14
    ) -> Dict[str, Any]:
        """
        Analyze sentiment in the period before a Fed meeting

        Args:
            meeting_date: Date of Fed meeting (YYYY-MM-DD)
            lookback_days: Days before meeting to analyze

        Returns:
            Dictionary with pre-meeting sentiment statistics
        """
        meeting_dt = datetime.strptime(meeting_date, "%Y-%m-%d")
        start_dt = meeting_dt - timedelta(days=lookback_days)

        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = meeting_date

        sentiment_df = self.prepare_sentiment_data(start_date, end_date)

        if sentiment_df.empty:
            self.logger.warning(f"No sentiment data for meeting on {meeting_date}")
            return {}

        # Calculate statistics
        analysis = {
            'meeting_date': meeting_date,
            'lookback_days': lookback_days,
            'avg_sentiment': float(sentiment_df['avg_sentiment'].mean()),
            'final_week_avg': float(sentiment_df.tail(7)['avg_sentiment'].mean()) if len(sentiment_df) >= 7 else None,
            'final_day_sentiment': float(sentiment_df.iloc[-1]['avg_sentiment']),
            'sentiment_trend': 'bullish' if sentiment_df['avg_sentiment'].iloc[-1] > sentiment_df['avg_sentiment'].iloc[0] else 'bearish',
            'total_tweets': int(sentiment_df['tweet_count'].sum()),
            'avg_daily_tweets': float(sentiment_df['tweet_count'].mean()),
            'bullish_percentage': float((sentiment_df['bullish_count'].sum() / sentiment_df['tweet_count'].sum()) * 100),
            'bearish_percentage': float((sentiment_df['bearish_count'].sum() / sentiment_df['tweet_count'].sum()) * 100)
        }

        return analysis

    def correlate_sentiment_with_decisions(
        self,
        fed_decisions: List[Dict[str, Any]],
        lookback_days: int = 14
    ) -> Dict[str, Any]:
        """
        Correlate pre-meeting sentiment with actual Fed decisions

        Args:
            fed_decisions: List of Fed decision dictionaries
            lookback_days: Days before meeting to consider

        Returns:
            Correlation analysis results
        """
        sentiment_before_decisions = []
        actual_decisions = []

        for decision in fed_decisions:
            meeting_date = decision['decision_date']
            sentiment_analysis = self.analyze_pre_meeting_sentiment(meeting_date, lookback_days)

            if not sentiment_analysis:
                continue

            sentiment_before_decisions.append(sentiment_analysis['avg_sentiment'])

            # Convert decision to numeric value
            # Positive = hawkish (hike), Negative = dovish (cut)
            decision_value = decision['rate_change']
            actual_decisions.append(decision_value)

        if len(sentiment_before_decisions) < 3:
            self.logger.warning("Not enough data points for correlation analysis")
            return {
                'correlation': None,
                'p_value': None,
                'sample_size': len(sentiment_before_decisions),
                'message': 'Insufficient data for correlation analysis'
            }

        # Calculate Pearson correlation
        correlation, p_value = stats.pearsonr(sentiment_before_decisions, actual_decisions)

        self.logger.info(f"Correlation: {correlation:.3f}, p-value: {p_value:.4f}")

        return {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'sample_size': len(sentiment_before_decisions),
            'statistically_significant': p_value < 0.05,
            'interpretation': self._interpret_correlation(correlation, p_value)
        }

    def _interpret_correlation(self, correlation: float, p_value: float) -> str:
        """
        Interpret correlation results

        Args:
            correlation: Correlation coefficient
            p_value: Statistical p-value

        Returns:
            Human-readable interpretation
        """
        if p_value >= 0.05:
            return "No statistically significant correlation found between Twitter sentiment and Fed decisions."

        strength = abs(correlation)
        if strength < 0.3:
            strength_desc = "weak"
        elif strength < 0.7:
            strength_desc = "moderate"
        else:
            strength_desc = "strong"

        direction = "positive" if correlation > 0 else "negative"

        return f"A {strength_desc} {direction} correlation exists (r={correlation:.3f}, p={p_value:.4f}). " \
               f"{'Bullish sentiment tends to precede rate hikes.' if correlation > 0 else 'Bullish sentiment tends to precede rate cuts (or vice versa).'}"

    def calculate_prediction_accuracy(
        self,
        predictions: List[Dict[str, Any]],
        actual_decisions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate accuracy of predictions against actual decisions

        Args:
            predictions: List of prediction dictionaries
            actual_decisions: List of actual Fed decisions

        Returns:
            Accuracy metrics
        """
        if not predictions or not actual_decisions:
            return {'accuracy': 0.0, 'total_predictions': 0}

        # Create lookup of actual decisions by date
        decisions_map = {d['decision_date']: d for d in actual_decisions}

        correct = 0
        total = 0

        for pred in predictions:
            target_date = pred['target_meeting_date']
            if target_date in decisions_map:
                predicted_action = pred['predicted_action']
                actual_action = decisions_map[target_date]['decision_type']

                if predicted_action == actual_action:
                    correct += 1
                total += 1

        accuracy = (correct / total * 100) if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'success_rate': f"{correct}/{total}"
        }
