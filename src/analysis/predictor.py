"""Federal Reserve decision predictor based on Twitter sentiment"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from ..utils import setup_logger, Database, load_config
from .correlation import CorrelationAnalyzer


class FedPredictor:
    """Predict Federal Reserve decisions based on Twitter sentiment"""

    def __init__(self, db: Database):
        """
        Initialize Fed predictor

        Args:
            db: Database instance
        """
        self.db = db
        self.config = load_config()
        self.logger = setup_logger("fed_predictor")
        self.correlation_analyzer = CorrelationAnalyzer(db)

    def predict_next_decision(
        self,
        meeting_date: str,
        lookback_days: int = 14
    ) -> Dict[str, Any]:
        """
        Predict the next Fed decision based on recent sentiment

        Args:
            meeting_date: Date of upcoming FOMC meeting (YYYY-MM-DD)
            lookback_days: Days of sentiment data to consider

        Returns:
            Prediction dictionary
        """
        # Analyze current sentiment trend
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        sentiment_df = self.correlation_analyzer.prepare_sentiment_data(start_date, end_date)

        if sentiment_df.empty:
            self.logger.warning("No sentiment data available for prediction")
            return {
                'predicted_action': 'unknown',
                'confidence': 0.0,
                'reasoning': 'Insufficient sentiment data'
            }

        # Calculate sentiment metrics
        avg_sentiment = float(sentiment_df['avg_sentiment'].mean())
        recent_sentiment = float(sentiment_df.tail(7)['avg_sentiment'].mean()) if len(sentiment_df) >= 7 else avg_sentiment
        sentiment_trend = recent_sentiment - sentiment_df.head(7)['avg_sentiment'].mean() if len(sentiment_df) >= 14 else 0

        # Calculate bullish/bearish percentages
        total_tweets = sentiment_df['tweet_count'].sum()
        bullish_pct = (sentiment_df['bullish_count'].sum() / total_tweets * 100) if total_tweets > 0 else 0
        bearish_pct = (sentiment_df['bearish_count'].sum() / total_tweets * 100) if total_tweets > 0 else 0

        # Make prediction based on sentiment thresholds
        predicted_action, confidence, reasoning = self._determine_prediction(
            avg_sentiment=avg_sentiment,
            recent_sentiment=recent_sentiment,
            sentiment_trend=sentiment_trend,
            bullish_pct=bullish_pct,
            bearish_pct=bearish_pct
        )

        prediction = {
            'prediction_date': end_date,
            'target_meeting_date': meeting_date,
            'predicted_action': predicted_action,
            'confidence': confidence,
            'sentiment_avg': avg_sentiment,
            'sentiment_recent': recent_sentiment,
            'sentiment_trend': sentiment_trend,
            'bullish_percentage': bullish_pct,
            'bearish_percentage': bearish_pct,
            'reasoning': reasoning,
            'total_tweets_analyzed': int(total_tweets)
        }

        self.logger.info(f"Prediction for {meeting_date}: {predicted_action} (confidence: {confidence:.2%})")

        return prediction

    def _determine_prediction(
        self,
        avg_sentiment: float,
        recent_sentiment: float,
        sentiment_trend: float,
        bullish_pct: float,
        bearish_pct: float
    ) -> tuple[str, float, str]:
        """
        Determine prediction based on sentiment metrics

        Args:
            avg_sentiment: Average sentiment score
            recent_sentiment: Recent sentiment score
            sentiment_trend: Sentiment trend (positive = becoming more bullish)
            bullish_pct: Percentage of bullish tweets
            bearish_pct: Percentage of bearish tweets

        Returns:
            Tuple of (predicted_action, confidence, reasoning)
        """
        # Initialize variables
        predicted_action = 'hold'
        confidence = 0.5
        reasoning_parts = []

        # Strong bearish sentiment suggests rate cuts
        if avg_sentiment < -0.4 and bearish_pct > 60:
            predicted_action = 'significant_cut'
            confidence = 0.75
            reasoning_parts.append(f"Strong bearish sentiment ({avg_sentiment:.2f}) with {bearish_pct:.1f}% bearish tweets")

        elif avg_sentiment < -0.2 and bearish_pct > 50:
            predicted_action = 'cut'
            confidence = 0.65
            reasoning_parts.append(f"Bearish sentiment ({avg_sentiment:.2f}) dominates")

        # Strong bullish sentiment suggests rate hikes
        elif avg_sentiment > 0.4 and bullish_pct > 60:
            predicted_action = 'significant_hike'
            confidence = 0.75
            reasoning_parts.append(f"Strong bullish sentiment ({avg_sentiment:.2f}) with {bullish_pct:.1f}% bullish tweets")

        elif avg_sentiment > 0.2 and bullish_pct > 50:
            predicted_action = 'hike'
            confidence = 0.65
            reasoning_parts.append(f"Bullish sentiment ({avg_sentiment:.2f}) prevails")

        # Neutral sentiment suggests hold
        else:
            predicted_action = 'hold'
            confidence = 0.60
            reasoning_parts.append(f"Neutral sentiment ({avg_sentiment:.2f}) suggests maintaining current policy")

        # Adjust confidence based on trend consistency
        if abs(sentiment_trend) > 0.2:
            if (sentiment_trend > 0 and 'hike' in predicted_action) or \
               (sentiment_trend < 0 and 'cut' in predicted_action):
                confidence += 0.1
                reasoning_parts.append("sentiment trend confirms prediction")
            else:
                confidence -= 0.05
                reasoning_parts.append("sentiment trend shows uncertainty")

        # Adjust confidence based on consensus
        sentiment_consensus = abs(bullish_pct - bearish_pct)
        if sentiment_consensus > 40:
            confidence += 0.1
            reasoning_parts.append("strong consensus in sentiment")
        elif sentiment_consensus < 20:
            confidence -= 0.1
            reasoning_parts.append("divided sentiment reduces confidence")

        # Cap confidence
        confidence = min(0.95, max(0.30, confidence))

        reasoning = "; ".join(reasoning_parts)

        return predicted_action, confidence, reasoning

    def generate_prediction_report(
        self,
        meeting_date: str,
        lookback_days: int = 14
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive prediction report

        Args:
            meeting_date: Date of upcoming FOMC meeting
            lookback_days: Days of data to analyze

        Returns:
            Comprehensive prediction report
        """
        prediction = self.predict_next_decision(meeting_date, lookback_days)

        # Add current market context
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        sentiment_df = self.correlation_analyzer.prepare_sentiment_data(start_date, end_date)

        report = {
            **prediction,
            'analysis_period': {
                'start_date': start_date,
                'end_date': end_date,
                'days_analyzed': lookback_days
            },
            'days_until_meeting': (datetime.strptime(meeting_date, "%Y-%m-%d") - datetime.now()).days,
            'data_quality': {
                'total_days_with_data': len(sentiment_df),
                'data_completeness': (len(sentiment_df) / lookback_days * 100) if lookback_days > 0 else 0
            }
        }

        return report
