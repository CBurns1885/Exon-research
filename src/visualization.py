"""Visualization module for sentiment and prediction analysis"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .utils import setup_logger


class Visualizer:
    """Create visualizations for sentiment analysis and predictions"""

    def __init__(self, output_dir: str = "data/plots"):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save plots
        """
        self.logger = setup_logger("visualizer")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def plot_sentiment_trend(
        self,
        sentiment_df: pd.DataFrame,
        title: str = "Twitter Sentiment Trend",
        save_path: str = None
    ):
        """
        Plot sentiment trend over time

        Args:
            sentiment_df: DataFrame with date and avg_sentiment columns
            title: Plot title
            save_path: Path to save plot (optional)
        """
        if sentiment_df.empty:
            self.logger.warning("No data to plot")
            return

        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot sentiment line
        ax.plot(sentiment_df['date'], sentiment_df['avg_sentiment'],
                linewidth=2, label='Average Sentiment', color='#2E86AB')

        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')

        # Color background regions
        ax.fill_between(sentiment_df['date'], 0, sentiment_df['avg_sentiment'],
                        where=(sentiment_df['avg_sentiment'] > 0), alpha=0.3,
                        color='green', label='Bullish Region')
        ax.fill_between(sentiment_df['date'], 0, sentiment_df['avg_sentiment'],
                        where=(sentiment_df['avg_sentiment'] < 0), alpha=0.3,
                        color='red', label='Bearish Region')

        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved plot to {save_path}")
        else:
            save_path = self.output_dir / f"sentiment_trend_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved plot to {save_path}")

        plt.close()

    def plot_sentiment_distribution(
        self,
        sentiments: List[Dict[str, Any]],
        save_path: str = None
    ):
        """
        Plot distribution of sentiment categories

        Args:
            sentiments: List of sentiment dictionaries
            save_path: Path to save plot
        """
        if not sentiments:
            self.logger.warning("No sentiment data to plot")
            return

        categories = [s['sentiment_category'] for s in sentiments]
        category_counts = pd.Series(categories).value_counts()

        # Define colors
        colors = {
            'very_bullish': '#2E7D32',
            'bullish': '#66BB6A',
            'neutral': '#FFA726',
            'bearish': '#EF5350',
            'very_bearish': '#C62828'
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(range(len(category_counts)), category_counts.values,
                      color=[colors.get(cat, 'gray') for cat in category_counts.index])

        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in category_counts.index])
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Sentiment Category Distribution', fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"sentiment_distribution_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.logger.info(f"Saved plot to {save_path}")
        plt.close()

    def plot_sentiment_vs_fed_decisions(
        self,
        sentiment_df: pd.DataFrame,
        fed_decisions: List[Dict[str, Any]],
        save_path: str = None
    ):
        """
        Plot sentiment trend with Fed decision markers

        Args:
            sentiment_df: DataFrame with sentiment data
            fed_decisions: List of Fed decision dictionaries
            save_path: Path to save plot
        """
        if sentiment_df.empty:
            self.logger.warning("No sentiment data to plot")
            return

        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot sentiment
        ax.plot(sentiment_df['date'], sentiment_df['avg_sentiment'],
                linewidth=2, label='Twitter Sentiment', color='#2E86AB')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Mark Fed decisions
        decision_colors = {
            'significant_cut': 'darkred',
            'cut': 'red',
            'hold': 'orange',
            'hike': 'lightgreen',
            'significant_hike': 'darkgreen'
        }

        for decision in fed_decisions:
            decision_date = pd.to_datetime(decision['decision_date'])
            decision_type = decision['decision_type']

            ax.axvline(x=decision_date, color=decision_colors.get(decision_type, 'gray'),
                      linestyle=':', alpha=0.7, linewidth=2)
            ax.scatter([decision_date], [0], marker='v', s=200,
                      color=decision_colors.get(decision_type, 'gray'),
                      zorder=5, label=f"{decision_type.replace('_', ' ').title()}")

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        ax.set_title('Twitter Sentiment vs Federal Reserve Decisions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"sentiment_vs_fed_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.logger.info(f"Saved plot to {save_path}")
        plt.close()

    def create_prediction_summary_chart(
        self,
        prediction: Dict[str, Any],
        save_path: str = None
    ):
        """
        Create a summary chart for a prediction

        Args:
            prediction: Prediction dictionary
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Prediction confidence gauge
        confidence = prediction['confidence'] * 100
        predicted_action = prediction['predicted_action'].replace('_', ' ').title()

        ax1.barh([0], [confidence], color='#2E86AB', height=0.3)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Confidence (%)', fontsize=12)
        ax1.set_title(f'Prediction: {predicted_action}\nConfidence: {confidence:.1f}%',
                     fontsize=14, fontweight='bold')
        ax1.set_yticks([])
        ax1.grid(axis='x', alpha=0.3)

        # Sentiment breakdown
        bullish = prediction.get('bullish_percentage', 0)
        bearish = prediction.get('bearish_percentage', 0)
        neutral = 100 - bullish - bearish

        categories = ['Bullish', 'Neutral', 'Bearish']
        percentages = [bullish, neutral, bearish]
        colors_pie = ['#66BB6A', '#FFA726', '#EF5350']

        ax2.pie(percentages, labels=categories, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax2.set_title('Sentiment Breakdown', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"prediction_summary_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.logger.info(f"Saved plot to {save_path}")
        plt.close()
