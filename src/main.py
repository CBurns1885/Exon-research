"""Main entry point for Fed Sentiment Analysis"""

import argparse
from datetime import datetime, timedelta
import sys

from .utils import load_config, Database, setup_logger
from .utils.config_loader import load_env
from .scraper import TwitterScraper
from .sentiment import SentimentAnalyzer
from .fed_tracker import FedTracker
from .analysis import CorrelationAnalyzer, FedPredictor
from .visualization import Visualizer


def scrape_and_analyze(db: Database, batch_size: int = 50):
    """
    Scrape tweets and perform sentiment analysis

    Args:
        db: Database instance
        batch_size: Number of tweets to analyze in each batch
    """
    logger = setup_logger("main")
    logger.info("Starting scrape and analyze workflow...")

    # Scrape tweets
    logger.info("Scraping Twitter...")
    scraper = TwitterScraper()
    tweets = scraper.run_full_scrape()

    if not tweets:
        logger.warning("No tweets collected")
        return

    logger.info(f"Collected {len(tweets)} tweets")

    # Save tweets to database
    for tweet in tweets:
        db.insert_tweet(tweet)

    # Get tweets that need sentiment analysis
    unanalyzed_tweets = db.get_tweets_for_sentiment_analysis(limit=batch_size)

    if not unanalyzed_tweets:
        logger.info("No tweets to analyze")
        return

    logger.info(f"Analyzing sentiment for {len(unanalyzed_tweets)} tweets...")

    # Analyze sentiment
    analyzer = SentimentAnalyzer()
    sentiments = analyzer.analyze_tweets_batch(unanalyzed_tweets)

    # Save sentiment results
    for sentiment in sentiments:
        db.insert_sentiment(sentiment)

    # Print summary
    aggregate = analyzer.get_aggregate_sentiment(sentiments)
    logger.info(f"Sentiment Summary:")
    logger.info(f"  Average: {aggregate['avg_sentiment']:.3f}")
    logger.info(f"  Bullish: {aggregate['bullish_percentage']:.1f}%")
    logger.info(f"  Bearish: {aggregate['bearish_percentage']:.1f}%")
    logger.info(f"  Neutral: {aggregate['neutral_percentage']:.1f}%")


def predict_next_meeting(db: Database):
    """
    Make prediction for the next Fed meeting

    Args:
        db: Database instance
    """
    logger = setup_logger("main")
    logger.info("Predicting next Fed meeting...")

    # Get next meeting date
    fed_tracker = FedTracker()
    next_meeting = fed_tracker.get_next_meeting()

    if not next_meeting:
        logger.warning("No upcoming Fed meetings found")
        return

    meeting_date = next_meeting['date']
    days_until = next_meeting['days_until']

    logger.info(f"Next FOMC meeting: {meeting_date} (in {days_until} days)")

    # Make prediction
    predictor = FedPredictor(db)
    prediction_report = predictor.generate_prediction_report(meeting_date)

    # Print prediction
    print("\n" + "="*60)
    print("FEDERAL RESERVE DECISION PREDICTION")
    print("="*60)
    print(f"Meeting Date: {meeting_date}")
    print(f"Days Until Meeting: {days_until}")
    print(f"\nPredicted Action: {prediction_report['predicted_action'].replace('_', ' ').title()}")
    print(f"Confidence: {prediction_report['confidence']*100:.1f}%")
    print(f"\nSentiment Metrics:")
    print(f"  Average Sentiment: {prediction_report['sentiment_avg']:.3f}")
    print(f"  Recent Sentiment: {prediction_report['sentiment_recent']:.3f}")
    print(f"  Sentiment Trend: {prediction_report['sentiment_trend']:.3f}")
    print(f"  Bullish: {prediction_report['bullish_percentage']:.1f}%")
    print(f"  Bearish: {prediction_report['bearish_percentage']:.1f}%")
    print(f"\nReasoning: {prediction_report['reasoning']}")
    print(f"\nTotal Tweets Analyzed: {prediction_report['total_tweets_analyzed']}")
    print("="*60 + "\n")

    # Create visualization
    visualizer = Visualizer()
    visualizer.create_prediction_summary_chart(prediction_report)
    logger.info("Prediction visualization created")

    # Save prediction to database
    db.cursor.execute("""
        INSERT INTO predictions
        (prediction_date, target_meeting_date, predicted_action, confidence, sentiment_avg, sentiment_trend)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        prediction_report['prediction_date'],
        prediction_report['target_meeting_date'],
        prediction_report['predicted_action'],
        prediction_report['confidence'],
        prediction_report['sentiment_avg'],
        prediction_report['sentiment_trend']
    ))
    db.conn.commit()


def analyze_historical(db: Database):
    """
    Analyze historical correlation between sentiment and Fed decisions

    Args:
        db: Database instance
    """
    logger = setup_logger("main")
    logger.info("Analyzing historical correlation...")

    # Get historical Fed decisions
    fed_tracker = FedTracker()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    decisions = fed_tracker.get_historical_decisions(start_date, end_date)

    if not decisions:
        logger.warning("No historical decisions found")
        return

    logger.info(f"Analyzing {len(decisions)} Fed decisions")

    # Analyze correlation
    correlation_analyzer = CorrelationAnalyzer(db)
    correlation_result = correlation_analyzer.correlate_sentiment_with_decisions(decisions)

    # Print results
    print("\n" + "="*60)
    print("HISTORICAL CORRELATION ANALYSIS")
    print("="*60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Sample Size: {correlation_result['sample_size']} meetings")

    if correlation_result['correlation'] is not None:
        print(f"\nCorrelation Coefficient: {correlation_result['correlation']:.3f}")
        print(f"P-Value: {correlation_result['p_value']:.4f}")
        print(f"Statistically Significant: {correlation_result['statistically_significant']}")
        print(f"\n{correlation_result['interpretation']}")
    else:
        print(f"\n{correlation_result['message']}")

    print("="*60 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Federal Reserve Interest Rate Prediction via Twitter Sentiment"
    )

    parser.add_argument(
        'command',
        choices=['scrape', 'predict', 'analyze', 'full'],
        help='Command to execute: scrape (collect and analyze tweets), '
             'predict (predict next meeting), analyze (historical correlation), '
             'full (run all steps)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of tweets to analyze per batch (default: 50)'
    )

    args = parser.parse_args()

    # Load environment variables
    load_env()

    # Initialize database
    db = Database()
    logger = setup_logger("main")

    try:
        if args.command == 'scrape':
            scrape_and_analyze(db, args.batch_size)

        elif args.command == 'predict':
            predict_next_meeting(db)

        elif args.command == 'analyze':
            analyze_historical(db)

        elif args.command == 'full':
            logger.info("Running full workflow...")
            scrape_and_analyze(db, args.batch_size)
            predict_next_meeting(db)
            analyze_historical(db)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        db.close()


if __name__ == "__main__":
    main()
