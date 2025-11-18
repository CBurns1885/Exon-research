"""Twitter/X scraper for economy-related keywords"""

import tweepy
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import time
from itertools import chain

from ..utils import load_config, load_keywords, setup_logger
from ..utils.config_loader import get_api_key


class TwitterScraper:
    """Scraper for collecting economy-related tweets from Twitter/X"""

    def __init__(self):
        """Initialize the Twitter scraper"""
        self.config = load_config()
        self.keywords_config = load_keywords()
        self.logger = setup_logger("twitter_scraper")

        # Set up Twitter API client
        self.client = None
        self._setup_client()

        # Configuration
        self.max_tweets = self.config['scraper']['max_tweets_per_query']
        self.days_lookback = self.config['scraper']['days_lookback']

    def _setup_client(self):
        """Set up Twitter API client with authentication"""
        try:
            bearer_token = get_api_key('twitter')
            self.client = tweepy.Client(
                bearer_token=bearer_token,
                wait_on_rate_limit=True
            )
            self.logger.info("Twitter API client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Twitter client: {e}")
            self.logger.warning("Twitter scraping will not be available")

    def build_search_queries(self) -> List[str]:
        """
        Build search queries from keywords configuration

        Returns:
            List of search query strings
        """
        queries = []

        # Get all keyword categories
        keywords = self.keywords_config.get('keywords', {})

        # Combine keywords into search queries
        # Strategy 1: Major topic combinations
        fed_keywords = keywords.get('federal_reserve', [])
        rate_keywords = keywords.get('interest_rates', [])

        # Core Fed queries
        for fed_kw in fed_keywords[:3]:  # Focus on top keywords
            queries.append(f'"{fed_kw}"')

        # Interest rate queries
        for rate_kw in rate_keywords[:3]:
            queries.append(f'"{rate_kw}"')

        # Combined queries for higher relevance
        queries.append('("Federal Reserve" OR "Fed" OR "FOMC") (rate OR inflation OR policy)')
        queries.append('("Jerome Powell" OR "Janet Yellen") (interest OR inflation OR economy)')
        queries.append('("rate hike" OR "rate cut") (Fed OR "Federal Reserve")')

        self.logger.info(f"Built {len(queries)} search queries")
        return queries

    def scrape_tweets(
        self,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        start_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape tweets based on query

        Args:
            query: Search query (if None, uses all configured queries)
            max_results: Maximum tweets to fetch (default from config)
            start_time: Start time for tweet search (default: days_lookback from config)

        Returns:
            List of tweet dictionaries
        """
        if not self.client:
            self.logger.error("Twitter client not initialized")
            return []

        if max_results is None:
            max_results = self.max_tweets

        if start_time is None:
            start_time = datetime.utcnow() - timedelta(days=self.days_lookback)

        all_tweets = []

        # Use provided query or build queries
        queries = [query] if query else self.build_search_queries()

        for search_query in queries:
            self.logger.info(f"Searching for: {search_query}")

            try:
                # Search for tweets
                tweets = self.client.search_recent_tweets(
                    query=f"{search_query} -is:retweet lang:en",
                    max_results=min(max_results, 100),  # API limit is 100 per request
                    start_time=start_time,
                    tweet_fields=['created_at', 'public_metrics', 'author_id', 'lang'],
                    user_fields=['username', 'name'],
                    expansions=['author_id']
                )

                if not tweets.data:
                    self.logger.info(f"No tweets found for query: {search_query}")
                    continue

                # Create user lookup dictionary
                users = {user.id: user for user in tweets.includes.get('users', [])}

                # Process tweets
                for tweet in tweets.data:
                    user = users.get(tweet.author_id)
                    tweet_data = {
                        'tweet_id': str(tweet.id),
                        'author': user.username if user else 'unknown',
                        'author_name': user.name if user else 'Unknown',
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'likes': tweet.public_metrics.get('like_count', 0),
                        'retweets': tweet.public_metrics.get('retweet_count', 0),
                        'replies': tweet.public_metrics.get('reply_count', 0),
                        'keywords': [search_query]
                    }
                    all_tweets.append(tweet_data)

                self.logger.info(f"Collected {len(tweets.data)} tweets for query: {search_query}")

                # Rate limiting - small pause between queries
                time.sleep(2)

            except tweepy.TweepyException as e:
                self.logger.error(f"Error scraping tweets for query '{search_query}': {e}")
                continue

            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                continue

        # Remove duplicates based on tweet_id
        unique_tweets = {t['tweet_id']: t for t in all_tweets}.values()
        self.logger.info(f"Total unique tweets collected: {len(unique_tweets)}")

        return list(unique_tweets)

    def scrape_from_accounts(
        self,
        usernames: List[str],
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Scrape recent tweets from specific accounts

        Args:
            usernames: List of Twitter usernames
            max_results: Maximum tweets per account

        Returns:
            List of tweet dictionaries
        """
        if not self.client:
            self.logger.error("Twitter client not initialized")
            return []

        all_tweets = []

        for username in usernames:
            try:
                # Get user ID
                user = self.client.get_user(username=username)
                if not user.data:
                    self.logger.warning(f"User not found: {username}")
                    continue

                user_id = user.data.id

                # Get user's tweets
                tweets = self.client.get_users_tweets(
                    id=user_id,
                    max_results=max_results,
                    tweet_fields=['created_at', 'public_metrics'],
                    exclude=['retweets', 'replies']
                )

                if not tweets.data:
                    continue

                for tweet in tweets.data:
                    tweet_data = {
                        'tweet_id': str(tweet.id),
                        'author': username,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'likes': tweet.public_metrics.get('like_count', 0),
                        'retweets': tweet.public_metrics.get('retweet_count', 0),
                        'replies': tweet.public_metrics.get('reply_count', 0),
                        'keywords': ['priority_account']
                    }
                    all_tweets.append(tweet_data)

                self.logger.info(f"Collected {len(tweets.data)} tweets from @{username}")
                time.sleep(2)

            except Exception as e:
                self.logger.error(f"Error scraping tweets from @{username}: {e}")
                continue

        return all_tweets

    def run_full_scrape(self) -> List[Dict[str, Any]]:
        """
        Run a full scrape using all configured queries and priority accounts

        Returns:
            List of all collected tweets
        """
        self.logger.info("Starting full Twitter scrape...")

        # Scrape based on keywords
        keyword_tweets = self.scrape_tweets()

        # Scrape from priority accounts
        priority_accounts = self.keywords_config.get('priority_accounts', [])
        if priority_accounts:
            account_tweets = self.scrape_from_accounts(priority_accounts)
        else:
            account_tweets = []

        # Combine and deduplicate
        all_tweets = list({t['tweet_id']: t for t in keyword_tweets + account_tweets}.values())

        self.logger.info(f"Full scrape complete. Total tweets: {len(all_tweets)}")
        return all_tweets
