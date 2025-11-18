"""LLM-based sentiment analysis for economic tweets"""

import anthropic
import openai
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from ..utils import load_config, setup_logger
from ..utils.config_loader import get_api_key


class SentimentAnalyzer:
    """Analyze sentiment of economic tweets using LLMs"""

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize sentiment analyzer

        Args:
            provider: LLM provider ('anthropic' or 'openai'). If None, uses config default
        """
        self.config = load_config()
        self.logger = setup_logger("sentiment_analyzer")

        # LLM configuration
        llm_config = self.config['llm']
        self.provider = provider or llm_config['provider']
        self.model = llm_config['model']
        self.temperature = llm_config['temperature']
        self.max_tokens = llm_config['max_tokens']

        # Sentiment configuration
        self.sentiment_config = self.config['sentiment']

        # Initialize LLM client
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Set up the LLM client based on provider"""
        try:
            if self.provider == 'anthropic':
                api_key = get_api_key('anthropic')
                self.client = anthropic.Anthropic(api_key=api_key)
                self.logger.info("Anthropic client initialized")

            elif self.provider == 'openai':
                api_key = get_api_key('openai')
                openai.api_key = api_key
                self.client = openai.OpenAI(api_key=api_key)
                self.logger.info("OpenAI client initialized")

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except Exception as e:
            self.logger.error(f"Failed to initialize {self.provider} client: {e}")
            raise

    def _get_sentiment_prompt(self, tweet_text: str) -> str:
        """
        Generate the prompt for sentiment analysis

        Args:
            tweet_text: The tweet text to analyze

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert financial analyst specializing in Federal Reserve policy and economic sentiment analysis.

Analyze the following tweet and determine its sentiment regarding:
1. Economic outlook (bullish/bearish on economic growth)
2. Inflation expectations (rising/falling)
3. Federal Reserve policy expectations (hawkish/dovish, rate hikes/cuts expected)

Tweet: "{tweet_text}"

Provide your analysis in the following JSON format:
{{
    "sentiment_score": <float between -1 and 1, where -1 is very bearish/dovish (expecting rate cuts), 0 is neutral, and 1 is very bullish/hawkish (expecting rate hikes)>,
    "sentiment_category": "<one of: very_bearish, bearish, neutral, bullish, very_bullish>",
    "confidence": <float between 0 and 1 indicating your confidence in this analysis>,
    "reasoning": "<brief explanation of your sentiment assessment, 1-2 sentences>",
    "key_factors": [<list of key words/phrases that influenced your assessment>]
}}

Guidelines:
- Very bearish (-1.0 to -0.6): Strong expectation of rate cuts, recession fears, deflationary concerns
- Bearish (-0.6 to -0.2): Moderate expectation of dovish policy, economic slowdown concerns
- Neutral (-0.2 to 0.2): Balanced view, no clear directional bias
- Bullish (0.2 to 0.6): Moderate expectation of hawkish policy, inflation concerns
- Very bullish (0.6 to 1.0): Strong expectation of rate hikes, overheating economy concerns

Respond ONLY with the JSON object, no additional text."""

        return prompt

    def analyze_tweet(self, tweet_text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single tweet

        Args:
            tweet_text: The tweet text to analyze

        Returns:
            Dictionary containing sentiment analysis results
        """
        prompt = self._get_sentiment_prompt(tweet_text)

        try:
            if self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result_text = response.content[0].text

            elif self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert financial analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                result_text = response.choices[0].message.content

            # Parse JSON response
            result = json.loads(result_text)

            # Add metadata
            result['llm_model'] = self.model
            result['analyzed_at'] = datetime.utcnow().isoformat()

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.error(f"Response text: {result_text}")
            # Return neutral sentiment as fallback
            return {
                'sentiment_score': 0.0,
                'sentiment_category': 'neutral',
                'confidence': 0.0,
                'reasoning': 'Failed to parse LLM response',
                'key_factors': [],
                'llm_model': self.model,
                'analyzed_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error analyzing tweet sentiment: {e}")
            raise

    def analyze_tweets_batch(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of tweets

        Args:
            tweets: List of tweet dictionaries with 'tweet_id' and 'text' keys

        Returns:
            List of sentiment analysis results with tweet_id
        """
        results = []

        for i, tweet in enumerate(tweets):
            try:
                self.logger.info(f"Analyzing tweet {i+1}/{len(tweets)}: {tweet['tweet_id']}")

                sentiment = self.analyze_tweet(tweet['text'])
                sentiment['tweet_id'] = tweet['tweet_id']
                results.append(sentiment)

                # Small delay to avoid rate limiting
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i+1}/{len(tweets)} tweets")

            except Exception as e:
                self.logger.error(f"Error analyzing tweet {tweet['tweet_id']}: {e}")
                # Add failed analysis with neutral sentiment
                results.append({
                    'tweet_id': tweet['tweet_id'],
                    'sentiment_score': 0.0,
                    'sentiment_category': 'neutral',
                    'confidence': 0.0,
                    'reasoning': f'Analysis failed: {str(e)}',
                    'key_factors': [],
                    'llm_model': self.model,
                    'analyzed_at': datetime.utcnow().isoformat()
                })

        self.logger.info(f"Completed sentiment analysis for {len(results)} tweets")
        return results

    def get_aggregate_sentiment(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate sentiment statistics

        Args:
            sentiments: List of sentiment analysis results

        Returns:
            Dictionary with aggregate statistics
        """
        if not sentiments:
            return {
                'avg_sentiment': 0.0,
                'median_sentiment': 0.0,
                'total_count': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'avg_confidence': 0.0
            }

        scores = [s['sentiment_score'] for s in sentiments]
        categories = [s['sentiment_category'] for s in sentiments]
        confidences = [s.get('confidence', 0.0) for s in sentiments]

        # Count categories
        bullish = sum(1 for c in categories if c in ['bullish', 'very_bullish'])
        bearish = sum(1 for c in categories if c in ['bearish', 'very_bearish'])
        neutral = sum(1 for c in categories if c == 'neutral')

        return {
            'avg_sentiment': sum(scores) / len(scores),
            'median_sentiment': sorted(scores)[len(scores) // 2],
            'total_count': len(sentiments),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'avg_confidence': sum(confidences) / len(confidences),
            'bullish_percentage': (bullish / len(sentiments)) * 100,
            'bearish_percentage': (bearish / len(sentiments)) * 100,
            'neutral_percentage': (neutral / len(sentiments)) * 100
        }
