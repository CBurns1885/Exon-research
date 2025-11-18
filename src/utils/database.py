"""Database management for storing tweets and sentiment data"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json


class Database:
    """SQLite database manager for Fed sentiment analysis"""

    def __init__(self, db_path: str = "data/fed_sentiment.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def create_tables(self):
        """Create necessary database tables"""

        # Tweets table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT UNIQUE NOT NULL,
                author TEXT,
                text TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                likes INTEGER DEFAULT 0,
                retweets INTEGER DEFAULT 0,
                replies INTEGER DEFAULT 0,
                keywords TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Sentiment analysis table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                sentiment_category TEXT NOT NULL,
                confidence REAL,
                reasoning TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                llm_model TEXT,
                FOREIGN KEY (tweet_id) REFERENCES tweets(tweet_id)
            )
        """)

        # Federal Reserve decisions table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS fed_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_date DATE NOT NULL UNIQUE,
                decision_type TEXT NOT NULL,
                rate_change REAL NOT NULL,
                new_rate REAL NOT NULL,
                statement_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Daily sentiment aggregates table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL UNIQUE,
                avg_sentiment REAL NOT NULL,
                tweet_count INTEGER NOT NULL,
                bullish_count INTEGER DEFAULT 0,
                bearish_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Predictions table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date DATE NOT NULL,
                target_meeting_date DATE NOT NULL,
                predicted_action TEXT NOT NULL,
                confidence REAL NOT NULL,
                sentiment_avg REAL,
                sentiment_trend REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    def insert_tweet(self, tweet_data: Dict[str, Any]) -> Optional[int]:
        """Insert a tweet into the database"""
        try:
            self.cursor.execute("""
                INSERT OR IGNORE INTO tweets
                (tweet_id, author, text, created_at, likes, retweets, replies, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tweet_data.get('tweet_id'),
                tweet_data.get('author'),
                tweet_data.get('text'),
                tweet_data.get('created_at'),
                tweet_data.get('likes', 0),
                tweet_data.get('retweets', 0),
                tweet_data.get('replies', 0),
                json.dumps(tweet_data.get('keywords', []))
            ))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def insert_sentiment(self, sentiment_data: Dict[str, Any]) -> int:
        """Insert sentiment analysis result"""
        self.cursor.execute("""
            INSERT INTO sentiment_analysis
            (tweet_id, sentiment_score, sentiment_category, confidence, reasoning, llm_model)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sentiment_data.get('tweet_id'),
            sentiment_data.get('sentiment_score'),
            sentiment_data.get('sentiment_category'),
            sentiment_data.get('confidence'),
            sentiment_data.get('reasoning'),
            sentiment_data.get('llm_model')
        ))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_fed_decision(self, decision_data: Dict[str, Any]) -> Optional[int]:
        """Insert Federal Reserve decision"""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO fed_decisions
                (decision_date, decision_type, rate_change, new_rate, statement_summary)
                VALUES (?, ?, ?, ?, ?)
            """, (
                decision_data.get('decision_date'),
                decision_data.get('decision_type'),
                decision_data.get('rate_change'),
                decision_data.get('new_rate'),
                decision_data.get('statement_summary')
            ))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError as e:
            print(f"Error inserting Fed decision: {e}")
            return None

    def get_tweets_for_sentiment_analysis(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get tweets that haven't been analyzed yet"""
        self.cursor.execute("""
            SELECT t.* FROM tweets t
            LEFT JOIN sentiment_analysis s ON t.tweet_id = s.tweet_id
            WHERE s.id IS NULL
            ORDER BY t.created_at DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in self.cursor.fetchall()]

    def get_sentiment_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get sentiment data for a date range"""
        self.cursor.execute("""
            SELECT
                DATE(t.created_at) as date,
                AVG(s.sentiment_score) as avg_sentiment,
                COUNT(*) as tweet_count,
                SUM(CASE WHEN s.sentiment_category IN ('bullish', 'very_bullish') THEN 1 ELSE 0 END) as bullish_count,
                SUM(CASE WHEN s.sentiment_category IN ('bearish', 'very_bearish') THEN 1 ELSE 0 END) as bearish_count,
                SUM(CASE WHEN s.sentiment_category = 'neutral' THEN 1 ELSE 0 END) as neutral_count
            FROM tweets t
            JOIN sentiment_analysis s ON t.tweet_id = s.tweet_id
            WHERE DATE(t.created_at) BETWEEN ? AND ?
            GROUP BY DATE(t.created_at)
            ORDER BY date
        """, (start_date, end_date))

        return [dict(row) for row in self.cursor.fetchall()]

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
