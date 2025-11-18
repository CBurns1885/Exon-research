"""Database models for central bank forecasting."""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Text, Index, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import config

Base = declarative_base()


class CentralBankRate(Base):
    """Central bank policy rates."""
    __tablename__ = "central_bank_rates"

    id = Column(Integer, primary_key=True, index=True)
    bank_code = Column(String(10), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    rate = Column(Float, nullable=False)
    change = Column(Float, nullable=True)  # Change from previous rate
    decision_type = Column(String(20), nullable=True)  # "hike", "cut", "hold"

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_bank_date', 'bank_code', 'date'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "bank_code": self.bank_code,
            "date": self.date.isoformat() if self.date else None,
            "rate": self.rate,
            "change": self.change,
            "decision_type": self.decision_type,
        }


class CentralBankSpeech(Base):
    """Central bank speeches and statements."""
    __tablename__ = "central_bank_speeches"

    id = Column(Integer, primary_key=True, index=True)
    bank_code = Column(String(10), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    title = Column(String(500), nullable=False)
    speaker = Column(String(200), nullable=True)
    url = Column(String(1000), nullable=True)

    # Full text
    text_content = Column(Text, nullable=True)

    # NLP analysis results
    sentiment = Column(String(50), nullable=True)  # very_dovish, dovish, neutral, hawkish, very_hawkish
    sentiment_score = Column(Float, nullable=True)  # -1 (very dovish) to +1 (very hawkish)
    confidence = Column(Float, nullable=True)  # 0-1

    # Key insights from LLM
    key_phrases = Column(Text, nullable=True)  # JSON array
    main_topics = Column(Text, nullable=True)  # JSON array
    policy_signals = Column(Text, nullable=True)  # JSON object

    # Analysis metadata
    analyzed = Column(Boolean, default=False)
    analyzed_at = Column(DateTime, nullable=True)
    llm_model = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_speech_bank_date', 'bank_code', 'date'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "bank_code": self.bank_code,
            "date": self.date.isoformat() if self.date else None,
            "title": self.title,
            "speaker": self.speaker,
            "url": self.url,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "key_phrases": self.key_phrases,
            "main_topics": self.main_topics,
            "policy_signals": self.policy_signals,
            "analyzed": self.analyzed,
        }


class EconomicIndicator(Base):
    """Economic indicators for central bank regions."""
    __tablename__ = "economic_indicators"

    id = Column(Integer, primary_key=True, index=True)
    bank_code = Column(String(10), index=True, nullable=False)
    indicator_type = Column(String(100), index=True, nullable=False)  # CPI, unemployment, GDP, etc.
    date = Column(Date, index=True, nullable=False)
    value = Column(Float, nullable=False)

    # Metadata
    source = Column(String(200), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_indicator_bank_type_date', 'bank_code', 'indicator_type', 'date'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "bank_code": self.bank_code,
            "indicator_type": self.indicator_type,
            "date": self.date.isoformat() if self.date else None,
            "value": self.value,
            "source": self.source,
        }


class RateForecast(Base):
    """Interest rate forecasts."""
    __tablename__ = "rate_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    bank_code = Column(String(10), index=True, nullable=False)
    forecast_date = Column(Date, index=True, nullable=False)  # Date of the forecast
    target_date = Column(Date, index=True, nullable=False)  # Date being forecasted

    predicted_rate = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)

    # Probability distribution (for different scenarios)
    prob_hike = Column(Float, nullable=True)  # Probability of rate hike
    prob_hold = Column(Float, nullable=True)  # Probability of hold
    prob_cut = Column(Float, nullable=True)  # Probability of rate cut

    # Model metadata
    model_type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=True)

    # Feature importance
    sentiment_weight = Column(Float, nullable=True)
    economic_weight = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_forecast_bank_dates', 'bank_code', 'forecast_date', 'target_date'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "bank_code": self.bank_code,
            "forecast_date": self.forecast_date.isoformat() if self.forecast_date else None,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "predicted_rate": self.predicted_rate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "prob_hike": self.prob_hike,
            "prob_hold": self.prob_hold,
            "prob_cut": self.prob_cut,
            "model_type": self.model_type,
            "confidence": self.confidence,
        }


class MeetingCalendar(Base):
    """Central bank meeting calendar."""
    __tablename__ = "meeting_calendar"

    id = Column(Integer, primary_key=True, index=True)
    bank_code = Column(String(10), index=True, nullable=False)
    meeting_date = Column(Date, index=True, nullable=False)
    meeting_type = Column(String(100), nullable=True)  # Regular, emergency, etc.

    # Actual outcome (filled after meeting)
    rate_decision = Column(Float, nullable=True)
    decision_type = Column(String(20), nullable=True)  # hike, cut, hold

    # Minutes/statement
    statement_url = Column(String(1000), nullable=True)
    minutes_url = Column(String(1000), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_meeting_bank_date', 'bank_code', 'meeting_date'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "bank_code": self.bank_code,
            "meeting_date": self.meeting_date.isoformat() if self.meeting_date else None,
            "meeting_type": self.meeting_type,
            "rate_decision": self.rate_decision,
            "decision_type": self.decision_type,
            "statement_url": self.statement_url,
        }


# Database setup
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in config.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
