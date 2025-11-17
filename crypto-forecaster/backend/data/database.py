"""Database models for cryptocurrency forecasting."""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import config

Base = declarative_base()


class CryptoPrice(Base):
    """Cryptocurrency price data model."""
    __tablename__ = "crypto_prices"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)

    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Additional metrics
    market_cap = Column(Float, nullable=True)
    total_volume = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite index for faster queries
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date'),
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "date": self.date.isoformat() if self.date else None,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "market_cap": self.market_cap,
            "total_volume": self.total_volume,
        }


class TechnicalIndicator(Base):
    """Technical indicators for cryptocurrencies."""
    __tablename__ = "technical_indicators"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)

    # Moving Averages
    sma_7 = Column(Float, nullable=True)
    sma_30 = Column(Float, nullable=True)
    ema_12 = Column(Float, nullable=True)
    ema_26 = Column(Float, nullable=True)

    # Momentum Indicators
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_histogram = Column(Float, nullable=True)

    # Bollinger Bands
    bollinger_high = Column(Float, nullable=True)
    bollinger_mid = Column(Float, nullable=True)
    bollinger_low = Column(Float, nullable=True)

    # Volume
    volume_sma = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_indicator_symbol_date', 'symbol', 'date'),
    )

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "date": self.date.isoformat() if self.date else None,
            "sma_7": self.sma_7,
            "sma_30": self.sma_30,
            "ema_12": self.ema_12,
            "ema_26": self.ema_26,
            "rsi": self.rsi,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "bollinger_high": self.bollinger_high,
            "bollinger_mid": self.bollinger_mid,
            "bollinger_low": self.bollinger_low,
            "volume_sma": self.volume_sma,
        }


class CryptoForecast(Base):
    """Cryptocurrency price forecasts."""
    __tablename__ = "crypto_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)

    # Forecast values
    predicted_price = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)

    # Model metadata
    model_type = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_forecast_symbol_date', 'symbol', 'date'),
    )

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "date": self.date.isoformat() if self.date else None,
            "predicted_price": self.predicted_price,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "model_type": self.model_type,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CryptoMetadata(Base):
    """Cryptocurrency metadata and current stats."""
    __tablename__ = "crypto_metadata"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)

    # Current stats
    current_price = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    market_cap_rank = Column(Integer, nullable=True)
    total_volume = Column(Float, nullable=True)

    # Price changes
    price_change_24h = Column(Float, nullable=True)
    price_change_percentage_24h = Column(Float, nullable=True)
    price_change_percentage_7d = Column(Float, nullable=True)
    price_change_percentage_30d = Column(Float, nullable=True)

    # All-time data
    ath = Column(Float, nullable=True)  # All-time high
    ath_date = Column(DateTime, nullable=True)
    atl = Column(Float, nullable=True)  # All-time low
    atl_date = Column(DateTime, nullable=True)

    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "name": self.name,
            "current_price": self.current_price,
            "market_cap": self.market_cap,
            "market_cap_rank": self.market_cap_rank,
            "total_volume": self.total_volume,
            "price_change_24h": self.price_change_24h,
            "price_change_percentage_24h": self.price_change_percentage_24h,
            "price_change_percentage_7d": self.price_change_percentage_7d,
            "price_change_percentage_30d": self.price_change_percentage_30d,
            "ath": self.ath,
            "ath_date": self.ath_date.isoformat() if self.ath_date else None,
            "atl": self.atl,
            "atl_date": self.atl_date.isoformat() if self.atl_date else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
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
