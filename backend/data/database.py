"""Database models and setup."""
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from backend.config import config

Base = declarative_base()

class EconomicData(Base):
    """Economic data model for historical data."""
    __tablename__ = "economic_data"

    id = Column(Integer, primary_key=True, index=True)
    country_code = Column(String(2), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    indicator = Column(String(50), index=True, nullable=False)
    value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "country_code": self.country_code,
            "date": self.date.isoformat() if self.date else None,
            "indicator": self.indicator,
            "value": self.value,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Forecast(Base):
    """Forecast model for predictions."""
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    country_code = Column(String(2), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    indicator = Column(String(50), index=True, nullable=False)
    forecast_value = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    model_type = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "country_code": self.country_code,
            "date": self.date.isoformat() if self.date else None,
            "indicator": self.indicator,
            "forecast_value": self.forecast_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat() if self.created_at else None
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
