"""FastAPI application for central bank rate forecasting."""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.data.database import (
    get_db, init_db, CentralBankRate, CentralBankSpeech,
    EconomicIndicator, RateForecast, MeetingCalendar
)
from backend.scrapers.speech_scraper import SpeechScraper
from backend.nlp.sentiment_analyzer import SentimentAnalyzer
from backend.models.rate_forecaster import RateForecaster
from backend.config import config
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Central Bank Rate Forecasting API",
    description="Interest rate forecasting using LLM-powered NLP analysis of central bank communications",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class BankInfo(BaseModel):
    code: str
    name: str
    country: str
    currency: str
    emoji: str
    current_chair: str


class ForecastRequest(BaseModel):
    bank_codes: Optional[List[str]] = None
    horizons: Optional[List[int]] = None


class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[dict] = None


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database."""
    print("Initializing central bank database...")
    init_db()
    print("Database initialized!")


# API Endpoints
@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint."""
    return StatusResponse(
        status="success",
        message="Central Bank Rate Forecasting API is running",
        details={
            "version": "1.0.0",
            "banks": len(config.CENTRAL_BANKS),
            "llm_provider": config.LLM_PROVIDER
        }
    )


@app.get("/api/central-banks", response_model=List[BankInfo])
async def get_banks():
    """Get list of all tracked central banks."""
    banks = []
    for code, info in config.CENTRAL_BANKS.items():
        banks.append(BankInfo(
            code=code,
            name=info["name"],
            country=info["country"],
            currency=info["currency"],
            emoji=info["emoji"],
            current_chair=info["current_chair"]
        ))
    return banks


@app.get("/api/central-banks/{bank_code}/current-rate")
async def get_current_rate(
    bank_code: str,
    db: Session = Depends(get_db)
):
    """Get current policy rate for a central bank."""
    latest_rate = db.query(CentralBankRate).filter(
        CentralBankRate.bank_code == bank_code.upper()
    ).order_by(CentralBankRate.date.desc()).first()

    if not latest_rate:
        raise HTTPException(status_code=404, detail="No rate data found")

    return latest_rate.to_dict()


@app.get("/api/central-banks/{bank_code}/rates")
async def get_rate_history(
    bank_code: str,
    days: Optional[int] = 365,
    db: Session = Depends(get_db)
):
    """Get historical rates for a central bank."""
    cutoff_date = datetime.now().date() - timedelta(days=days) if days else None

    query = db.query(CentralBankRate).filter(
        CentralBankRate.bank_code == bank_code.upper()
    )

    if cutoff_date:
        query = query.filter(CentralBankRate.date >= cutoff_date)

    rates = query.order_by(CentralBankRate.date).all()

    if not rates:
        raise HTTPException(status_code=404, detail="No rate data found")

    return [r.to_dict() for r in rates]


@app.get("/api/central-banks/{bank_code}/speeches")
async def get_speeches(
    bank_code: str,
    limit: int = 20,
    analyzed_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get speeches for a central bank."""
    query = db.query(CentralBankSpeech).filter(
        CentralBankSpeech.bank_code == bank_code.upper()
    )

    if analyzed_only:
        query = query.filter(CentralBankSpeech.analyzed == True)

    speeches = query.order_by(CentralBankSpeech.date.desc()).limit(limit).all()

    return [s.to_dict() for s in speeches]


@app.get("/api/central-banks/{bank_code}/sentiment")
async def get_sentiment_analysis(
    bank_code: str,
    days: int = 90,
    db: Session = Depends(get_db)
):
    """Get aggregated sentiment analysis."""
    cutoff_date = datetime.now().date() - timedelta(days=days)

    speeches = db.query(CentralBankSpeech).filter(
        CentralBankSpeech.bank_code == bank_code.upper(),
        CentralBankSpeech.date >= cutoff_date,
        CentralBankSpeech.analyzed == True
    ).order_by(CentralBankSpeech.date.desc()).all()

    if not speeches:
        return {
            "bank_code": bank_code.upper(),
            "period_days": days,
            "avg_sentiment_score": None,
            "sentiment_trend": "insufficient_data",
            "speeches_analyzed": 0
        }

    # Calculate average sentiment
    sentiments = [s.sentiment_score for s in speeches if s.sentiment_score is not None]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

    # Determine trend
    if avg_sentiment > 0.3:
        trend = "hawkish"
    elif avg_sentiment < -0.3:
        trend = "dovish"
    else:
        trend = "neutral"

    # Recent vs earlier comparison
    mid_point = len(sentiments) // 2
    recent_avg = sum(sentiments[:mid_point]) / mid_point if mid_point > 0 else 0
    earlier_avg = sum(sentiments[mid_point:]) / (len(sentiments) - mid_point) if mid_point < len(sentiments) else 0

    return {
        "bank_code": bank_code.upper(),
        "period_days": days,
        "avg_sentiment_score": round(avg_sentiment, 3),
        "sentiment_trend": trend,
        "recent_sentiment": round(recent_avg, 3),
        "earlier_sentiment": round(earlier_avg, 3),
        "sentiment_shift": round(recent_avg - earlier_avg, 3),
        "speeches_analyzed": len(speeches),
        "latest_speech": speeches[0].to_dict() if speeches else None
    }


@app.get("/api/central-banks/{bank_code}/forecast")
async def get_forecast(
    bank_code: str,
    db: Session = Depends(get_db)
):
    """Get rate forecasts for a central bank."""
    # Get latest forecasts
    forecasts = db.query(RateForecast).filter(
        RateForecast.bank_code == bank_code.upper()
    ).order_by(RateForecast.forecast_date.desc(), RateForecast.target_date).limit(10).all()

    if not forecasts:
        raise HTTPException(status_code=404, detail="No forecasts found")

    return [f.to_dict() for f in forecasts]


@app.get("/api/central-banks/{bank_code}/indicators")
async def get_indicators(
    bank_code: str,
    indicator_type: Optional[str] = None,
    days: int = 365,
    db: Session = Depends(get_db)
):
    """Get economic indicators for a central bank's region."""
    cutoff_date = datetime.now().date() - timedelta(days=days)

    query = db.query(EconomicIndicator).filter(
        EconomicIndicator.bank_code == bank_code.upper(),
        EconomicIndicator.date >= cutoff_date
    )

    if indicator_type:
        query = query.filter(EconomicIndicator.indicator_type == indicator_type)

    indicators = query.order_by(EconomicIndicator.date.desc()).all()

    return [i.to_dict() for i in indicators]


@app.post("/api/scrape-speeches", response_model=StatusResponse)
async def scrape_speeches(
    background_tasks: BackgroundTasks,
    bank_code: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Scrape latest speeches from central bank websites."""
    try:
        scraper = SpeechScraper()

        if bank_code:
            # Scrape specific bank
            speeches = {bank_code.upper(): []}
            if bank_code.upper() == "FED":
                speeches["FED"] = scraper.scrape_fed_speeches()
            elif bank_code.upper() == "ECB":
                speeches["ECB"] = scraper.scrape_ecb_speeches()
            elif bank_code.upper() == "BOE":
                speeches["BOE"] = scraper.scrape_boe_speeches()
        else:
            # Scrape all banks
            speeches = scraper.scrape_all_banks(max_per_bank=config.MAX_SPEECHES_PER_SCRAPE)

        # Save to database
        count = scraper.save_speeches_to_db(speeches, db)

        return StatusResponse(
            status="success",
            message=f"Scraped and saved {count} new speeches",
            details={"speeches_saved": count}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-sentiment", response_model=StatusResponse)
async def analyze_sentiment(
    background_tasks: BackgroundTasks,
    bank_code: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Analyze sentiment of unanalyzed speeches."""
    try:
        # Get unanalyzed speeches
        query = db.query(CentralBankSpeech).filter(
            CentralBankSpeech.analyzed == False,
            CentralBankSpeech.text_content != None
        )

        if bank_code:
            query = query.filter(CentralBankSpeech.bank_code == bank_code.upper())

        speeches = query.limit(20).all()

        if not speeches:
            return StatusResponse(
                status="success",
                message="No unanalyzed speeches found",
                details={"analyzed": 0}
            )

        # Analyze speeches
        analyzer = SentimentAnalyzer()

        for speech in speeches:
            result = analyzer.analyze_speech(
                text=speech.text_content or "",
                title=speech.title,
                speaker=speech.speaker or ""
            )

            # Update speech record
            speech.sentiment = result['sentiment']
            speech.sentiment_score = result['sentiment_score']
            speech.confidence = result['confidence']
            speech.key_phrases = result['key_phrases']
            speech.main_topics = result['main_topics']
            speech.policy_signals = result['policy_signals']
            speech.analyzed = True
            speech.analyzed_at = datetime.utcnow()
            speech.llm_model = f"{config.LLM_PROVIDER}:{config.LLM_MODEL}"

        db.commit()

        return StatusResponse(
            status="success",
            message=f"Analyzed {len(speeches)} speeches",
            details={"analyzed": len(speeches), "llm_model": config.LLM_MODEL}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-forecasts", response_model=StatusResponse)
async def generate_forecasts(
    request: ForecastRequest,
    db: Session = Depends(get_db)
):
    """Generate rate forecasts for central banks."""
    try:
        bank_codes = request.bank_codes or config.get_bank_codes()
        horizons = request.horizons or config.FORECAST_MONTHS

        forecaster = RateForecaster()
        total_forecasts = 0

        for bank_code in bank_codes:
            # Get historical rates
            rates = db.query(CentralBankRate).filter(
                CentralBankRate.bank_code == bank_code
            ).order_by(CentralBankRate.date).all()

            if not rates:
                continue

            rates_df = pd.DataFrame([r.to_dict() for r in rates])

            # Get sentiment data
            speeches = db.query(CentralBankSpeech).filter(
                CentralBankSpeech.bank_code == bank_code,
                CentralBankSpeech.analyzed == True
            ).all()

            sentiment_df = pd.DataFrame([
                {"date": s.date, "sentiment_score": s.sentiment_score}
                for s in speeches if s.sentiment_score is not None
            ])

            # Get economic indicators
            indicators = db.query(EconomicIndicator).filter(
                EconomicIndicator.bank_code == bank_code
            ).all()

            indicators_df = pd.DataFrame([i.to_dict() for i in indicators])

            # Generate forecasts
            forecasts = forecaster.forecast_bank_rates(
                bank_code=bank_code,
                historical_rates=rates_df,
                sentiment_df=sentiment_df,
                indicators_df=indicators_df,
                forecast_horizons=horizons
            )

            # Save forecasts
            if forecasts:
                count = forecaster.save_forecasts(forecasts, db)
                total_forecasts += count

        return StatusResponse(
            status="success",
            message=f"Generated {total_forecasts} forecasts",
            details={
                "total_forecasts": total_forecasts,
                "banks": len(bank_codes),
                "horizons": horizons
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
