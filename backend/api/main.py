"""FastAPI application for African Economy Forecasting."""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date
import pandas as pd

from backend.data.database import get_db, init_db, EconomicData, Forecast
from backend.data.generator import EconomicDataGenerator
from backend.models.forecaster import EconomicForecaster
from backend.config import config

# Initialize FastAPI app
app = FastAPI(
    title="African Economy Forecasting API",
    description="Economic forecasting for African frontier markets",
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


# Pydantic models for API
from pydantic import BaseModel


class CountryInfo(BaseModel):
    """Country information model."""
    code: str
    name: str
    currency: str
    region: str
    emoji: str


class EconomicDataPoint(BaseModel):
    """Economic data point model."""
    id: int
    country_code: str
    date: date
    indicator: str
    value: float


class ForecastPoint(BaseModel):
    """Forecast point model."""
    id: int
    country_code: str
    date: date
    indicator: str
    forecast_value: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    model_type: str


class ForecastRequest(BaseModel):
    """Forecast generation request."""
    model_type: str = "prophet"  # prophet, arima, exponential


class StatusResponse(BaseModel):
    """Status response model."""
    status: str
    message: str
    details: Optional[dict] = None


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and generate initial data."""
    print("Initializing database...")
    init_db()
    print("Database initialized!")


# API Endpoints

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint with API information."""
    return StatusResponse(
        status="success",
        message="African Economy Forecasting API is running",
        details={
            "version": "1.0.0",
            "countries": len(config.COUNTRIES),
            "indicators": len(config.INDICATORS)
        }
    )


@app.get("/api/countries", response_model=List[CountryInfo])
async def get_countries():
    """Get list of all countries."""
    countries = []
    for code, info in config.COUNTRIES.items():
        countries.append(CountryInfo(
            code=code,
            name=info["name"],
            currency=info["currency"],
            region=info["region"],
            emoji=info["emoji"]
        ))
    return countries


@app.get("/api/indicators")
async def get_indicators():
    """Get list of all indicators."""
    return {
        "indicators": config.INDICATORS
    }


@app.get("/api/data/{country_code}")
async def get_country_data(
    country_code: str,
    indicator: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """Get historical data for a country."""
    query = db.query(EconomicData).filter(
        EconomicData.country_code == country_code.upper()
    )

    if indicator:
        query = query.filter(EconomicData.indicator == indicator)

    if start_date:
        query = query.filter(EconomicData.date >= start_date)

    if end_date:
        query = query.filter(EconomicData.date <= end_date)

    data = query.order_by(EconomicData.date).all()

    if not data:
        raise HTTPException(status_code=404, message="No data found")

    return [d.to_dict() for d in data]


@app.get("/api/forecasts/{country_code}")
async def get_country_forecasts(
    country_code: str,
    indicator: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get forecasts for a country."""
    query = db.query(Forecast).filter(
        Forecast.country_code == country_code.upper()
    )

    if indicator:
        query = query.filter(Forecast.indicator == indicator)

    forecasts = query.order_by(Forecast.date).all()

    if not forecasts:
        raise HTTPException(status_code=404, detail="No forecasts found")

    return [f.to_dict() for f in forecasts]


@app.get("/api/data")
async def get_all_data(
    indicator: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """Get all historical data."""
    query = db.query(EconomicData)

    if indicator:
        query = query.filter(EconomicData.indicator == indicator)

    if start_date:
        query = query.filter(EconomicData.date >= start_date)

    if end_date:
        query = query.filter(EconomicData.date <= end_date)

    data = query.order_by(EconomicData.country_code, EconomicData.date).all()

    return [d.to_dict() for d in data]


@app.get("/api/forecasts")
async def get_all_forecasts(
    indicator: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all forecasts."""
    query = db.query(Forecast)

    if indicator:
        query = query.filter(Forecast.indicator == indicator)

    forecasts = query.order_by(Forecast.country_code, Forecast.date).all()

    return [f.to_dict() for f in forecasts]


@app.post("/api/generate-data", response_model=StatusResponse)
async def generate_data(db: Session = Depends(get_db)):
    """Generate synthetic historical data."""
    try:
        generator = EconomicDataGenerator(years=config.HISTORICAL_YEARS)
        count = generator.save_to_database(db)

        return StatusResponse(
            status="success",
            message=f"Generated {count} data points",
            details={"count": count}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-forecasts", response_model=StatusResponse)
async def generate_forecasts(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Generate forecasts for all countries and indicators."""
    try:
        # Get historical data
        data_query = db.query(EconomicData).all()

        if not data_query:
            raise HTTPException(
                status_code=400,
                detail="No historical data available. Generate data first."
            )

        # Convert to DataFrame
        data_list = [d.to_dict() for d in data_query]
        df = pd.DataFrame(data_list)

        # Generate forecasts
        forecaster = EconomicForecaster(forecast_months=config.FORECAST_MONTHS)
        forecasts_df = forecaster.forecast_all(df, model_type=request.model_type)

        if forecasts_df.empty:
            raise HTTPException(status_code=500, detail="Failed to generate forecasts")

        # Save to database
        count = forecaster.save_forecasts(forecasts_df, db)

        return StatusResponse(
            status="success",
            message=f"Generated {count} forecasts using {request.model_type} model",
            details={
                "count": count,
                "model_type": request.model_type,
                "forecast_months": config.FORECAST_MONTHS
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/summary/{country_code}")
async def get_country_summary(
    country_code: str,
    db: Session = Depends(get_db)
):
    """Get summary statistics for a country."""
    country_code = country_code.upper()

    if country_code not in config.COUNTRIES:
        raise HTTPException(status_code=404, detail="Country not found")

    # Get latest data
    latest_data = {}
    for indicator in config.INDICATORS:
        data = db.query(EconomicData).filter(
            EconomicData.country_code == country_code,
            EconomicData.indicator == indicator
        ).order_by(EconomicData.date.desc()).first()

        if data:
            latest_data[indicator] = {
                "value": data.value,
                "date": data.date.isoformat()
            }

    # Get forecasts
    latest_forecasts = {}
    for indicator in config.INDICATORS:
        forecast = db.query(Forecast).filter(
            Forecast.country_code == country_code,
            Forecast.indicator == indicator
        ).order_by(Forecast.date.desc()).first()

        if forecast:
            latest_forecasts[indicator] = {
                "forecast_value": forecast.forecast_value,
                "date": forecast.date.isoformat(),
                "lower_bound": forecast.lower_bound,
                "upper_bound": forecast.upper_bound
            }

    return {
        "country": config.COUNTRIES[country_code],
        "latest_data": latest_data,
        "latest_forecasts": latest_forecasts
    }


@app.get("/api/dashboard-data")
async def get_dashboard_data(db: Session = Depends(get_db)):
    """Get aggregated data for dashboard."""
    dashboard_data = {}

    for country_code in config.get_country_codes():
        country_info = config.COUNTRIES[country_code]

        # Get latest values
        latest_values = {}
        for indicator in config.INDICATORS:
            data = db.query(EconomicData).filter(
                EconomicData.country_code == country_code,
                EconomicData.indicator == indicator
            ).order_by(EconomicData.date.desc()).first()

            if data:
                latest_values[indicator] = data.value

        # Get latest forecasts
        latest_forecasts = {}
        for indicator in config.INDICATORS:
            forecast = db.query(Forecast).filter(
                Forecast.country_code == country_code,
                Forecast.indicator == indicator
            ).order_by(Forecast.date.desc()).first()

            if forecast:
                latest_forecasts[indicator] = forecast.forecast_value

        dashboard_data[country_code] = {
            "country": country_info,
            "latest_values": latest_values,
            "latest_forecasts": latest_forecasts
        }

    return dashboard_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
