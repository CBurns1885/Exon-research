"""FastAPI application for cryptocurrency price forecasting."""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.data.database import (
    get_db, init_db, CryptoPrice, CryptoForecast,
    CryptoMetadata, TechnicalIndicator
)
from backend.data.fetchers import CryptoDataFetcher
from backend.models.forecaster import CryptoPriceForecaster
from backend.utils.indicators import TechnicalIndicators
from backend.config import config
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Cryptocurrency Price Forecasting API",
    description="Real-time crypto price forecasting for top 10 cryptocurrencies",
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
class CryptoInfo(BaseModel):
    symbol: str
    name: str
    emoji: str
    color: str


class ForecastRequest(BaseModel):
    model_type: str = "prophet"


class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[dict] = None


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database."""
    print("Initializing crypto database...")
    init_db()
    print("Database initialized!")


# API Endpoints
@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint."""
    return StatusResponse(
        status="success",
        message="Cryptocurrency Price Forecasting API is running",
        details={
            "version": "1.0.0",
            "cryptocurrencies": len(config.CRYPTOCURRENCIES),
            "forecast_days": config.FORECAST_DAYS
        }
    )


@app.get("/api/crypto/coins", response_model=List[CryptoInfo])
async def get_coins():
    """Get list of all tracked cryptocurrencies."""
    coins = []
    for symbol, info in config.CRYPTOCURRENCIES.items():
        coins.append(CryptoInfo(
            symbol=symbol,
            name=info["name"],
            emoji=info["emoji"],
            color=info["color"]
        ))
    return coins


@app.get("/api/crypto/prices/{symbol}")
async def get_prices(
    symbol: str,
    days: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get historical prices for a cryptocurrency."""
    query = db.query(CryptoPrice).filter(
        CryptoPrice.symbol == symbol.upper()
    )

    if days:
        cutoff_date = datetime.now().date() - pd.Timedelta(days=days)
        query = query.filter(CryptoPrice.date >= cutoff_date)

    prices = query.order_by(CryptoPrice.date).all()

    if not prices:
        raise HTTPException(status_code=404, detail="No price data found")

    return [p.to_dict() for p in prices]


@app.get("/api/crypto/forecast/{symbol}")
async def get_forecast(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Get price forecast for a cryptocurrency."""
    forecasts = db.query(CryptoForecast).filter(
        CryptoForecast.symbol == symbol.upper()
    ).order_by(CryptoForecast.date).all()

    if not forecasts:
        raise HTTPException(status_code=404, detail="No forecasts found")

    return [f.to_dict() for f in forecasts]


@app.get("/api/crypto/metadata/{symbol}")
async def get_metadata(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Get cryptocurrency metadata and current stats."""
    metadata = db.query(CryptoMetadata).filter(
        CryptoMetadata.symbol == symbol.upper()
    ).first()

    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not found")

    return metadata.to_dict()


@app.get("/api/crypto/dashboard")
async def get_dashboard(db: Session = Depends(get_db)):
    """Get dashboard summary for all cryptocurrencies."""
    dashboard_data = {}

    for symbol in config.get_crypto_symbols():
        # Get latest price
        latest_price = db.query(CryptoPrice).filter(
            CryptoPrice.symbol == symbol
        ).order_by(CryptoPrice.date.desc()).first()

        # Get metadata
        metadata = db.query(CryptoMetadata).filter(
            CryptoMetadata.symbol == symbol
        ).first()

        # Get latest forecast
        latest_forecast = db.query(CryptoForecast).filter(
            CryptoForecast.symbol == symbol
        ).order_by(CryptoForecast.date.desc()).first()

        crypto_info = config.get_crypto_info(symbol)

        dashboard_data[symbol] = {
            "info": crypto_info,
            "latest_price": latest_price.to_dict() if latest_price else None,
            "metadata": metadata.to_dict() if metadata else None,
            "latest_forecast": latest_forecast.to_dict() if latest_forecast else None
        }

    return dashboard_data


@app.post("/api/crypto/update-prices", response_model=StatusResponse)
async def update_prices(db: Session = Depends(get_db)):
    """Fetch and update current prices for all cryptocurrencies."""
    try:
        fetcher = CryptoDataFetcher()

        # Get current prices
        current_prices = fetcher.get_current_prices_all()

        # Update metadata
        for symbol, data in current_prices.items():
            metadata = db.query(CryptoMetadata).filter(
                CryptoMetadata.symbol == symbol
            ).first()

            if metadata:
                # Update existing
                metadata.current_price = data.get("current_price")
                metadata.market_cap = data.get("market_cap")
                metadata.market_cap_rank = data.get("market_cap_rank")
                metadata.total_volume = data.get("total_volume")
                metadata.price_change_24h = data.get("price_change_24h")
                metadata.price_change_percentage_24h = data.get("price_change_percentage_24h")
                metadata.price_change_percentage_7d = data.get("price_change_percentage_7d")
                metadata.price_change_percentage_30d = data.get("price_change_percentage_30d")
                metadata.last_updated = datetime.utcnow()
            else:
                # Create new
                metadata = CryptoMetadata(
                    symbol=symbol,
                    name=data.get("name", config.get_crypto_name(symbol)),
                    current_price=data.get("current_price"),
                    market_cap=data.get("market_cap"),
                    market_cap_rank=data.get("market_cap_rank"),
                    total_volume=data.get("total_volume"),
                    price_change_24h=data.get("price_change_24h"),
                    price_change_percentage_24h=data.get("price_change_percentage_24h"),
                    price_change_percentage_7d=data.get("price_change_percentage_7d"),
                    price_change_percentage_30d=data.get("price_change_percentage_30d")
                )
                db.add(metadata)

        db.commit()

        return StatusResponse(
            status="success",
            message=f"Updated prices for {len(current_prices)} cryptocurrencies",
            details={"count": len(current_prices)}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crypto/fetch-historical", response_model=StatusResponse)
async def fetch_historical(db: Session = Depends(get_db)):
    """Fetch historical price data for all cryptocurrencies."""
    try:
        fetcher = CryptoDataFetcher()

        # Get historical data
        historical_data = fetcher.get_historical_data_all(days=config.HISTORICAL_DAYS)

        total_records = 0

        for symbol, df in historical_data.items():
            # Save to database
            for _, row in df.iterrows():
                # Check if record exists
                existing = db.query(CryptoPrice).filter(
                    CryptoPrice.symbol == symbol,
                    CryptoPrice.date == row['date']
                ).first()

                if not existing:
                    price_record = CryptoPrice(
                        symbol=symbol,
                        timestamp=row['timestamp'],
                        date=row['date'],
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']),
                        market_cap=float(row.get('market_cap', 0)) if pd.notna(row.get('market_cap')) else None
                    )
                    db.add(price_record)
                    total_records += 1

            # Calculate and save technical indicators
            if not df.empty:
                df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)

                for _, row in df_with_indicators.iterrows():
                    # Check if indicator exists
                    existing_ind = db.query(TechnicalIndicator).filter(
                        TechnicalIndicator.symbol == symbol,
                        TechnicalIndicator.date == row['date']
                    ).first()

                    if not existing_ind and pd.notna(row.get('rsi')):
                        indicator = TechnicalIndicator(
                            symbol=symbol,
                            date=row['date'],
                            sma_7=float(row['sma_7']) if pd.notna(row['sma_7']) else None,
                            sma_30=float(row['sma_30']) if pd.notna(row['sma_30']) else None,
                            ema_12=float(row['ema_12']) if pd.notna(row['ema_12']) else None,
                            ema_26=float(row['ema_26']) if pd.notna(row['ema_26']) else None,
                            rsi=float(row['rsi']) if pd.notna(row['rsi']) else None,
                            macd=float(row['macd']) if pd.notna(row['macd']) else None,
                            macd_signal=float(row['macd_signal']) if pd.notna(row['macd_signal']) else None,
                            macd_histogram=float(row['macd_histogram']) if pd.notna(row['macd_histogram']) else None,
                            bollinger_high=float(row['bollinger_high']) if pd.notna(row['bollinger_high']) else None,
                            bollinger_mid=float(row['bollinger_mid']) if pd.notna(row['bollinger_mid']) else None,
                            bollinger_low=float(row['bollinger_low']) if pd.notna(row['bollinger_low']) else None,
                            volume_sma=float(row['volume_sma']) if pd.notna(row['volume_sma']) else None
                        )
                        db.add(indicator)

        db.commit()

        return StatusResponse(
            status="success",
            message=f"Fetched historical data for {len(historical_data)} cryptocurrencies",
            details={
                "cryptocurrencies": len(historical_data),
                "total_records": total_records
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crypto/generate-forecasts", response_model=StatusResponse)
async def generate_forecasts(
    request: ForecastRequest,
    db: Session = Depends(get_db)
):
    """Generate price forecasts for all cryptocurrencies."""
    try:
        # Get historical data from database
        historical_data = {}

        for symbol in config.get_crypto_symbols():
            prices = db.query(CryptoPrice).filter(
                CryptoPrice.symbol == symbol
            ).order_by(CryptoPrice.date).all()

            if prices:
                df = pd.DataFrame([p.to_dict() for p in prices])
                historical_data[symbol] = df

        if not historical_data:
            raise HTTPException(
                status_code=400,
                detail="No historical data available. Fetch historical data first."
            )

        # Generate forecasts
        forecaster = CryptoPriceForecaster(forecast_days=config.FORECAST_DAYS)
        forecasts = forecaster.forecast_all_cryptos(historical_data, model_type=request.model_type)

        # Save forecasts
        count = forecaster.save_forecasts(forecasts, db)

        return StatusResponse(
            status="success",
            message=f"Generated forecasts for {len(forecasts)} cryptocurrencies using {request.model_type} model",
            details={
                "cryptocurrencies": len(forecasts),
                "total_forecasts": count,
                "model_type": request.model_type,
                "forecast_days": config.FORECAST_DAYS
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
