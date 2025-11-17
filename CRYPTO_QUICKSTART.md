# Cryptocurrency Price Forecaster - Quick Start Guide

## Overview

This system provides **real-time cryptocurrency price forecasting** for the top 10 cryptocurrencies by market cap, using advanced machine learning models and live data from multiple sources.

## Cryptocurrencies Tracked

1. 🪙 **Bitcoin (BTC)**
2. 💎 **Ethereum (ETH)**
3. ⚡ **BNB** (Binance Coin)
4. ☀️ **Solana (SOL)**
5. 💧 **XRP** (Ripple)
6. 🔷 **Cardano (ADA)**
7. 🐕 **Dogecoin (DOGE)**
8. 🟣 **Polygon (MATIC)**
9. ⚫ **Polkadot (DOT)**
10. 🔗 **Chainlink (LINK)**

## One-Click Setup

### Prerequisites

- Python 3.8+
- pip

### Run the System

```bash
# Install dependencies (first time only)
pip install -r crypto-requirements.txt

# Run the complete forecasting pipeline
python run_crypto.py
```

That's it! This single command will:
1. ✓ Initialize the SQLite database
2. ✓ Fetch current cryptocurrency prices from CoinGecko API
3. ✓ Download 1 year of historical price data
4. ✓ Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)
5. ✓ Generate 30-day price forecasts using Prophet model
6. ✓ Start the FastAPI server at http://localhost:8001

## Using the API

### Interactive Documentation

Visit http://localhost:8001/docs for the full interactive API documentation.

### Key Endpoints

#### Get All Cryptocurrencies
```bash
curl http://localhost:8001/api/crypto/coins
```

#### Get Dashboard Summary
```bash
curl http://localhost:8001/api/crypto/dashboard
```

#### Get Historical Prices
```bash
# Get all historical prices for Bitcoin
curl http://localhost:8001/api/crypto/prices/BTC

# Get last 30 days
curl http://localhost:8001/api/crypto/prices/BTC?days=30
```

#### Get Price Forecast
```bash
# Get 30-day forecast for Ethereum
curl http://localhost:8001/api/crypto/forecast/ETH
```

#### Get Cryptocurrency Metadata
```bash
# Get current stats for Solana
curl http://localhost:8001/api/crypto/metadata/SOL
```

### Update Data

#### Refresh Current Prices
```bash
curl -X POST http://localhost:8001/api/crypto/update-prices
```

#### Fetch Fresh Historical Data
```bash
curl -X POST http://localhost:8001/api/crypto/fetch-historical
```

#### Regenerate Forecasts
```bash
# Using Prophet (default, best for crypto)
curl -X POST http://localhost:8001/api/crypto/generate-forecasts \
  -H "Content-Type: application/json" \
  -d '{"model_type": "prophet"}'

# Using ARIMA
curl -X POST http://localhost:8001/api/crypto/generate-forecasts \
  -H "Content-Type: application/json" \
  -d '{"model_type": "arima"}'

# Using Ensemble (combines Prophet + ARIMA)
curl -X POST http://localhost:8001/api/crypto/generate-forecasts \
  -H "Content-Type: application/json" \
  -d '{"model_type": "ensemble"}'
```

## Data Sources

### Primary: CoinGecko API (Free)
- **Current Prices**: Real-time cryptocurrency prices
- **Historical Data**: Up to 1 year of OHLCV data
- **Market Data**: Market cap, volume, price changes
- **No API key required** for basic usage

### Backup: Coinbase API
- Spot price verification
- Additional price data source

### Backup: Web Scraping
- CoinMarketCap price scraping as fallback
- Used only if API sources fail

## Features

### Price Forecasting
- **30-day forecasts** with confidence intervals
- **Prophet model** optimized for crypto volatility
- **Seasonal patterns** detection
- **95% confidence intervals** for predictions

### Technical Indicators
- Simple Moving Average (SMA) - 7-day, 30-day
- Exponential Moving Average (EMA) - 12-day, 26-day
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume trends

### Price Metrics
- Current price
- 24-hour change
- 7-day change
- 30-day change
- All-time high (ATH)
- All-time low (ATL)
- Market capitalization
- Trading volume

## Example API Responses

### Dashboard Data
```json
{
  "BTC": {
    "info": {
      "name": "Bitcoin",
      "symbol": "BTC",
      "emoji": "🪙",
      "color": "#F7931A"
    },
    "latest_price": {
      "close": 43250.50,
      "date": "2025-11-17",
      "volume": 25000000000
    },
    "metadata": {
      "current_price": 43250.50,
      "market_cap": 845000000000,
      "price_change_percentage_24h": 2.5,
      "price_change_percentage_7d": 5.8
    },
    "latest_forecast": {
      "predicted_price": 44100.00,
      "lower_bound": 41500.00,
      "upper_bound": 46700.00,
      "date": "2025-12-17"
    }
  }
}
```

### Price Forecast
```json
[
  {
    "symbol": "ETH",
    "date": "2025-11-18",
    "predicted_price": 2250.75,
    "lower_bound": 2100.50,
    "upper_bound": 2400.00,
    "model_type": "prophet"
  },
  ...
]
```

## Database Schema

The system uses SQLite with the following tables:

- **crypto_prices**: OHLCV historical price data
- **technical_indicators**: Calculated technical indicators
- **crypto_forecasts**: Price predictions
- **crypto_metadata**: Current stats and metadata

Database file: `crypto_forecaster.db`

## Customization

### Change Forecast Period

Edit `crypto-forecaster/.env`:
```bash
FORECAST_DAYS=60  # Default is 30
```

### Change Historical Data Period

```bash
HISTORICAL_DAYS=730  # Default is 365 (1 year)
```

### Use Different Port

```bash
CRYPTO_API_PORT=8002  # Default is 8001
```

## Technology Stack

- **Backend**: Python, FastAPI
- **Data Fetching**: Requests, BeautifulSoup4
- **Database**: SQLite, SQLAlchemy
- **Forecasting**: Prophet, ARIMA (statsmodels)
- **Technical Analysis**: pandas-ta, Custom indicators
- **APIs**: CoinGecko, Coinbase

## Performance

- **Initial Setup**: ~3-5 minutes (fetching historical data for 10 cryptocurrencies)
- **Price Updates**: ~15-30 seconds
- **Forecast Generation**: ~1-2 minutes
- **API Response Time**: <100ms for most endpoints

## Troubleshooting

### Port Already in Use
```bash
# Change port in crypto-forecaster/.env
CRYPTO_API_PORT=8002
```

### API Rate Limiting
The system includes automatic rate limiting and retry logic. If you encounter rate limits:
- Wait a few minutes before retrying
- Consider reducing the number of cryptocurrencies
- CoinGecko free tier: 50 calls/minute

### Missing Data
If some cryptocurrencies show no data:
- Check internet connection
- Verify CoinGecko API is accessible
- Try updating prices manually via the API

### Forecast Errors
If forecasts fail:
- Ensure sufficient historical data (minimum 30 days)
- Check database has price data
- Try different model_type (prophet, arima, ensemble)

## Next Steps

1. **Build a Dashboard**: Create a web frontend using the API
2. **Real-time Updates**: Implement WebSocket for live price updates
3. **Alerts**: Add price alert notifications
4. **More Cryptocurrencies**: Extend to top 20 or 50
5. **Advanced Models**: Add LSTM neural networks
6. **Trading Signals**: Generate buy/sell signals from indicators

## API Key Setup (Optional)

For higher rate limits, add API keys to `crypto-forecaster/.env`:

```bash
# CoinGecko Pro (optional)
COINGECKO_API_KEY=your_key_here

# Coinbase (optional)
COINBASE_API_KEY=your_key_here
COINBASE_API_SECRET=your_secret_here
```

The system works perfectly without API keys using free public endpoints.

## Support

- API Documentation: http://localhost:8001/docs
- Test Endpoint: http://localhost:8001/
- Health Check: Verify server is running

## License

MIT - Free to use and modify

---

**Happy Forecasting!** 🚀📈💰
