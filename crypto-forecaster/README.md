# Cryptocurrency Price Forecasting System

Advanced cryptocurrency price forecasting system for the top 10 cryptocurrencies by market cap.

## Features

- **Real-time Data**: Fetches live price data from Coinbase API and CoinGecko
- **Web Scraping**: Backup data collection via web scraping
- **Top 10 Cryptocurrencies**: Bitcoin, Ethereum, BNB, Solana, XRP, Cardano, Dogecoin, Polygon, Polkadot, Chainlink
- **Advanced Forecasting**: Multiple ML models (Prophet, ARIMA, LSTM)
- **Interactive Dashboard**: Real-time price charts and forecasts
- **30-Day Predictions**: Short-term price forecasts with confidence intervals

## Cryptocurrencies Covered

1. 🪙 Bitcoin (BTC)
2. 💎 Ethereum (ETH)
3. ⚡ BNB (Binance Coin)
4. ☀️ Solana (SOL)
5. 💧 XRP (Ripple)
6. 🔷 Cardano (ADA)
7. 🐕 Dogecoin (DOGE)
8. 🟣 Polygon (MATIC)
9. ⚫ Polkadot (DOT)
10. 🔗 Chainlink (LINK)

## Quick Start

```bash
# Install dependencies
pip install -r crypto-requirements.txt

# Run the forecasting system
python run_crypto.py

# In another terminal, start the dashboard
cd crypto-dashboard
npm install
npm start
```

## Data Sources

- **Coinbase Advanced Trade API**: Historical OHLCV data
- **CoinGecko API**: Market cap, volume, and supplementary data
- **Web Scraping**: Backup for real-time prices

## Metrics Tracked

- Price (USD)
- Market Capitalization
- 24h Trading Volume
- Price Change (24h, 7d, 30d)
- Volatility
- RSI (Relative Strength Index)
- Moving Averages (7d, 30d, 90d)

## Technology Stack

- **Backend**: Python, FastAPI, Pandas, NumPy
- **Data Collection**: Coinbase API, CoinGecko API, BeautifulSoup
- **ML/Forecasting**: Prophet, ARIMA, Scikit-learn, TensorFlow/Keras
- **Database**: SQLite
- **Frontend**: React, Recharts, Tailwind CSS

## API Endpoints

- `GET /api/crypto/coins` - List all tracked cryptocurrencies
- `GET /api/crypto/prices/{symbol}` - Get historical prices
- `GET /api/crypto/forecast/{symbol}` - Get price forecasts
- `GET /api/crypto/dashboard` - Get dashboard summary
- `POST /api/crypto/update-prices` - Fetch latest prices
- `POST /api/crypto/generate-forecasts` - Generate new forecasts

## License

MIT
