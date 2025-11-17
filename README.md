# African Frontier Markets Economic Forecasting Model

A comprehensive economic forecasting system for 5 African frontier markets, providing 6-month forecasts for key economic indicators.

## Countries Covered
- 🇰🇪 Kenya
- 🇳🇬 Nigeria
- 🇬🇭 Ghana
- 🇷🇼 Rwanda
- 🇪🇬 Egypt

## Key Indicators
- **Interest Rates** (Central Bank policy rates)
- **Foreign Exchange Rates** (vs USD)
- **Capital Flows** (FDI, Portfolio investments)
- **Trade Balance** (Imports vs Exports)
- **GDP Growth** (Quarterly)

## Features
- 6-month forecasting using time-series models
- Interactive dashboard with real-time visualizations
- One-click execution for complete forecasting pipeline
- Historical data analysis and trend visualization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete forecasting pipeline (one-click)
python run.py

# Start the dashboard
cd dashboard && npm install && npm start
```

## Project Structure
```
├── backend/
│   ├── data/              # Data collection and storage
│   ├── models/            # Forecasting models
│   ├── api/               # FastAPI backend
│   └── utils/             # Utilities
├── dashboard/             # React dashboard
├── run.py                 # One-click execution script
└── requirements.txt       # Python dependencies
```

## Technology Stack
- **Backend**: Python, FastAPI, Pandas, Scikit-learn, Prophet
- **Database**: SQLite
- **Frontend**: React, Recharts, Tailwind CSS
- **Forecasting**: ARIMA, Prophet, LSTM models
