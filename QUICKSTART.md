# Quick Start Guide

## Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- Git

## One-Click Setup and Run

### Backend Setup (Terminal 1)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the complete pipeline (generates data, forecasts, and starts API)
python run.py
```

This single command will:
1. Initialize the SQLite database
2. Generate 5 years of historical economic data
3. Create 6-month forecasts for all indicators
4. Start the FastAPI server at http://localhost:8000

### Dashboard Setup (Terminal 2)

```bash
# Navigate to dashboard directory
cd dashboard

# Install Node dependencies (first time only)
npm install

# Start the React dashboard
npm start
```

The dashboard will open automatically at http://localhost:3000

## What You'll See

### Dashboard (http://localhost:3000)
- Overview of all 5 countries (Kenya, Nigeria, Ghana, Rwanda, Egypt)
- Latest economic indicators for each country
- Click any country card to see detailed forecasts

### Country Detail Pages
- Historical data visualization
- 6-month forecast charts with confidence intervals
- All 7 economic indicators:
  - Interest Rate
  - FX Rate (vs USD)
  - GDP Growth
  - Capital Inflows
  - Capital Outflows
  - Imports
  - Exports

### API (http://localhost:8000)
- Interactive API documentation at http://localhost:8000/docs
- All endpoints available for custom queries

## API Endpoints

```bash
# Get all countries
curl http://localhost:8000/api/countries

# Get dashboard summary
curl http://localhost:8000/api/dashboard-data

# Get country data
curl http://localhost:8000/api/data/KE

# Get forecasts
curl http://localhost:8000/api/forecasts/KE

# Get country summary
curl http://localhost:8000/api/summary/KE
```

## Regenerating Data

If you want to regenerate the data or forecasts:

```bash
# Regenerate historical data
curl -X POST http://localhost:8000/api/generate-data

# Regenerate forecasts (using Prophet model)
curl -X POST http://localhost:8000/api/generate-forecasts \
  -H "Content-Type: application/json" \
  -d '{"model_type": "prophet"}'

# Try different models: "prophet", "arima", or "exponential"
```

## Project Structure

```
├── backend/
│   ├── api/main.py          # FastAPI application
│   ├── data/
│   │   ├── database.py      # Database models
│   │   └── generator.py     # Data generation
│   ├── models/
│   │   └── forecaster.py    # Forecasting models
│   └── config.py            # Configuration
├── dashboard/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   └── services/        # API service
│   └── package.json
├── run.py                   # One-click execution script
└── requirements.txt         # Python dependencies
```

## Troubleshooting

### Backend won't start
- Make sure port 8000 is available
- Check if all Python dependencies are installed: `pip install -r requirements.txt`

### Dashboard won't start
- Make sure port 3000 is available
- Run `npm install` in the dashboard directory
- Check if backend is running at http://localhost:8000

### No data showing
- Make sure the backend ran successfully and generated data
- Check browser console for errors
- Verify API is accessible at http://localhost:8000/api/dashboard-data

## Features

- **Real-time Forecasting**: Uses Prophet, ARIMA, and Exponential Smoothing models
- **Interactive Charts**: Historical data + 6-month forecasts with confidence intervals
- **Responsive Design**: Works on desktop, tablet, and mobile
- **REST API**: Full API access for custom integrations
- **One-Click Setup**: Single command to generate everything

## Technologies Used

- **Backend**: Python, FastAPI, SQLAlchemy, Prophet, Statsmodels
- **Frontend**: React, Recharts, Tailwind CSS
- **Database**: SQLite
- **Forecasting**: Facebook Prophet, ARIMA, Exponential Smoothing

## Next Steps

1. Explore different countries by clicking on their cards
2. View detailed forecasts for each indicator
3. Try regenerating forecasts with different models
4. Integrate the API into your own applications
5. Customize the indicators or add more countries

Enjoy exploring African frontier markets!
