# Crypto Forecaster Dashboard

Beautiful React dashboard for cryptocurrency price forecasting with real-time data and ML predictions.

## Features

- 🎨 Dark theme with glassmorphism effects
- 📊 Real-time cryptocurrency prices
- 📈 Interactive price charts with Recharts
- 🔮 30-day ML-powered forecasts with confidence intervals
- 📉 Technical indicators (RSI, MACD, Bollinger Bands)
- 💹 Market statistics and performance metrics
- 📱 Fully responsive design
- ⚡ Fast and optimized

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The dashboard will open at http://localhost:3000

## Requirements

- Node.js 16+
- Backend API running at http://localhost:8001

## Project Structure

```
dashboard/
├── src/
│   ├── components/       # React components
│   │   ├── CryptoCard.js
│   │   ├── PriceChart.js
│   │   ├── TechnicalIndicators.js
│   │   ├── MarketStats.js
│   │   └── LoadingSpinner.js
│   ├── pages/           # Page components
│   │   ├── Dashboard.js
│   │   └── CryptoDetail.js
│   ├── services/        # API services
│   │   └── api.js
│   ├── utils/           # Utility functions
│   │   └── formatters.js
│   ├── App.js           # Main app component
│   ├── index.js         # Entry point
│   └── index.css        # Global styles
├── public/
│   └── index.html
└── package.json
```

## Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests

## Environment Variables

Create a `.env` file:

```bash
REACT_APP_API_URL=http://localhost:8001
```

## Features in Detail

### Dashboard Page
- Grid of cryptocurrency cards
- Real-time price updates
- Quick forecast previews
- Market statistics
- Refresh button for manual updates

### Crypto Detail Page
- Historical price charts (90 days)
- 30-day forecast with confidence intervals
- Technical indicators dashboard
- Market statistics
- All-time high/low information
- Forecast change percentage

### Components
- **CryptoCard**: Displays crypto overview with emoji, price, changes
- **PriceChart**: Interactive chart with historical + forecast data
- **TechnicalIndicators**: RSI, MACD, Bollinger Bands, SMAs, EMAs
- **MarketStats**: Market cap, volume, price changes, ATH/ATL
- **LoadingSpinner**: Smooth loading states

## Styling

- Tailwind CSS for utility-first styling
- Custom dark theme with gradient backgrounds
- Glassmorphism effects
- Smooth animations and transitions
- Responsive breakpoints

## Color Scheme

- Background: Dark slate (#0f172a, #1e293b)
- Cards: Dark with borders
- Accent colors:
  - Green: #10b981 (price up)
  - Red: #ef4444 (price down)
  - Blue: #3b82f6 (primary)
  - Purple: #8b5cf6 (forecast)

## API Integration

The dashboard connects to the FastAPI backend:

- GET `/api/crypto/dashboard` - All crypto data
- GET `/api/crypto/coins` - Crypto list
- GET `/api/crypto/prices/{symbol}` - Historical prices
- GET `/api/crypto/forecast/{symbol}` - Forecasts
- GET `/api/crypto/metadata/{symbol}` - Market stats
- POST `/api/crypto/update-prices` - Refresh prices

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance

- Lazy loading of components
- Optimized chart rendering
- Auto-refresh every 5 minutes
- Minimal re-renders

## License

MIT
