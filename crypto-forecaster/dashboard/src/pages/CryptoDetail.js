/**
 * CryptoDetail page - Detailed view of a single cryptocurrency
 */
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FiArrowLeft, FiRefreshCw } from 'react-icons/fi';
import PriceChart from '../components/PriceChart';
import MarketStats from '../components/MarketStats';
import TechnicalIndicators from '../components/TechnicalIndicators';
import LoadingSpinner from '../components/LoadingSpinner';
import cryptoApi from '../services/api';
import { formatPrice, formatPercent, getPriceChangeClass } from '../utils/formatters';

const CryptoDetail = () => {
  const { symbol } = useParams();
  const navigate = useNavigate();

  const [metadata, setMetadata] = useState(null);
  const [prices, setPrices] = useState([]);
  const [forecasts, setForecasts] = useState([]);
  const [indicators, setIndicators] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [cryptoInfo, setCryptoInfo] = useState(null);

  useEffect(() => {
    loadCryptoData();
  }, [symbol]);

  const loadCryptoData = async () => {
    try {
      setLoading(true);

      // Load all data in parallel
      const [coinsData, metadataData, pricesData, forecastsData] = await Promise.all([
        cryptoApi.getCoins(),
        cryptoApi.getMetadata(symbol.toUpperCase()).catch(() => null),
        cryptoApi.getPrices(symbol.toUpperCase(), 90).catch(() => []),
        cryptoApi.getForecast(symbol.toUpperCase()).catch(() => []),
      ]);

      // Find crypto info
      const info = coinsData.find(c => c.symbol === symbol.toUpperCase());
      setCryptoInfo(info);

      setMetadata(metadataData);
      setPrices(pricesData);
      setForecasts(forecastsData);

      // Extract indicators from prices (if available)
      // In a real implementation, you'd fetch these separately
      setIndicators(pricesData.slice(-30)); // Last 30 days for indicators

      setError(null);
    } catch (err) {
      setError('Failed to load cryptocurrency data');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <LoadingSpinner message={`Loading ${symbol.toUpperCase()} data...`} />;
  }

  if (error || !cryptoInfo) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center max-w-md mx-4">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-white mb-2">Error Loading Data</h2>
          <p className="text-gray-400 mb-6">{error || 'Cryptocurrency not found'}</p>
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const currentPrice = prices.length > 0 ? prices[prices.length - 1].close : metadata?.current_price;
  const priceChange24h = metadata?.price_change_percentage_24h;

  // Calculate forecast vs current
  const latestForecast = forecasts.length > 0 ? forecasts[forecasts.length - 1] : null;
  const forecastChange = currentPrice && latestForecast
    ? ((latestForecast.predicted_price - currentPrice) / currentPrice) * 100
    : null;

  return (
    <div className="min-h-screen pb-12">
      {/* Header */}
      <div className="bg-gradient-to-r from-dark-card to-dark-bg border-b border-dark-border">
        <div className="container mx-auto px-4 py-8">
          <button
            onClick={() => navigate('/')}
            className="flex items-center space-x-2 text-blue-400 hover:text-blue-300 mb-4 transition-colors"
          >
            <FiArrowLeft />
            <span>Back to Dashboard</span>
          </button>

          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center space-x-4">
              <div
                className="text-6xl w-20 h-20 flex items-center justify-center rounded-full"
                style={{ backgroundColor: `${cryptoInfo.color}20` }}
              >
                {cryptoInfo.emoji}
              </div>
              <div>
                <h1 className="text-4xl font-bold text-white mb-1">
                  {cryptoInfo.name}
                </h1>
                <p className="text-xl text-gray-400">{symbol.toUpperCase()}</p>
              </div>
            </div>

            <div className="text-right">
              <div className="text-4xl font-bold text-white mb-1">
                {formatPrice(currentPrice)}
              </div>
              {priceChange24h !== null && priceChange24h !== undefined && (
                <div className={`text-lg ${getPriceChangeClass(priceChange24h)}`}>
                  {priceChange24h >= 0 ? '↑' : '↓'} {formatPercent(Math.abs(priceChange24h), false)} (24h)
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {/* Forecast Summary */}
        {latestForecast && (
          <div className="mb-8 bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-lg p-6 border border-purple-500/30">
            <h3 className="text-xl font-semibold text-white mb-4">30-Day Forecast Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <p className="text-sm text-gray-400 mb-1">Predicted Price</p>
                <p className="text-3xl font-bold text-purple-400">
                  {formatPrice(latestForecast.predicted_price)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Expected Change</p>
                <p className={`text-3xl font-bold ${getPriceChangeClass(forecastChange)}`}>
                  {forecastChange !== null ? formatPercent(forecastChange, true) : 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Confidence Range</p>
                <p className="text-lg font-bold text-blue-400">
                  {formatPrice(latestForecast.lower_bound)} - {formatPrice(latestForecast.upper_bound)}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Price Chart */}
        <div className="mb-8">
          <PriceChart
            historicalData={prices}
            forecastData={forecasts}
            cryptoName={cryptoInfo.name}
            cryptoColor={cryptoInfo.color}
          />
        </div>

        {/* Market Stats */}
        <div className="mb-8">
          <MarketStats
            metadata={metadata}
            latestPrice={prices.length > 0 ? prices[prices.length - 1] : null}
          />
        </div>

        {/* Technical Indicators */}
        {indicators.length > 0 && (
          <div className="mb-8">
            <TechnicalIndicators indicators={indicators} />
          </div>
        )}

        {/* Model Information */}
        <div className="bg-blue-900/20 rounded-lg p-6 border border-blue-500/30">
          <h3 className="text-lg font-semibold text-white mb-3">About the Forecasts</h3>
          <div className="space-y-2 text-sm text-gray-400">
            <p>
              <span className="text-white font-semibold">Model:</span> Facebook Prophet optimized for cryptocurrency volatility with weekly and monthly seasonality detection.
            </p>
            <p>
              <span className="text-white font-semibold">Confidence Intervals:</span> 95% confidence bands showing the range where the actual price is expected to fall.
            </p>
            <p>
              <span className="text-white font-semibold">Historical Data:</span> Based on 1 year of historical OHLCV (Open, High, Low, Close, Volume) data from CoinGecko.
            </p>
            <p>
              <span className="text-white font-semibold">Update Frequency:</span> Forecasts are regenerated hourly with the latest market data.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CryptoDetail;
