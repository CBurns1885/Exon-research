/**
 * CryptoCard component - displays cryptocurrency overview
 */
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { formatPrice, formatPercent, formatMarketCap, getPriceChangeClass, getPriceChangeIcon } from '../utils/formatters';

const CryptoCard = ({ symbol, info, latestPrice, metadata, latestForecast }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(`/crypto/${symbol}`);
  };

  const price = latestPrice?.close || metadata?.current_price;
  const change24h = metadata?.price_change_percentage_24h;
  const change7d = metadata?.price_change_percentage_7d;
  const marketCap = metadata?.market_cap;
  const forecastPrice = latestForecast?.predicted_price;

  // Calculate forecast change
  const forecastChange = price && forecastPrice
    ? ((forecastPrice - price) / price) * 100
    : null;

  return (
    <div className="crypto-card animate-slide-up" onClick={handleClick}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div
            className="text-4xl w-14 h-14 flex items-center justify-center rounded-full"
            style={{ backgroundColor: `${info?.color}20` }}
          >
            {info?.emoji}
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">{info?.name}</h3>
            <p className="text-sm text-gray-400">{symbol}</p>
          </div>
        </div>
        {metadata?.market_cap_rank && (
          <div className="badge badge-blue">
            #{metadata.market_cap_rank}
          </div>
        )}
      </div>

      {/* Price */}
      <div className="mb-4">
        <div className="text-3xl font-bold text-white mb-1">
          {formatPrice(price)}
        </div>
        {change24h !== null && change24h !== undefined && (
          <div className={`flex items-center space-x-2 ${getPriceChangeClass(change24h)}`}>
            <span className="text-lg">{getPriceChangeIcon(change24h)}</span>
            <span className="text-sm font-semibold">
              {formatPercent(change24h)} (24h)
            </span>
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="metric-card">
          <p className="text-xs text-gray-400 mb-1">Market Cap</p>
          <p className="text-sm font-semibold text-white">
            {formatMarketCap(marketCap)}
          </p>
        </div>
        <div className="metric-card">
          <p className="text-xs text-gray-400 mb-1">7d Change</p>
          <p className={`text-sm font-semibold ${getPriceChangeClass(change7d)}`}>
            {formatPercent(change7d, true)}
          </p>
        </div>
      </div>

      {/* Forecast Preview */}
      {forecastPrice && (
        <div className="border-t border-dark-border pt-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-gray-400 mb-1">30d Forecast</p>
              <p className="text-lg font-bold text-blue-400">
                {formatPrice(forecastPrice)}
              </p>
            </div>
            {forecastChange !== null && (
              <div className={`badge ${forecastChange >= 0 ? 'badge-green' : 'badge-red'}`}>
                {getPriceChangeIcon(forecastChange)} {formatPercent(Math.abs(forecastChange), false)}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Click indicator */}
      <div className="mt-4 text-center">
        <p className="text-xs text-gray-500">Click for detailed analysis →</p>
      </div>
    </div>
  );
};

export default CryptoCard;
