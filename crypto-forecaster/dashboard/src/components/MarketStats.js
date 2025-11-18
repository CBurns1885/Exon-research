/**
 * MarketStats component - displays market statistics
 */
import React from 'react';
import { formatPrice, formatPercent, formatMarketCap, formatDateTime, getPriceChangeClass } from '../utils/formatters';

const MarketStats = ({ metadata, latestPrice }) => {
  if (!metadata) {
    return (
      <div className="chart-container">
        <p className="text-gray-400 text-center">No market data available</p>
      </div>
    );
  }

  const stats = [
    {
      label: 'Current Price',
      value: formatPrice(metadata.current_price || latestPrice?.close),
      color: 'text-blue-400',
    },
    {
      label: 'Market Cap',
      value: formatMarketCap(metadata.market_cap),
      color: 'text-purple-400',
    },
    {
      label: 'Market Cap Rank',
      value: metadata.market_cap_rank ? `#${metadata.market_cap_rank}` : 'N/A',
      color: 'text-green-400',
    },
    {
      label: '24h Volume',
      value: formatMarketCap(metadata.total_volume),
      color: 'text-orange-400',
    },
    {
      label: '24h Change',
      value: formatPercent(metadata.price_change_percentage_24h, true),
      color: getPriceChangeClass(metadata.price_change_percentage_24h),
    },
    {
      label: '7d Change',
      value: formatPercent(metadata.price_change_percentage_7d, true),
      color: getPriceChangeClass(metadata.price_change_percentage_7d),
    },
    {
      label: '30d Change',
      value: formatPercent(metadata.price_change_percentage_30d, true),
      color: getPriceChangeClass(metadata.price_change_percentage_30d),
    },
    {
      label: 'All-Time High',
      value: formatPrice(metadata.ath),
      color: 'text-green-400',
      subtitle: metadata.ath_date ? formatDateTime(metadata.ath_date) : null,
    },
    {
      label: 'All-Time Low',
      value: formatPrice(metadata.atl),
      color: 'text-red-400',
      subtitle: metadata.atl_date ? formatDateTime(metadata.atl_date) : null,
    },
  ];

  return (
    <div className="chart-container">
      <h3 className="text-lg font-semibold text-white mb-4">Market Statistics</h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {stats.map((stat, index) => (
          <div key={index} className="metric-card">
            <p className="text-xs text-gray-400 mb-2">{stat.label}</p>
            <p className={`text-xl font-bold ${stat.color}`}>
              {stat.value}
            </p>
            {stat.subtitle && (
              <p className="text-xs text-gray-500 mt-1">{stat.subtitle}</p>
            )}
          </div>
        ))}
      </div>

      {metadata.last_updated && (
        <div className="mt-4 text-xs text-gray-500 text-center">
          Last updated: {formatDateTime(metadata.last_updated)}
        </div>
      )}
    </div>
  );
};

export default MarketStats;
