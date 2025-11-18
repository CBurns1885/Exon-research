/**
 * TechnicalIndicators component - displays technical analysis indicators
 */
import React from 'react';
import { formatNumber } from '../utils/formatters';

const TechnicalIndicators = ({ indicators }) => {
  if (!indicators || indicators.length === 0) {
    return (
      <div className="chart-container">
        <p className="text-gray-400 text-center">No technical indicators available</p>
      </div>
    );
  }

  // Get latest indicators
  const latest = indicators[indicators.length - 1];

  const getRSIStatus = (rsi) => {
    if (!rsi) return { text: 'N/A', color: 'text-gray-400' };
    if (rsi >= 70) return { text: 'Overbought', color: 'text-red-400' };
    if (rsi <= 30) return { text: 'Oversold', color: 'text-green-400' };
    return { text: 'Neutral', color: 'text-blue-400' };
  };

  const getMACDSignal = (macd, signal) => {
    if (!macd || !signal) return { text: 'N/A', color: 'text-gray-400' };
    if (macd > signal) return { text: 'Bullish', color: 'text-green-400' };
    return { text: 'Bearish', color: 'text-red-400' };
  };

  const rsiStatus = getRSIStatus(latest.rsi);
  const macdSignal = getMACDSignal(latest.macd, latest.macd_signal);

  return (
    <div className="chart-container">
      <h3 className="text-lg font-semibold text-white mb-4">Technical Indicators</h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {/* RSI */}
        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-gray-400">RSI (14)</p>
            <span className={`badge ${rsiStatus.color.includes('red') ? 'badge-red' : rsiStatus.color.includes('green') ? 'badge-green' : 'badge-blue'}`}>
              {rsiStatus.text}
            </span>
          </div>
          <p className="text-2xl font-bold text-white">
            {latest.rsi ? formatNumber(latest.rsi, 2) : 'N/A'}
          </p>
        </div>

        {/* MACD */}
        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-gray-400">MACD</p>
            <span className={`badge ${macdSignal.color.includes('green') ? 'badge-green' : 'badge-red'}`}>
              {macdSignal.text}
            </span>
          </div>
          <p className="text-lg font-bold text-white">
            {latest.macd ? formatNumber(latest.macd, 2) : 'N/A'}
          </p>
          <p className="text-xs text-gray-400 mt-1">
            Signal: {latest.macd_signal ? formatNumber(latest.macd_signal, 2) : 'N/A'}
          </p>
        </div>

        {/* SMA 7 */}
        <div className="metric-card">
          <p className="text-xs text-gray-400 mb-2">SMA (7-day)</p>
          <p className="text-2xl font-bold text-white">
            ${latest.sma_7 ? formatNumber(latest.sma_7, 2) : 'N/A'}
          </p>
        </div>

        {/* SMA 30 */}
        <div className="metric-card">
          <p className="text-xs text-gray-400 mb-2">SMA (30-day)</p>
          <p className="text-2xl font-bold text-white">
            ${latest.sma_30 ? formatNumber(latest.sma_30, 2) : 'N/A'}
          </p>
        </div>

        {/* EMA 12 */}
        <div className="metric-card">
          <p className="text-xs text-gray-400 mb-2">EMA (12-day)</p>
          <p className="text-2xl font-bold text-white">
            ${latest.ema_12 ? formatNumber(latest.ema_12, 2) : 'N/A'}
          </p>
        </div>

        {/* EMA 26 */}
        <div className="metric-card">
          <p className="text-xs text-gray-400 mb-2">EMA (26-day)</p>
          <p className="text-2xl font-bold text-white">
            ${latest.ema_26 ? formatNumber(latest.ema_26, 2) : 'N/A'}
          </p>
        </div>

        {/* Bollinger High */}
        <div className="metric-card">
          <p className="text-xs text-gray-400 mb-2">Bollinger High</p>
          <p className="text-lg font-bold text-green-400">
            ${latest.bollinger_high ? formatNumber(latest.bollinger_high, 2) : 'N/A'}
          </p>
        </div>

        {/* Bollinger Mid */}
        <div className="metric-card">
          <p className="text-xs text-gray-400 mb-2">Bollinger Mid</p>
          <p className="text-lg font-bold text-blue-400">
            ${latest.bollinger_mid ? formatNumber(latest.bollinger_mid, 2) : 'N/A'}
          </p>
        </div>

        {/* Bollinger Low */}
        <div className="metric-card">
          <p className="text-xs text-gray-400 mb-2">Bollinger Low</p>
          <p className="text-lg font-bold text-red-400">
            ${latest.bollinger_low ? formatNumber(latest.bollinger_low, 2) : 'N/A'}
          </p>
        </div>
      </div>

      {/* Indicator Descriptions */}
      <div className="mt-6 space-y-2 text-sm text-gray-400">
        <p><span className="text-white font-semibold">RSI:</span> Measures momentum. Above 70 = overbought, below 30 = oversold</p>
        <p><span className="text-white font-semibold">MACD:</span> Trend indicator. MACD above signal = bullish, below = bearish</p>
        <p><span className="text-white font-semibold">Bollinger Bands:</span> Price volatility bands. Price near high = resistance, near low = support</p>
      </div>
    </div>
  );
};

export default TechnicalIndicators;
