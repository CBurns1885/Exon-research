/**
 * Utility functions for formatting data
 */

export const formatPrice = (price) => {
  if (price === null || price === undefined) return 'N/A';

  if (price < 0.01) {
    return `$${price.toFixed(6)}`;
  } else if (price < 1) {
    return `$${price.toFixed(4)}`;
  } else if (price < 100) {
    return `$${price.toFixed(2)}`;
  } else {
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
};

export const formatNumber = (num, decimals = 0) => {
  if (num === null || num === undefined) return 'N/A';
  return num.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
};

export const formatPercent = (percent, showSign = true) => {
  if (percent === null || percent === undefined) return 'N/A';
  const sign = showSign && percent > 0 ? '+' : '';
  return `${sign}${percent.toFixed(2)}%`;
};

export const formatMarketCap = (marketCap) => {
  if (marketCap === null || marketCap === undefined) return 'N/A';

  if (marketCap >= 1e12) {
    return `$${(marketCap / 1e12).toFixed(2)}T`;
  } else if (marketCap >= 1e9) {
    return `$${(marketCap / 1e9).toFixed(2)}B`;
  } else if (marketCap >= 1e6) {
    return `$${(marketCap / 1e6).toFixed(2)}M`;
  } else {
    return `$${formatNumber(marketCap, 0)}`;
  }
};

export const formatVolume = (volume) => {
  return formatMarketCap(volume);
};

export const formatDate = (dateString) => {
  if (!dateString) return 'N/A';
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
};

export const formatDateTime = (dateString) => {
  if (!dateString) return 'N/A';
  const date = new Date(dateString);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

export const getPriceChangeClass = (change) => {
  if (change === null || change === undefined) return 'text-gray-400';
  return change >= 0 ? 'text-crypto-green' : 'text-crypto-red';
};

export const getPriceChangeIcon = (change) => {
  if (change === null || change === undefined) return '→';
  return change >= 0 ? '↑' : '↓';
};

export const getConfidenceColor = (confidence) => {
  if (confidence >= 0.8) return 'text-green-400';
  if (confidence >= 0.6) return 'text-yellow-400';
  return 'text-red-400';
};
