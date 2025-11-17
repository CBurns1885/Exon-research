/**
 * Metric Card Component - displays a single economic metric
 */
import React from 'react';

const MetricCard = ({ title, value, unit, change, forecast, emoji }) => {
  const formatValue = (val) => {
    if (val === null || val === undefined) return 'N/A';
    if (typeof val === 'number') {
      return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
    }
    return val;
  };

  const getChangeColor = (change) => {
    if (!change) return 'text-gray-500';
    return change > 0 ? 'text-green-600' : 'text-red-600';
  };

  return (
    <div className="metric-card">
      <div className="flex justify-between items-start mb-2">
        <h3 className="text-sm font-medium text-gray-600">{title}</h3>
        {emoji && <span className="text-2xl">{emoji}</span>}
      </div>

      <div className="mt-2">
        <p className="text-2xl font-bold text-gray-900">
          {formatValue(value)} {unit}
        </p>

        {change !== undefined && change !== null && (
          <p className={`text-sm mt-1 ${getChangeColor(change)}`}>
            {change > 0 ? '↑' : '↓'} {Math.abs(change).toFixed(2)}%
          </p>
        )}

        {forecast !== undefined && forecast !== null && (
          <p className="text-sm text-blue-600 mt-1">
            Forecast: {formatValue(forecast)} {unit}
          </p>
        )}
      </div>
    </div>
  );
};

export default MetricCard;
