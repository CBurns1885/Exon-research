/**
 * Forecast Chart Component - displays historical data and forecasts
 */
import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from 'recharts';

const ForecastChart = ({ historicalData, forecastData, indicator, countryName }) => {
  // Combine historical and forecast data
  const combinedData = [];

  // Add historical data
  historicalData.forEach((item) => {
    combinedData.push({
      date: item.date,
      actual: item.value,
      type: 'historical',
    });
  });

  // Add forecast data
  forecastData.forEach((item) => {
    combinedData.push({
      date: item.date,
      forecast: item.forecast_value,
      lower: item.lower_bound,
      upper: item.upper_bound,
      type: 'forecast',
    });
  });

  // Sort by date
  combinedData.sort((a, b) => new Date(a.date) - new Date(b.date));

  // Format indicator name
  const formatIndicator = (ind) => {
    return ind
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="text-sm font-semibold mb-1">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value?.toFixed(2)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        {formatIndicator(indicator)} - {countryName}
      </h3>

      <ResponsiveContainer width="100%" height={350}>
        <ComposedChart data={combinedData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {/* Confidence interval area */}
          <Area
            type="monotone"
            dataKey="upper"
            fill="#93c5fd"
            stroke="none"
            fillOpacity={0.3}
            name="Upper Bound"
          />
          <Area
            type="monotone"
            dataKey="lower"
            fill="#93c5fd"
            stroke="none"
            fillOpacity={0.3}
            name="Lower Bound"
          />

          {/* Historical data */}
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#2563eb"
            strokeWidth={2}
            dot={{ r: 3 }}
            name="Actual"
            connectNulls
          />

          {/* Forecast data */}
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#dc2626"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={{ r: 3 }}
            name="Forecast"
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="mt-4 flex items-center justify-center space-x-6 text-sm text-gray-600">
        <div className="flex items-center">
          <div className="w-4 h-0.5 bg-blue-600 mr-2"></div>
          <span>Historical</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-0.5 bg-red-600 mr-2" style={{ borderTop: '2px dashed' }}></div>
          <span>Forecast</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-3 bg-blue-200 mr-2 opacity-30"></div>
          <span>Confidence Interval</span>
        </div>
      </div>
    </div>
  );
};

export default ForecastChart;
