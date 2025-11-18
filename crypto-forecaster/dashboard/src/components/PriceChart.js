/**
 * PriceChart component - displays historical price chart with forecasts
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
import { formatPrice, formatDate } from '../utils/formatters';

const PriceChart = ({ historicalData, forecastData, cryptoName, cryptoColor }) => {
  // Combine historical and forecast data
  const chartData = [];

  // Add historical data
  historicalData.forEach((item) => {
    chartData.push({
      date: item.date,
      price: item.close,
      type: 'historical',
    });
  });

  // Add forecast data
  forecastData.forEach((item) => {
    chartData.push({
      date: item.date,
      forecast: item.predicted_price,
      lower: item.lower_bound,
      upper: item.upper_bound,
      type: 'forecast',
    });
  });

  // Sort by date
  chartData.sort((a, b) => new Date(a.date) - new Date(b.date));

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-dark-card border border-dark-border rounded-lg p-3 shadow-xl">
          <p className="text-sm font-semibold text-gray-300 mb-2">{formatDate(label)}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              <span className="font-semibold">{entry.name}:</span> {formatPrice(entry.value)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="chart-container">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">
          {cryptoName} Price Chart
        </h3>
        <div className="flex items-center space-x-4 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-0.5 bg-blue-400"></div>
            <span className="text-gray-400">Historical</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-0.5 bg-purple-400" style={{ borderTop: '2px dashed' }}></div>
            <span className="text-gray-400">Forecast</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-purple-400 opacity-30"></div>
            <span className="text-gray-400">Confidence</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={chartData}>
          <defs>
            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={cryptoColor || "#3b82f6"} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={cryptoColor || "#3b82f6"} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="date"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickFormatter={(value) => {
              const date = new Date(value);
              return `${date.getMonth() + 1}/${date.getDate()}`;
            }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickFormatter={(value) => formatPrice(value)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ paddingTop: '20px' }} />

          {/* Confidence interval */}
          <Area
            type="monotone"
            dataKey="upper"
            fill="#a78bfa"
            stroke="none"
            fillOpacity={0.2}
            name="Upper Bound"
          />
          <Area
            type="monotone"
            dataKey="lower"
            fill="#a78bfa"
            stroke="none"
            fillOpacity={0.2}
            name="Lower Bound"
          />

          {/* Historical price */}
          <Line
            type="monotone"
            dataKey="price"
            stroke={cryptoColor || "#3b82f6"}
            strokeWidth={2}
            dot={false}
            name="Price"
            connectNulls
          />

          {/* Forecast */}
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#a78bfa"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name="Forecast"
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;
