/**
 * Country Detail Page - Detailed view of a single country
 */
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import ForecastChart from '../components/ForecastChart';
import MetricCard from '../components/MetricCard';
import apiService from '../services/api';

const CountryDetail = () => {
  const { countryCode } = useParams();
  const navigate = useNavigate();

  const [summary, setSummary] = useState(null);
  const [historicalData, setHistoricalData] = useState({});
  const [forecastData, setForecastData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const indicators = [
    'interest_rate',
    'fx_rate',
    'gdp_growth',
    'capital_inflow',
    'capital_outflow',
    'imports',
    'exports',
  ];

  useEffect(() => {
    loadCountryData();
  }, [countryCode]);

  const loadCountryData = async () => {
    try {
      setLoading(true);

      // Load summary
      const summaryData = await apiService.getCountrySummary(countryCode.toUpperCase());
      setSummary(summaryData);

      // Load historical data for each indicator
      const historical = {};
      const forecast = {};

      for (const indicator of indicators) {
        try {
          const histData = await apiService.getCountryData(
            countryCode.toUpperCase(),
            indicator
          );
          historical[indicator] = histData;

          const fcstData = await apiService.getCountryForecasts(
            countryCode.toUpperCase(),
            indicator
          );
          forecast[indicator] = fcstData;
        } catch (err) {
          console.error(`Error loading ${indicator}:`, err);
        }
      }

      setHistoricalData(historical);
      setForecastData(forecast);
      setError(null);
    } catch (err) {
      setError('Failed to load country data');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading country data...</p>
        </div>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Error Loading Data</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const formatIndicator = (ind) => {
    return ind
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const getUnit = (indicator) => {
    switch (indicator) {
      case 'interest_rate':
      case 'gdp_growth':
        return '%';
      case 'fx_rate':
        return summary.country.currency + '/USD';
      case 'capital_inflow':
      case 'capital_outflow':
      case 'imports':
      case 'exports':
        return 'M USD';
      default:
        return '';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white">
        <div className="container mx-auto px-4 py-8">
          <button
            onClick={() => navigate('/')}
            className="mb-4 text-blue-100 hover:text-white flex items-center"
          >
            ← Back to Dashboard
          </button>

          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2">
                {summary.country.emoji} {summary.country.name}
              </h1>
              <p className="text-blue-100 text-lg">
                {summary.country.region} | Currency: {summary.country.currency}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Key Metrics Summary */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Latest Indicators</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {indicators.map((indicator) => {
              const latest = summary.latest_data[indicator];
              const forecast = summary.latest_forecasts[indicator];

              return (
                <MetricCard
                  key={indicator}
                  title={formatIndicator(indicator)}
                  value={latest?.value}
                  unit={getUnit(indicator)}
                  forecast={forecast?.forecast_value}
                />
              );
            })}
          </div>
        </div>

        {/* Trade Balance Card */}
        <div className="mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Trade Balance</h3>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-gray-600">Exports</p>
                <p className="text-2xl font-bold text-green-600">
                  {summary.latest_data.exports?.value?.toFixed(0)} M
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Imports</p>
                <p className="text-2xl font-bold text-red-600">
                  {summary.latest_data.imports?.value?.toFixed(0)} M
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Balance</p>
                <p
                  className={`text-2xl font-bold ${
                    (summary.latest_data.exports?.value || 0) -
                      (summary.latest_data.imports?.value || 0) >
                    0
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}
                >
                  {(
                    (summary.latest_data.exports?.value || 0) -
                    (summary.latest_data.imports?.value || 0)
                  ).toFixed(0)}{' '}
                  M
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Forecast Charts */}
        <div>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">6-Month Forecasts</h2>
          <div className="space-y-6">
            {indicators.map((indicator) => {
              const historical = historicalData[indicator] || [];
              const forecast = forecastData[indicator] || [];

              if (historical.length === 0) return null;

              return (
                <ForecastChart
                  key={indicator}
                  historicalData={historical}
                  forecastData={forecast}
                  indicator={indicator}
                  countryName={summary.country.name}
                />
              );
            })}
          </div>
        </div>

        {/* Model Information */}
        <div className="mt-8 bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">About the Forecasts</h3>
          <p className="text-gray-700">
            These forecasts are generated using Facebook's Prophet model, which accounts for
            seasonality, trends, and historical patterns. The shaded areas represent 95%
            confidence intervals for the predictions. All forecasts are based on historical
            data from the past 5 years.
          </p>
        </div>
      </div>
    </div>
  );
};

export default CountryDetail;
