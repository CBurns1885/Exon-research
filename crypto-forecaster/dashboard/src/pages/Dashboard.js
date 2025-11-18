/**
 * Dashboard page - Main overview of all cryptocurrencies
 */
import React, { useState, useEffect } from 'react';
import { FiRefreshCw, FiTrendingUp } from 'react-icons/fi';
import CryptoCard from '../components/CryptoCard';
import LoadingSpinner from '../components/LoadingSpinner';
import cryptoApi from '../services/api';

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadDashboardData();
    // Auto-refresh every 5 minutes
    const interval = setInterval(loadDashboardData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const data = await cryptoApi.getDashboard();
      setDashboardData(data);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data. Please ensure the backend is running at http://localhost:8001');
      console.error('Error loading dashboard:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      await cryptoApi.updatePrices();
      await loadDashboardData();
    } catch (err) {
      console.error('Error refreshing prices:', err);
    } finally {
      setRefreshing(false);
    }
  };

  if (loading) {
    return <LoadingSpinner message="Loading cryptocurrency data..." />;
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center max-w-md mx-4">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-white mb-2">Error Loading Data</h2>
          <p className="text-gray-400 mb-6">{error}</p>
          <button
            onClick={loadDashboardData}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const cryptoCount = dashboardData ? Object.keys(dashboardData).length : 0;

  // Calculate total market cap
  const totalMarketCap = dashboardData
    ? Object.values(dashboardData).reduce((sum, data) => {
        return sum + (data.metadata?.market_cap || 0);
      }, 0)
    : 0;

  return (
    <div className="min-h-screen">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900 via-purple-900 to-pink-900">
        <div className="container mx-auto px-4 py-12">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-5xl font-bold mb-2 text-white">
                💎 Crypto Forecaster
              </h1>
              <p className="text-blue-200 text-lg">
                Real-time cryptocurrency prices & ML-powered 30-day forecasts
              </p>
            </div>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className={`flex items-center space-x-2 px-6 py-3 bg-white/10 backdrop-blur-sm text-white rounded-lg hover:bg-white/20 transition-all border border-white/20 ${
                refreshing ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              <FiRefreshCw className={refreshing ? 'animate-spin' : ''} />
              <span>{refreshing ? 'Refreshing...' : 'Refresh Prices'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="bg-dark-card border-b border-dark-border">
        <div className="container mx-auto px-4 py-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <p className="text-gray-400 text-sm mb-1">Cryptocurrencies</p>
              <p className="text-3xl font-bold text-white">{cryptoCount}</p>
            </div>
            <div className="text-center">
              <p className="text-gray-400 text-sm mb-1">Total Market Cap</p>
              <p className="text-3xl font-bold text-green-400">
                ${(totalMarketCap / 1e12).toFixed(2)}T
              </p>
            </div>
            <div className="text-center">
              <p className="text-gray-400 text-sm mb-1">Forecast Period</p>
              <p className="text-3xl font-bold text-blue-400">30 Days</p>
            </div>
            <div className="text-center">
              <p className="text-gray-400 text-sm mb-1">ML Model</p>
              <p className="text-3xl font-bold text-purple-400">Prophet</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Crypto Cards Grid */}
        <div className="mb-8">
          <div className="flex items-center space-x-2 mb-6">
            <FiTrendingUp className="text-2xl text-blue-400" />
            <h2 className="text-3xl font-bold text-white">Top 10 Cryptocurrencies</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {dashboardData &&
              Object.entries(dashboardData)
                .sort((a, b) => {
                  const rankA = a[1].metadata?.market_cap_rank || 999;
                  const rankB = b[1].metadata?.market_cap_rank || 999;
                  return rankA - rankB;
                })
                .map(([symbol, data]) => (
                  <CryptoCard
                    key={symbol}
                    symbol={symbol}
                    info={data.info}
                    latestPrice={data.latest_price}
                    metadata={data.metadata}
                    latestForecast={data.latest_forecast}
                  />
                ))}
          </div>
        </div>

        {/* Info Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          <div className="bg-dark-card rounded-lg p-6 border border-dark-border">
            <h3 className="text-lg font-semibold text-white mb-3">📊 Real-Time Data</h3>
            <p className="text-sm text-gray-400">
              Live prices from CoinGecko API updated every 5 minutes. Historical data covers 1 year of trading activity.
            </p>
          </div>

          <div className="bg-dark-card rounded-lg p-6 border border-dark-border">
            <h3 className="text-lg font-semibold text-white mb-3">🤖 ML Forecasting</h3>
            <p className="text-sm text-gray-400">
              Advanced Prophet model optimized for crypto volatility with 95% confidence intervals on all predictions.
            </p>
          </div>

          <div className="bg-dark-card rounded-lg p-6 border border-dark-border">
            <h3 className="text-lg font-semibold text-white mb-3">📈 Technical Analysis</h3>
            <p className="text-sm text-gray-400">
              12+ technical indicators including RSI, MACD, Bollinger Bands, and moving averages for each cryptocurrency.
            </p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-dark-card border-t border-dark-border mt-12">
        <div className="container mx-auto px-4 py-6 text-center">
          <p className="text-gray-500 text-sm">
            Crypto Forecaster | Powered by CoinGecko API & Prophet ML | Data updates every 5 minutes
          </p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
