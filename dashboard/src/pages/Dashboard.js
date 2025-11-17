/**
 * Dashboard Page - Main overview of all countries
 */
import React, { useState, useEffect } from 'react';
import CountryCard from '../components/CountryCard';
import apiService from '../services/api';

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const data = await apiService.getDashboardData();
      setDashboardData(data);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data. Please ensure the backend is running.');
      console.error('Error loading dashboard:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center max-w-md">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Error Loading Data</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={loadDashboardData}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white">
        <div className="container mx-auto px-4 py-8">
          <h1 className="text-4xl font-bold mb-2">
            African Frontier Markets
          </h1>
          <p className="text-blue-100 text-lg">
            Economic Forecasting Dashboard - 6 Month Outlook
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="text-3xl mb-2">🌍</div>
            <h3 className="text-2xl font-bold text-gray-800">
              {dashboardData ? Object.keys(dashboardData).length : 0}
            </h3>
            <p className="text-gray-600">Countries</p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="text-3xl mb-2">📊</div>
            <h3 className="text-2xl font-bold text-gray-800">7</h3>
            <p className="text-gray-600">Economic Indicators</p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="text-3xl mb-2">📈</div>
            <h3 className="text-2xl font-bold text-gray-800">6</h3>
            <p className="text-gray-600">Month Forecasts</p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="text-3xl mb-2">🔮</div>
            <h3 className="text-2xl font-bold text-gray-800">Prophet</h3>
            <p className="text-gray-600">Forecast Model</p>
          </div>
        </div>

        {/* Countries Grid */}
        <div>
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Country Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {dashboardData &&
              Object.entries(dashboardData).map(([code, data]) => (
                <CountryCard
                  key={code}
                  country={{ code, ...data.country }}
                  latestValues={data.latest_values}
                  latestForecasts={data.latest_forecasts}
                />
              ))}
          </div>
        </div>

        {/* Key Indicators Section */}
        <div className="mt-12">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Key Economic Indicators</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">💰 Interest Rates</h3>
              <p className="text-sm text-gray-600">
                Central bank policy rates reflecting monetary policy stance
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">💱 FX Rates</h3>
              <p className="text-sm text-gray-600">
                Exchange rates against USD showing currency strength
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">📊 GDP Growth</h3>
              <p className="text-sm text-gray-600">
                Quarterly economic growth rates
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">💸 Capital Flows</h3>
              <p className="text-sm text-gray-600">
                Foreign investment inflows and outflows
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">📦 Imports</h3>
              <p className="text-sm text-gray-600">
                Monthly import values in USD millions
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">📤 Exports</h3>
              <p className="text-sm text-gray-600">
                Monthly export values in USD millions
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-gray-800 text-white mt-12">
        <div className="container mx-auto px-4 py-6 text-center">
          <p className="text-gray-400">
            African Frontier Markets Economic Forecasting Model | Data updated in real-time
          </p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
