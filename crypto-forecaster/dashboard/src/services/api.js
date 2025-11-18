/**
 * API service for cryptocurrency forecasting backend
 */
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const cryptoApi = {
  // Get all cryptocurrencies
  getCoins: async () => {
    const response = await api.get('/api/crypto/coins');
    return response.data;
  },

  // Get dashboard data for all cryptocurrencies
  getDashboard: async () => {
    const response = await api.get('/api/crypto/dashboard');
    return response.data;
  },

  // Get historical prices for a cryptocurrency
  getPrices: async (symbol, days = null) => {
    const params = days ? { days } : {};
    const response = await api.get(`/api/crypto/prices/${symbol}`, { params });
    return response.data;
  },

  // Get forecast for a cryptocurrency
  getForecast: async (symbol) => {
    const response = await api.get(`/api/crypto/forecast/${symbol}`);
    return response.data;
  },

  // Get metadata for a cryptocurrency
  getMetadata: async (symbol) => {
    const response = await api.get(`/api/crypto/metadata/${symbol}`);
    return response.data;
  },

  // Update current prices
  updatePrices: async () => {
    const response = await api.post('/api/crypto/update-prices');
    return response.data;
  },

  // Fetch historical data
  fetchHistorical: async () => {
    const response = await api.post('/api/crypto/fetch-historical');
    return response.data;
  },

  // Generate new forecasts
  generateForecasts: async (modelType = 'prophet') => {
    const response = await api.post('/api/crypto/generate-forecasts', {
      model_type: modelType
    });
    return response.data;
  },
};

export default cryptoApi;
