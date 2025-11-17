/**
 * API service for communicating with the backend
 */
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Get all countries
  getCountries: async () => {
    const response = await api.get('/api/countries');
    return response.data;
  },

  // Get dashboard data
  getDashboardData: async () => {
    const response = await api.get('/api/dashboard-data');
    return response.data;
  },

  // Get country data
  getCountryData: async (countryCode, indicator = null, startDate = null, endDate = null) => {
    const params = {};
    if (indicator) params.indicator = indicator;
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;

    const response = await api.get(`/api/data/${countryCode}`, { params });
    return response.data;
  },

  // Get country forecasts
  getCountryForecasts: async (countryCode, indicator = null) => {
    const params = {};
    if (indicator) params.indicator = indicator;

    const response = await api.get(`/api/forecasts/${countryCode}`, { params });
    return response.data;
  },

  // Get country summary
  getCountrySummary: async (countryCode) => {
    const response = await api.get(`/api/summary/${countryCode}`);
    return response.data;
  },

  // Get all data
  getAllData: async (indicator = null) => {
    const params = {};
    if (indicator) params.indicator = indicator;

    const response = await api.get('/api/data', { params });
    return response.data;
  },

  // Get all forecasts
  getAllForecasts: async (indicator = null) => {
    const params = {};
    if (indicator) params.indicator = indicator;

    const response = await api.get('/api/forecasts', { params });
    return response.data;
  },

  // Generate data
  generateData: async () => {
    const response = await api.post('/api/generate-data');
    return response.data;
  },

  // Generate forecasts
  generateForecasts: async (modelType = 'prophet') => {
    const response = await api.post('/api/generate-forecasts', { model_type: modelType });
    return response.data;
  },
};

export default apiService;
