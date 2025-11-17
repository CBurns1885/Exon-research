/**
 * Main App Component
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import CountryDetail from './pages/CountryDetail';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/country/:countryCode" element={<CountryDetail />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
