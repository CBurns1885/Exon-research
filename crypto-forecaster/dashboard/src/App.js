/**
 * Main App Component
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import CryptoDetail from './pages/CryptoDetail';

function App() {
  return (
    <Router>
      <div className="App min-h-screen bg-dark-bg">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/crypto/:symbol" element={<CryptoDetail />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
