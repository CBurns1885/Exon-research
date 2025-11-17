/**
 * Country Card Component - displays country overview
 */
import React from 'react';
import { useNavigate } from 'react-router-dom';

const CountryCard = ({ country, latestValues, latestForecasts }) => {
  const navigate = useNavigate();

  const formatValue = (val) => {
    if (val === null || val === undefined) return 'N/A';
    return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
  };

  const handleClick = () => {
    navigate(`/country/${country.code}`);
  };

  return (
    <div className="country-card" onClick={handleClick}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">
            {country.emoji} {country.name}
          </h2>
          <p className="text-sm text-gray-600">{country.region}</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-500">Currency</p>
          <p className="text-lg font-semibold text-gray-700">{country.currency}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mt-4">
        {/* Interest Rate */}
        <div className="p-3 bg-blue-50 rounded-lg">
          <p className="text-xs text-gray-600 mb-1">Interest Rate</p>
          <p className="text-lg font-bold text-blue-700">
            {formatValue(latestValues?.interest_rate)}%
          </p>
        </div>

        {/* FX Rate */}
        <div className="p-3 bg-green-50 rounded-lg">
          <p className="text-xs text-gray-600 mb-1">FX Rate (USD)</p>
          <p className="text-lg font-bold text-green-700">
            {formatValue(latestValues?.fx_rate)}
          </p>
        </div>

        {/* GDP Growth */}
        <div className="p-3 bg-purple-50 rounded-lg">
          <p className="text-xs text-gray-600 mb-1">GDP Growth</p>
          <p className="text-lg font-bold text-purple-700">
            {formatValue(latestValues?.gdp_growth)}%
          </p>
        </div>

        {/* Trade Balance */}
        <div className="p-3 bg-orange-50 rounded-lg">
          <p className="text-xs text-gray-600 mb-1">Trade Balance</p>
          <p className="text-lg font-bold text-orange-700">
            {latestValues?.exports && latestValues?.imports
              ? formatValue(latestValues.exports - latestValues.imports)
              : 'N/A'}M
          </p>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500 text-center">
          Click to view detailed forecasts →
        </p>
      </div>
    </div>
  );
};

export default CountryCard;
