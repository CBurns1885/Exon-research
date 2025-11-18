"""Federal Reserve decision tracker and data collector"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

from ..utils import load_config, setup_logger
from ..utils.config_loader import get_api_key


class FedTracker:
    """Track Federal Reserve decisions and interest rate changes"""

    # Historical FOMC meeting dates for 2024-2025 (approximate)
    FOMC_MEETING_DATES = [
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"
    ]

    def __init__(self):
        """Initialize Fed tracker"""
        self.config = load_config()
        self.logger = setup_logger("fed_tracker")
        self.fred_api_key = None

        # Try to get FRED API key (optional)
        try:
            self.fred_api_key = get_api_key('fred')
            self.logger.info("FRED API key loaded")
        except ValueError:
            self.logger.warning("FRED API key not found. Some features may be limited.")

    def get_current_fed_rate(self) -> Optional[float]:
        """
        Get the current Federal Funds rate from FRED API

        Returns:
            Current fed funds rate or None if unavailable
        """
        if not self.fred_api_key:
            self.logger.warning("FRED API key not available")
            return None

        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'FEDFUNDS',
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 1
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if data.get('observations'):
                rate = float(data['observations'][0]['value'])
                self.logger.info(f"Current Fed Funds Rate: {rate}%")
                return rate

        except Exception as e:
            self.logger.error(f"Error fetching Fed rate from FRED: {e}")

        return None

    def get_economic_indicators(self) -> Dict[str, Any]:
        """
        Fetch key economic indicators from FRED API

        Returns:
            Dictionary of economic indicators
        """
        if not self.fred_api_key:
            self.logger.warning("FRED API key not available")
            return {}

        indicators = {}
        series_map = {
            'fed_funds_rate': 'FEDFUNDS',
            'cpi': 'CPIAUCSL',
            'unemployment': 'UNRATE',
            'gdp': 'GDP',
            'pce_inflation': 'PCEPI'
        }

        for indicator_name, series_id in series_map.items():
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': 1
                }

                response = requests.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                if data.get('observations'):
                    obs = data['observations'][0]
                    indicators[indicator_name] = {
                        'value': float(obs['value']),
                        'date': obs['date']
                    }

            except Exception as e:
                self.logger.error(f"Error fetching {indicator_name}: {e}")

        return indicators

    def get_upcoming_meetings(self, days_ahead: int = 90) -> List[Dict[str, Any]]:
        """
        Get upcoming FOMC meetings

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of upcoming meeting dates
        """
        today = datetime.now().date()
        cutoff_date = today + timedelta(days=days_ahead)

        upcoming = []
        for date_str in self.FOMC_MEETING_DATES:
            meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            if today <= meeting_date <= cutoff_date:
                upcoming.append({
                    'date': date_str,
                    'days_until': (meeting_date - today).days
                })

        upcoming.sort(key=lambda x: x['date'])
        self.logger.info(f"Found {len(upcoming)} upcoming FOMC meetings in next {days_ahead} days")

        return upcoming

    def get_next_meeting(self) -> Optional[Dict[str, Any]]:
        """
        Get the next FOMC meeting

        Returns:
            Dictionary with next meeting info or None
        """
        upcoming = self.get_upcoming_meetings(days_ahead=365)
        return upcoming[0] if upcoming else None

    def get_historical_decisions(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get historical Fed decisions (mock data for now)

        In a production system, this would fetch from a real API or database

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of historical decisions
        """
        # This is mock data - in production, you'd fetch from a real source
        # or maintain a database of historical decisions
        historical_decisions = [
            {
                'decision_date': '2024-01-31',
                'decision_type': 'hold',
                'rate_change': 0.0,
                'new_rate': 5.50,
                'statement_summary': 'Committee decided to maintain the target range at 5.25-5.50%'
            },
            {
                'decision_date': '2024-03-20',
                'decision_type': 'hold',
                'rate_change': 0.0,
                'new_rate': 5.50,
                'statement_summary': 'Maintained current policy stance while monitoring inflation'
            },
            {
                'decision_date': '2024-05-01',
                'decision_type': 'hold',
                'rate_change': 0.0,
                'new_rate': 5.50,
                'statement_summary': 'Held rates steady, awaiting more data on inflation trajectory'
            },
            {
                'decision_date': '2024-06-12',
                'decision_type': 'hold',
                'rate_change': 0.0,
                'new_rate': 5.50,
                'statement_summary': 'Rates unchanged, updated economic projections'
            },
            {
                'decision_date': '2024-07-31',
                'decision_type': 'hold',
                'rate_change': 0.0,
                'new_rate': 5.50,
                'statement_summary': 'Maintained policy rate while monitoring labor market'
            },
            {
                'decision_date': '2024-09-18',
                'decision_type': 'cut',
                'rate_change': -0.50,
                'new_rate': 5.00,
                'statement_summary': 'First rate cut in over four years, 50 basis points'
            },
            {
                'decision_date': '2024-11-07',
                'decision_type': 'cut',
                'rate_change': -0.25,
                'new_rate': 4.75,
                'statement_summary': 'Continued easing cycle with 25 basis point cut'
            }
        ]

        # Filter by date range
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        filtered = [
            d for d in historical_decisions
            if start <= datetime.strptime(d['decision_date'], "%Y-%m-%d").date() <= end
        ]

        self.logger.info(f"Retrieved {len(filtered)} historical decisions")
        return filtered

    def classify_decision(self, rate_change: float) -> str:
        """
        Classify a Fed decision based on rate change

        Args:
            rate_change: Change in basis points

        Returns:
            Decision classification
        """
        if rate_change <= -0.5:
            return 'significant_cut'
        elif rate_change < 0:
            return 'cut'
        elif rate_change == 0:
            return 'hold'
        elif rate_change <= 0.25:
            return 'hike'
        else:
            return 'significant_hike'

    def get_fed_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of Fed status

        Returns:
            Dictionary with current Fed status and upcoming events
        """
        summary = {
            'current_rate': self.get_current_fed_rate(),
            'next_meeting': self.get_next_meeting(),
            'economic_indicators': self.get_economic_indicators(),
            'timestamp': datetime.utcnow().isoformat()
        }

        return summary
