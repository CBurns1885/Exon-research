#!/usr/bin/env python3
"""
One-click execution script for African Economy Forecasting Model.

This script:
1. Initializes the database
2. Generates historical economic data
3. Generates forecasts for all countries
4. Starts the API server
"""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.data.database import init_db, SessionLocal
from backend.data.generator import EconomicDataGenerator
from backend.models.forecaster import EconomicForecaster
from backend.config import config
import pandas as pd


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║  African Frontier Markets Economic Forecasting Model      ║
    ║  🇰🇪 Kenya | 🇳🇬 Nigeria | 🇬🇭 Ghana | 🇷🇼 Rwanda | 🇪🇬 Egypt  ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def step(number: int, total: int, description: str):
    """Print step information."""
    print(f"\n[{number}/{total}] {description}")
    print("─" * 60)


def main():
    """Run the complete forecasting pipeline."""
    print_banner()

    try:
        # Step 1: Initialize database
        step(1, 4, "Initializing database")
        init_db()
        print("✓ Database initialized")

        # Step 2: Generate historical data
        step(2, 4, "Generating historical economic data")
        db = SessionLocal()
        try:
            generator = EconomicDataGenerator(years=config.HISTORICAL_YEARS)
            count = generator.save_to_database(db)
            print(f"✓ Generated {count} historical data points")
            print(f"  - Countries: {len(config.COUNTRIES)}")
            print(f"  - Indicators: {len(config.INDICATORS)}")
            print(f"  - Time period: {config.HISTORICAL_YEARS} years")
        finally:
            db.close()

        # Step 3: Generate forecasts
        step(3, 4, "Generating 6-month forecasts")
        db = SessionLocal()
        try:
            # Load data
            from backend.data.database import EconomicData
            data_query = db.query(EconomicData).all()
            data_list = [d.to_dict() for d in data_query]
            df = pd.DataFrame(data_list)

            # Generate forecasts using Prophet (best model)
            forecaster = EconomicForecaster(forecast_months=config.FORECAST_MONTHS)
            forecasts_df = forecaster.forecast_all(df, model_type="prophet")

            # Save forecasts
            forecast_count = forecaster.save_forecasts(forecasts_df, db)
            print(f"\n✓ Generated {forecast_count} forecasts")
            print(f"  - Forecast period: {config.FORECAST_MONTHS} months")
            print(f"  - Model: Prophet (with seasonality)")
        finally:
            db.close()

        # Step 4: Start API server
        step(4, 4, "Starting API server")
        print(f"✓ API server starting at http://{config.API_HOST}:{config.API_PORT}")
        print(f"\n{'─' * 60}")
        print("API Endpoints:")
        print(f"  - Dashboard data: http://localhost:{config.API_PORT}/api/dashboard-data")
        print(f"  - Countries: http://localhost:{config.API_PORT}/api/countries")
        print(f"  - All data: http://localhost:{config.API_PORT}/api/data")
        print(f"  - All forecasts: http://localhost:{config.API_PORT}/api/forecasts")
        print(f"\n{'─' * 60}")
        print("\n🚀 Setup complete! Starting server...\n")

        # Start server
        import uvicorn
        from backend.api.main import app

        uvicorn.run(
            app,
            host=config.API_HOST,
            port=config.API_PORT,
            log_level="info"
        )

    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
