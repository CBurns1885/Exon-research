#!/usr/bin/env python3
"""
One-click execution script for Cryptocurrency Price Forecasting System.

This script:
1. Initializes the database
2. Fetches current cryptocurrency prices
3. Fetches historical price data
4. Generates price forecasts
5. Starts the API server
"""
import sys
import os
from pathlib import Path

# Add crypto-forecaster to path
sys.path.insert(0, str(Path(__file__).parent / "crypto-forecaster"))

from backend.data.database import init_db, SessionLocal
from backend.data.fetchers import CryptoDataFetcher
from backend.models.forecaster import CryptoPriceForecaster
from backend.utils.indicators import TechnicalIndicators
from backend.config import config
import pandas as pd


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║    Cryptocurrency Price Forecasting System                   ║
    ║    🪙 BTC | 💎 ETH | ⚡ BNB | ☀️ SOL | 💧 XRP                ║
    ║    🔷 ADA | 🐕 DOGE | 🟣 MATIC | ⚫ DOT | 🔗 LINK           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def step(number: int, total: int, description: str):
    """Print step information."""
    print(f"\n[{number}/{total}] {description}")
    print("─" * 70)


def main():
    """Run the complete forecasting pipeline."""
    print_banner()

    try:
        # Step 1: Initialize database
        step(1, 5, "Initializing database")
        init_db()
        print("✓ Database initialized")

        # Step 2: Fetch current prices
        step(2, 5, "Fetching current cryptocurrency prices")
        db = SessionLocal()
        try:
            from backend.data.database import CryptoMetadata

            fetcher = CryptoDataFetcher()
            current_prices = fetcher.get_current_prices_all()

            # Save metadata
            for symbol, data in current_prices.items():
                metadata = db.query(CryptoMetadata).filter(
                    CryptoMetadata.symbol == symbol
                ).first()

                if metadata:
                    metadata.current_price = data.get("current_price")
                    metadata.market_cap = data.get("market_cap")
                    metadata.market_cap_rank = data.get("market_cap_rank")
                    metadata.total_volume = data.get("total_volume")
                    metadata.price_change_percentage_24h = data.get("price_change_percentage_24h")
                    metadata.price_change_percentage_7d = data.get("price_change_percentage_7d")
                else:
                    metadata = CryptoMetadata(
                        symbol=symbol,
                        name=data.get("name", config.get_crypto_name(symbol)),
                        current_price=data.get("current_price"),
                        market_cap=data.get("market_cap"),
                        market_cap_rank=data.get("market_cap_rank"),
                        total_volume=data.get("total_volume"),
                        price_change_percentage_24h=data.get("price_change_percentage_24h"),
                        price_change_percentage_7d=data.get("price_change_percentage_7d")
                    )
                    db.add(metadata)

            db.commit()
            print(f"✓ Fetched prices for {len(current_prices)} cryptocurrencies")

        finally:
            db.close()

        # Step 3: Fetch historical data
        step(3, 5, "Fetching historical price data (this may take a few minutes)")
        db = SessionLocal()
        try:
            from backend.data.database import CryptoPrice, TechnicalIndicator

            historical_data = fetcher.get_historical_data_all(days=config.HISTORICAL_DAYS)

            total_records = 0
            for symbol, df in historical_data.items():
                print(f"  Processing {symbol}...")

                # Save price data
                for _, row in df.iterrows():
                    existing = db.query(CryptoPrice).filter(
                        CryptoPrice.symbol == symbol,
                        CryptoPrice.date == row['date']
                    ).first()

                    if not existing:
                        price_record = CryptoPrice(
                            symbol=symbol,
                            timestamp=row['timestamp'],
                            date=row['date'],
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=float(row['volume']),
                            market_cap=float(row.get('market_cap', 0)) if pd.notna(row.get('market_cap')) else None
                        )
                        db.add(price_record)
                        total_records += 1

                # Calculate technical indicators
                if not df.empty:
                    df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)

                    for _, row in df_with_indicators.iterrows():
                        existing_ind = db.query(TechnicalIndicator).filter(
                            TechnicalIndicator.symbol == symbol,
                            TechnicalIndicator.date == row['date']
                        ).first()

                        if not existing_ind and pd.notna(row.get('rsi')):
                            indicator = TechnicalIndicator(
                                symbol=symbol,
                                date=row['date'],
                                sma_7=float(row['sma_7']) if pd.notna(row['sma_7']) else None,
                                sma_30=float(row['sma_30']) if pd.notna(row['sma_30']) else None,
                                ema_12=float(row['ema_12']) if pd.notna(row['ema_12']) else None,
                                ema_26=float(row['ema_26']) if pd.notna(row['ema_26']) else None,
                                rsi=float(row['rsi']) if pd.notna(row['rsi']) else None,
                                macd=float(row['macd']) if pd.notna(row['macd']) else None,
                                bollinger_high=float(row['bollinger_high']) if pd.notna(row['bollinger_high']) else None,
                                bollinger_low=float(row['bollinger_low']) if pd.notna(row['bollinger_low']) else None,
                                volume_sma=float(row['volume_sma']) if pd.notna(row['volume_sma']) else None
                            )
                            db.add(indicator)

            db.commit()
            print(f"\n✓ Saved {total_records} historical records")

        finally:
            db.close()

        # Step 4: Generate forecasts
        step(4, 5, "Generating 30-day price forecasts")
        db = SessionLocal()
        try:
            from backend.data.database import CryptoPrice

            # Get historical data
            historical_data = {}
            for symbol in config.get_crypto_symbols():
                prices = db.query(CryptoPrice).filter(
                    CryptoPrice.symbol == symbol
                ).order_by(CryptoPrice.date).all()

                if prices:
                    df = pd.DataFrame([p.to_dict() for p in prices])
                    historical_data[symbol] = df

            # Generate forecasts
            forecaster = CryptoPriceForecaster(forecast_days=config.FORECAST_DAYS)
            forecasts = forecaster.forecast_all_cryptos(historical_data, model_type="prophet")

            # Save forecasts
            forecast_count = forecaster.save_forecasts(forecasts, db)

            print(f"\n✓ Generated {forecast_count} forecasts ({config.FORECAST_DAYS} days)")
            print(f"  - Model: Prophet with crypto-optimized parameters")

        finally:
            db.close()

        # Step 5: Start API server
        step(5, 5, "Starting API server")
        print(f"✓ API server starting at http://{config.API_HOST}:{config.API_PORT}")
        print(f"\n{'─' * 70}")
        print("API Endpoints:")
        print(f"  - API Docs: http://localhost:{config.API_PORT}/docs")
        print(f"  - Dashboard data: http://localhost:{config.API_PORT}/api/crypto/dashboard")
        print(f"  - Cryptocurrencies: http://localhost:{config.API_PORT}/api/crypto/coins")
        print(f"\n{'─' * 70}")
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
