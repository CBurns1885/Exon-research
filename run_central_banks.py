#!/usr/bin/env python3
"""
One-click execution script for Central Bank Rate Forecaster.

This script:
1. Initializes the database
2. Scrapes central bank speeches
3. Analyzes sentiment using LLM
4. Generates rate forecasts
5. Starts the API server
"""
import sys
import os
from pathlib import Path

# Add central-bank-forecaster to path
sys.path.insert(0, str(Path(__file__).parent / "central-bank-forecaster"))

from backend.data.database import init_db, SessionLocal
from backend.scrapers.speech_scraper import SpeechScraper
from backend.nlp.sentiment_analyzer import SentimentAnalyzer
from backend.models.rate_forecaster import RateForecaster
from backend.config import config
import pandas as pd


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔═════════════════════════════════════════════════════════════════╗
    ║     Central Bank Interest Rate Forecaster                       ║
    ║     LLM-Powered NLP Analysis + Economic Forecasting             ║
    ║                                                                 ║
    ║  🇺🇸 Fed | 🇪🇺 ECB | 🇬🇧 BoE | 🇯🇵 BoJ | 🇨🇦 BoC              ║
    ║  🇦🇺 RBA | 🇨🇭 SNB | 🇳🇿 RBNZ                                  ║
    ╚═════════════════════════════════════════════════════════════════╝
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

        # Step 2: Scrape speeches
        step(2, 5, "Scraping central bank speeches")
        db = SessionLocal()
        try:
            scraper = SpeechScraper()

            print("  Scraping speeches from central banks...")
            speeches = scraper.scrape_all_banks(max_per_bank=10)

            # Save to database
            count = scraper.save_speeches_to_db(speeches, db)
            print(f"✓ Scraped and saved {count} speeches")

            # Show breakdown
            for bank_code, bank_speeches in speeches.items():
                if bank_speeches:
                    print(f"  - {bank_code}: {len(bank_speeches)} speeches")

        finally:
            db.close()

        # Step 3: Analyze sentiment
        step(3, 5, "Analyzing sentiment with LLM")
        db = SessionLocal()
        try:
            from backend.data.database import CentralBankSpeech

            # Get unanalyzed speeches
            speeches = db.query(CentralBankSpeech).filter(
                CentralBankSpeech.analyzed == False,
                CentralBankSpeech.text_content != None
            ).limit(10).all()

            if speeches:
                print(f"  Analyzing {len(speeches)} speeches with {config.LLM_PROVIDER}...")
                print(f"  Model: {config.LLM_MODEL}")

                analyzer = SentimentAnalyzer()

                for i, speech in enumerate(speeches, 1):
                    print(f"  [{i}/{len(speeches)}] {speech.title[:50]}...")

                    result = analyzer.analyze_speech(
                        text=speech.text_content or "",
                        title=speech.title,
                        speaker=speech.speaker or ""
                    )

                    # Update speech
                    speech.sentiment = result['sentiment']
                    speech.sentiment_score = result['sentiment_score']
                    speech.confidence = result['confidence']
                    speech.key_phrases = result['key_phrases']
                    speech.main_topics = result['main_topics']
                    speech.policy_signals = result['policy_signals']
                    speech.analyzed = True
                    speech.llm_model = f"{config.LLM_PROVIDER}:{config.LLM_MODEL}"

                db.commit()
                print(f"\n✓ Analyzed {len(speeches)} speeches")
            else:
                print("  No new speeches to analyze")

        finally:
            db.close()

        # Step 4: Generate forecasts
        step(4, 5, "Generating interest rate forecasts")
        db = SessionLocal()
        try:
            from backend.data.database import CentralBankRate, CentralBankSpeech, EconomicIndicator

            forecaster = RateForecaster()
            total_forecasts = 0

            # Note: For demo, we'll need some sample rate data
            # In production, this would be fetched from APIs or manually entered

            print("  Note: Historical rate data should be loaded from data sources")
            print("  For demo purposes, forecasting capability is ready")
            print("  Use API endpoints to add rate data and generate forecasts")

            print("\n✓ Forecasting system ready")

        finally:
            db.close()

        # Step 5: Start API server
        step(5, 5, "Starting API server")
        print(f"✓ API server starting at http://{config.API_HOST}:{config.API_PORT}")
        print(f"\n{'─' * 70}")
        print("API Endpoints:")
        print(f"  - API Docs: http://localhost:{config.API_PORT}/docs")
        print(f"  - Central banks: http://localhost:{config.API_PORT}/api/central-banks")
        print(f"  - Speeches: http://localhost:{config.API_PORT}/api/central-banks/FED/speeches")
        print(f"  - Sentiment: http://localhost:{config.API_PORT}/api/central-banks/FED/sentiment")
        print(f"  - Forecasts: http://localhost:{config.API_PORT}/api/central-banks/FED/forecast")
        print(f"\n{'─' * 70}")
        print(f"\n🚀 Setup complete! Starting server...\n")
        print(f"LLM Provider: {config.LLM_PROVIDER}")
        print(f"Model: {config.LLM_MODEL}")
        print(f"\n")

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
