# Central Bank Rate Forecaster

Advanced interest rate forecasting system using LLM-powered NLP analysis of central bank communications combined with economic indicators.

## Features

- **Multi-Bank Coverage**: Fed, ECB, BoE, BoJ, BoC, RBA, SNB, RBNZ
- **Speech Analysis**: Scrapes and analyzes central bank speeches and statements
- **LLM-Powered NLP**: Uses OpenAI/Claude or local LLMs for sentiment analysis
- **Economic Indicators**: Real-time inflation, employment, GDP data
- **Rate Forecasting**: ML models predict rate changes up to 12 months ahead
- **Sentiment Tracking**: Hawkish/dovish sentiment scoring
- **Meeting Predictions**: Forecasts for upcoming monetary policy meetings

## Central Banks Tracked

1. 🇺🇸 **Federal Reserve (Fed)** - United States
2. 🇪🇺 **European Central Bank (ECB)** - Eurozone
3. 🇬🇧 **Bank of England (BoE)** - United Kingdom
4. 🇯🇵 **Bank of Japan (BoJ)** - Japan
5. 🇨🇦 **Bank of Canada (BoC)** - Canada
6. 🇦🇺 **Reserve Bank of Australia (RBA)** - Australia
7. 🇨🇭 **Swiss National Bank (SNB)** - Switzerland
8. 🇳🇿 **Reserve Bank of New Zealand (RBNZ)** - New Zealand

## Data Sources

### Central Bank Communications
- Official websites (speeches, statements, minutes)
- Press conferences (transcripts)
- Monetary policy reports
- Economic projections
- Meeting calendars

### Economic Indicators
- Inflation rates (CPI, PCE, Core inflation)
- Employment data (unemployment rate, payrolls)
- GDP growth rates
- Consumer confidence
- Manufacturing PMI

### News & Analysis
- Reuters, Bloomberg feeds
- Central bank Twitter/social media
- Economic news aggregation

## NLP Analysis Features

### Sentiment Analysis
- **Hawkish**: Favors rate hikes, inflation control
- **Dovish**: Favors rate cuts, growth support
- **Neutral**: Balanced stance
- **Confidence Score**: 0-100% certainty

### Key Phrase Extraction
- Policy keywords detection
- Forward guidance parsing
- Data dependency identification
- Inflation/growth focus analysis

### Topic Modeling
- Main concerns identification
- Priority themes tracking
- Narrative change detection

## Technology Stack

- **Python**: Core backend
- **OpenAI API**: GPT-4 for advanced NLP (or Claude, Llama)
- **LangChain**: LLM orchestration
- **BeautifulSoup/Scrapy**: Web scraping
- **Transformers**: Local NLP models (BERT, FinBERT)
- **Prophet/ARIMA**: Time series forecasting
- **FastAPI**: REST API
- **SQLite**: Database
- **ChromaDB**: Vector database for semantic search

## Quick Start

```bash
# Install dependencies
pip install -r central-bank-requirements.txt

# Set up API keys
cp central-bank-forecaster/.env.example central-bank-forecaster/.env
# Edit .env and add your OpenAI/Anthropic API key

# Run the forecaster
python run_central_banks.py
```

## API Endpoints

- `GET /api/central-banks` - List all tracked central banks
- `GET /api/central-banks/{bank}/current-rate` - Current policy rate
- `GET /api/central-banks/{bank}/forecast` - Rate forecasts
- `GET /api/central-banks/{bank}/speeches` - Recent speeches
- `GET /api/central-banks/{bank}/sentiment` - Sentiment analysis
- `GET /api/central-banks/{bank}/indicators` - Economic indicators
- `POST /api/central-banks/scrape-speeches` - Trigger speech scraping
- `POST /api/central-banks/analyze-sentiment` - Run NLP analysis

## Forecasting Approach

1. **Collect Communications**: Scrape speeches, statements, minutes
2. **NLP Analysis**: Extract sentiment and policy signals using LLMs
3. **Economic Data**: Gather inflation, employment, GDP data
4. **Feature Engineering**: Combine text sentiment with economic indicators
5. **Model Training**: Train ML models on historical rate decisions
6. **Prediction**: Generate 3, 6, 12-month rate forecasts
7. **Confidence Intervals**: Provide probability distributions

## Example Output

```json
{
  "central_bank": "FED",
  "current_rate": 5.25,
  "forecasts": [
    {
      "date": "2025-12-15",
      "predicted_rate": 4.75,
      "probability": 0.65,
      "sentiment": "dovish",
      "key_factors": ["inflation cooling", "labor market softening"]
    }
  ],
  "latest_speech": {
    "speaker": "Jerome Powell",
    "date": "2025-11-10",
    "sentiment": "slightly_dovish",
    "confidence": 0.82,
    "key_phrases": ["data dependent", "inflation progress", "patient approach"]
  }
}
```

## License

MIT
