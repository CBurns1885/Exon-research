# Federal Reserve Interest Rate Prediction via Twitter Sentiment Analysis

A research project that uses LLM-based sentiment analysis of Twitter/X posts to predict Federal Reserve interest rate decisions.

## Overview

This project scrapes Twitter/X for economy-related keywords, analyzes sentiment using Large Language Models, and correlates the findings with historical Federal Reserve interest rate decisions to evaluate the predictive power of social media sentiment.

## Features

- **Twitter/X Scraper**: Collects tweets containing economy-related keywords
- **LLM Sentiment Analysis**: Uses advanced language models to analyze economic sentiment
- **Fed Decision Tracker**: Tracks historical and upcoming Federal Reserve decisions
- **Correlation Analysis**: Analyzes relationship between Twitter sentiment and Fed decisions
- **Visualization**: Charts and graphs showing sentiment trends vs. Fed actions

## Project Structure

```
.
├── src/
│   ├── scraper/          # Twitter/X scraping module
│   ├── sentiment/        # LLM-based sentiment analysis
│   ├── fed_tracker/      # Federal Reserve decision tracking
│   ├── analysis/         # Correlation and prediction analysis
│   └── utils/            # Utility functions
├── data/                 # Data storage (gitignored)
├── config/               # Configuration files
└── notebooks/            # Jupyter notebooks for analysis
```

## Quick Start

### 1. Setup

Run the automated setup script:

```bash
bash setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Create necessary directories
- Generate a `.env` file from the template

### 2. Configure API Keys

Edit `.env` and add your API keys:

```bash
# Twitter API (get from https://developer.twitter.com)
TWITTER_BEARER_TOKEN=your_token_here

# LLM API (choose one)
ANTHROPIC_API_KEY=your_key_here  # For Claude
# OR
OPENAI_API_KEY=your_key_here     # For GPT

# Optional: Federal Reserve Economic Data
FRED_API_KEY=your_key_here       # From https://fred.stlouisfed.org
```

### 3. Run the System

```bash
# Activate virtual environment
source venv/bin/activate

# Scrape Twitter and analyze sentiment
python -m src.main scrape

# Predict next Fed meeting decision
python -m src.main predict

# Analyze historical correlation
python -m src.main analyze

# Run full workflow
python -m src.main full
```

## How It Works

### 1. Data Collection
The system searches Twitter/X for posts containing economy-related keywords:
- **Federal Reserve**: "Federal Reserve", "FOMC", "Jerome Powell"
- **Interest Rates**: "rate hike", "rate cut", "Fed funds rate"
- **Economic Indicators**: "inflation", "CPI", "unemployment", "GDP"
- **Monetary Policy**: "hawkish", "dovish", "quantitative easing"

### 2. Sentiment Analysis
Each tweet is analyzed by a Large Language Model (Claude or GPT) which:
- Assigns a sentiment score from -1 (very bearish/dovish) to +1 (very bullish/hawkish)
- Categorizes sentiment as: very_bearish, bearish, neutral, bullish, very_bullish
- Provides confidence scores and reasoning
- Identifies key factors influencing the assessment

### 3. Federal Reserve Tracking
The system monitors:
- Current Federal Funds rate
- Upcoming FOMC meeting dates
- Historical Fed decisions (rate changes)
- Key economic indicators (CPI, unemployment, GDP)

### 4. Prediction & Correlation
- Analyzes sentiment trends leading up to Fed meetings
- Correlates pre-meeting sentiment with actual decisions
- Generates predictions for upcoming meetings
- Calculates statistical significance of correlations

### 5. Visualization
Creates charts showing:
- Sentiment trends over time
- Sentiment distribution by category
- Correlation between sentiment and Fed actions
- Prediction confidence and breakdown

## Research Questions

This project helps answer:

1. **Can Twitter sentiment predict Federal Reserve decisions?**
   - Statistical correlation analysis between pre-meeting sentiment and actual decisions

2. **How far in advance does sentiment shift?**
   - Tracking when market expectations change relative to Fed meetings

3. **What's the accuracy of crowd-sourced predictions?**
   - Comparing Twitter-based predictions to actual outcomes

4. **Which economic topics generate the most discussion?**
   - Identifying which keywords and events drive Twitter engagement

## Example Output

### Prediction Report

```
FEDERAL RESERVE DECISION PREDICTION
============================================================
Meeting Date: 2025-03-19
Days Until Meeting: 45

Predicted Action: Rate Hold
Confidence: 72.5%

Sentiment Metrics:
  Average Sentiment: 0.12
  Recent Sentiment: 0.15
  Sentiment Trend: 0.03
  Bullish: 45.2%
  Bearish: 32.1%

Reasoning: Neutral sentiment (0.12) suggests maintaining current
policy; sentiment trend confirms prediction; divided sentiment
reduces confidence

Total Tweets Analyzed: 847
============================================================
```

## Documentation

- **USAGE.md**: Detailed usage guide with examples
- **config/**: Configuration files for keywords and system settings
- **notebooks/**: Jupyter notebooks for interactive analysis

## Keywords Tracked

The system monitors a comprehensive set of economy-related terms:

- **Federal Reserve**: Federal Reserve, Fed, FOMC, Jerome Powell, Janet Yellen
- **Interest Rates**: rate hike, rate cut, basis points, Fed funds rate
- **Economic Indicators**: inflation, CPI, unemployment, jobs report, GDP, recession
- **Monetary Policy**: hawkish, dovish, quantitative easing, taper

See `config/keywords.yaml` for the complete list and customization options.

## Requirements

- Python 3.8+
- Twitter API access (Free tier works, but has limits)
- Anthropic API key (for Claude) OR OpenAI API key (for GPT-4)
- Optional: FRED API key for economic data

## Data Storage

All data is stored locally in SQLite:
- `data/fed_sentiment.db`: Tweets, sentiment analysis, predictions
- `data/plots/`: Generated visualization charts
- `logs/`: Application logs

## Contributing

Contributions welcome! Areas for improvement:
- Additional data sources (Reddit, financial news, etc.)
- Machine learning models for prediction
- Real-time monitoring and alerts
- Enhanced visualizations and dashboards
- Backtesting framework

## Disclaimer

This is a research project for educational purposes. It should NOT be used as the sole basis for financial or investment decisions. Federal Reserve policy is influenced by numerous complex factors beyond social media sentiment.

## License

See LICENSE file for details.
