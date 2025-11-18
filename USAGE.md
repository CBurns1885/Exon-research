# Usage Guide

## Setup

1. **Clone the repository and navigate to the project directory**

2. **Run the setup script:**
   ```bash
   bash setup.sh
   ```

3. **Configure your API keys:**
   Edit the `.env` file and add your API keys:
   - Twitter API credentials (get from https://developer.twitter.com)
   - Anthropic API key (for Claude) or OpenAI API key (for GPT)
   - FRED API key (optional, for economic data from https://fred.stlouisfed.org/docs/api/api_key.html)

4. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

## Basic Commands

### Scrape Twitter and Analyze Sentiment

Collect tweets about the economy and analyze their sentiment:

```bash
python -m src.main scrape --batch-size 50
```

This will:
- Search Twitter for economy-related keywords
- Save tweets to the database
- Analyze sentiment using LLM
- Display sentiment summary

### Predict Next Fed Meeting

Make a prediction for the next Federal Reserve meeting:

```bash
python -m src.main predict
```

This will:
- Identify the next FOMC meeting date
- Analyze recent Twitter sentiment
- Generate a prediction (cut/hold/hike)
- Display confidence level and reasoning
- Create visualization charts

### Analyze Historical Correlation

Analyze how well Twitter sentiment correlated with past Fed decisions:

```bash
python -m src.main analyze
```

This will:
- Retrieve historical Fed decisions
- Calculate correlation with pre-meeting sentiment
- Display statistical significance
- Provide interpretation

### Run Full Workflow

Run all steps in sequence:

```bash
python -m src.main full
```

This executes: scrape → predict → analyze

## Configuration

### Modify Keywords

Edit `config/keywords.yaml` to customize:
- Search keywords for Twitter
- Priority accounts to track
- Search parameters

### Adjust Settings

Edit `config/config.yaml` to modify:
- Scraping frequency and limits
- LLM provider and model
- Sentiment scoring thresholds
- Analysis parameters

## Using Jupyter Notebooks

An example notebook is provided for interactive analysis:

```bash
jupyter notebook notebooks/example_analysis.ipynb
```

The notebook demonstrates:
- Step-by-step workflow
- Data exploration
- Custom analysis
- Visualization creation

## Database

The system uses SQLite to store:
- **Tweets**: Raw tweet data
- **Sentiment**: LLM analysis results
- **Fed Decisions**: Historical and future meeting data
- **Predictions**: Model predictions with confidence scores

Database location: `data/fed_sentiment.db`

### Query the Database

```python
from src.utils import Database

db = Database()

# Get recent tweets
tweets = db.cursor.execute("SELECT * FROM tweets LIMIT 10").fetchall()

# Get sentiment summary
sentiment = db.cursor.execute("""
    SELECT
        sentiment_category,
        COUNT(*) as count,
        AVG(confidence) as avg_confidence
    FROM sentiment_analysis
    GROUP BY sentiment_category
""").fetchall()

db.close()
```

## Visualization

Charts are automatically saved to `data/plots/`:

- **sentiment_trend_YYYYMMDD.png**: Time series of sentiment
- **sentiment_distribution_YYYYMMDD.png**: Breakdown by category
- **prediction_summary_YYYYMMDD.png**: Prediction confidence and breakdown
- **sentiment_vs_fed_YYYYMMDD.png**: Sentiment overlaid with Fed decisions

## API Rate Limits

Be mindful of API rate limits:

- **Twitter API**: Free tier has strict limits
  - Consider using Twitter API v2 Essential access
  - Add delays between requests (configured in `config.yaml`)

- **LLM APIs**:
  - Anthropic Claude: Rate limits vary by plan
  - OpenAI GPT: Rate limits vary by plan
  - Batch processing helps manage costs

## Tips for Best Results

1. **Collect data regularly**: Run scraping daily or multiple times per day
2. **Build historical data**: More data improves correlation analysis
3. **Monitor sentiment trends**: Look for shifts, not just absolute values
4. **Consider context**: Major economic events can skew sentiment
5. **Validate predictions**: Track accuracy over time to improve model

## Troubleshooting

### No tweets collected
- Verify Twitter API credentials in `.env`
- Check rate limits on Twitter Developer portal
- Ensure keywords are configured in `config/keywords.yaml`

### Sentiment analysis fails
- Verify LLM API key (Anthropic or OpenAI) in `.env`
- Check API quota/credits
- Review error logs in `logs/app.log`

### Database errors
- Ensure `data/` directory exists and is writable
- Check database file permissions
- Try deleting `data/fed_sentiment.db` to recreate

### Poor predictions
- Collect more historical data
- Adjust sentiment thresholds in `config/config.yaml`
- Review and refine keywords for better signal

## Example Workflow

Here's a recommended workflow for using the system:

1. **Initial setup** (one time):
   ```bash
   bash setup.sh
   # Edit .env with API keys
   ```

2. **Daily data collection**:
   ```bash
   source venv/bin/activate
   python -m src.main scrape
   ```

3. **Before Fed meetings** (7-14 days prior):
   ```bash
   python -m src.main predict
   ```

4. **After Fed meetings** (to evaluate accuracy):
   ```bash
   python -m src.main analyze
   ```

5. **Regular analysis** (weekly/monthly):
   - Use Jupyter notebook for deep dives
   - Review visualizations
   - Track prediction accuracy

## Advanced Usage

### Custom Sentiment Analysis

```python
from src.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer(provider='anthropic')  # or 'openai'

text = "Fed likely to cut rates due to cooling inflation"
result = analyzer.analyze_tweet(text)

print(f"Sentiment: {result['sentiment_score']}")
print(f"Category: {result['sentiment_category']}")
print(f"Reasoning: {result['reasoning']}")
```

### Custom Predictions

```python
from src.analysis import FedPredictor
from src.utils import Database

db = Database()
predictor = FedPredictor(db)

# Predict specific meeting
prediction = predictor.predict_next_decision(
    meeting_date='2025-03-19',
    lookback_days=14
)

print(prediction)
```

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
- Check `logs/app.log` for detailed error information
- Review configuration files
- Consult the main README.md
