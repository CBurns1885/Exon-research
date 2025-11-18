#!/bin/bash
# Setup script for Fed Sentiment Analysis project

echo "Setting up Federal Reserve Sentiment Analysis Project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs data/plots notebooks

# Copy .env.example to .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your API keys"
else
    echo ".env file already exists"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the scraper: python -m src.main scrape"
echo "4. Make predictions: python -m src.main predict"
echo ""
