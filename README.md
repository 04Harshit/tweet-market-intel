# Real-time Market Intelligence System

A Python-based system for collecting, processing, and analyzing Twitter/X data for Indian stock market intelligence.

## Features
- Scrapes tweets with Indian stock market hashtags (#nifty50, #sensex, #intraday, #banknifty)
- Handles rate limiting and anti-bot measures
- Processes Unicode and Indian language content
- Converts text to quantitative trading signals using TF-IDF
- Stores data efficiently in Parquet format
- Provides low-memory visualization
- Supports concurrent processing

## Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd twitter-market-intel
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install snscrape (for Twitter scraping without API):
```bash
pip install git+https://github.com/JustAnotherArchivist/snscrape.git
```

### Usage

1. Configure settings (optional):
```bash
cp config/settings.yaml.example config/settings.yaml
```

2. Run the complete pipeline:
```bash
python scripts/run_pipeline.py
```

3. Or run individual components:
```bash
# Collect tweets
python src/collector/twitter_scraper.py --hashtags "#nifty50,#sensex" --hours 24 --limit 2000

# Process collected data
python src/processor/data_cleaner.py --input data/raw/ --output data/processed/

# Generate signals
python src/analyzer/signal_generator.py --input data/processed/ --output data/signals/

# Visualize results
python src/visualization/stream_visualizer.py --input data/signals/
```

## Project Structure
```
github-repo/
├── src/                    # Source code
├── config/                 # Configuration files
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests
├── scripts/                # Execution scripts
├── data/                   # Sample output data
└── docs/                   # Technical documentation
```

## Configuration

Edit `config/settings.yaml` to customize:
- Hashtags to monitor
- Collection time window
- Rate limiting parameters
- Storage paths
- Analysis parameters

## Sample Output

The system generates:
- Raw tweet data in JSON/Parquet format
- Cleaned and processed data
- Quantitative trading signals
- Visualizations showing signal trends

## License

MIT