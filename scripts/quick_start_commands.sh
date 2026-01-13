# Clone and setup
git clone <repository-url>
cd twitter-market-intel

# Make setup script executable and run it
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/JustAnotherArchivist/snscrape.git

# Copy config examples
cp config/settings.yaml.example config/settings.yaml
cp config/logging_config.yaml.example config/logging_config.yaml

# Create sample data
python scripts/create_sample_data.py

# Run tests
python -m pytest tests/ -v

# Run the complete pipeline
python scripts/run_pipeline.py

# Run individual stages
python scripts/run_pipeline.py --mode collect --hours 24 --limit 2000
python scripts/run_pipeline.py --mode process
python scripts/run_pipeline.py --mode analyze
python scripts/run_pipeline.py --mode visualize

# Test with sample data
python scripts/run_pipeline.py --mode test