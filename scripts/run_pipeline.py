#!/usr/bin/env python3
"""
Complete pipeline runner for market intelligence system.
Runs data collection, processing, analysis, and visualization.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
from collector.twitter_scraper import TwitterScraper
from processor.data_cleaner import DataCleaner
from analyzer.signal_generator import SignalGenerator
from storage.parquet_handler import ParquetHandler
from visualization.stream_visualizer import StreamVisualizer


def setup_logging(config_path: str = 'config/logging_config.yaml'):
    """Setup logging from configuration"""
    import logging.config
    
    try:
        with open(config_path, 'r') as f:
            log_config = yaml.safe_load(f)
        logging.config.dictConfig(log_config)
    except Exception as e:
        # Basic logging if config fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.warning(f"Could not load logging config: {e}")
    
    return logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the complete data pipeline"""
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.logger = setup_logging()
        
        # Initialize components
        self.scraper = TwitterScraper(self.config)
        self.cleaner = DataCleaner(self.config)
        self.generator = SignalGenerator(self.config)
        self.storage = ParquetHandler(self.config)
        self.visualizer = StreamVisualizer(self.config)
        
        # Setup paths
        self.base_data_path = Path(self.config['storage']['paths']['raw']).parent
        self.ensure_directories()
        
        self.logger.info("Pipeline initialized")
    
    def ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.base_data_path / 'raw',
            self.base_data_path / 'processed',
            self.base_data_path / 'signals',
            self.base_data_path / 'models',
            self.base_data_path / 'visualizations',
            self.base_data_path / 'analysis',
            Path('logs')
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")
    
    def run_collection(self, **kwargs):
        """Run data collection phase"""
        self.logger.info("=== Starting Data Collection ===")
        
        try:
            # Collect tweets
            df_tweets = self.scraper.scrape_tweets(**kwargs)
            
            if df_tweets.empty:
                self.logger.error("No tweets collected")
                return None
            
            # Save raw data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            raw_path = self.base_data_path / 'raw' / f'tweets_{timestamp}.parquet'
            self.scraper.save_tweets(df_tweets, str(raw_path))
            
            self.logger.info(f"✅ Collection complete: {len(df_tweets)} tweets")
            return df_tweets
            
        except Exception as e:
            self.logger.error(f"Collection failed: {e}")
            return None
    
    def run_processing(self, df_tweets):
        """Run data processing phase"""
        self.logger.info("=== Starting Data Processing ===")
        
        try:
            # Clean data
            df_clean = self.cleaner.clean_dataframe(df_tweets)
            
            if df_clean.empty:
                self.logger.error("No data after cleaning")
                return None
            
            # Save cleaned data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_path = self.base_data_path / 'processed' / f'cleaned_{timestamp}.parquet'
            df_clean.to_parquet(
                processed_path,
                engine='pyarrow',
                compression='snappy'
            )
            
            self.logger.info(f"✅ Processing complete: {len(df_clean)} cleaned tweets")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return None
    
    def run_analysis(self, df_clean):
        """Run analysis phase"""
        self.logger.info("=== Starting Analysis ===")
        
        try:
            # Generate signals
            df_signals = self.generator.generate_signals(df_clean)
            
            if df_signals.empty:
                self.logger.error("No signals generated")
                return None
            
            # Save signals
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            signals_path = self.base_data_path / 'signals' / f'signals_{timestamp}.parquet'
            df_signals.to_parquet(
                signals_path,
                engine='pyarrow',
                compression='snappy'
            )
            
            # Save model
            model_path = self.base_data_path / 'models' / f'model_{timestamp}.joblib'
            self.generator.save_model(str(model_path))
            
            self.logger.info(f"✅ Analysis complete: {len(df_signals)} signal windows")
            return df_signals
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return None
    
    def run_visualization(self, df_signals):
        """Run visualization phase"""
        self.logger.info("=== Starting Visualization ===")
        
        try:
            # Create dashboard
            fig = self.visualizer.create_dashboard(df_signals)
            
            # Save visualization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            viz_path = self.base_data_path / 'visualizations' / f'dashboard_{timestamp}.html'
            self.visualizer.save_visualization(fig, str(viz_path))
            
            # Create streaming plot
            fig_stream = self.visualizer.create_streaming_plot(df_signals)
            stream_path = self.base_data_path / 'visualizations' / f'streaming_{timestamp}.html'
            self.visualizer.save_visualization(fig_stream, str(stream_path))
            
            # Create heatmap
            fig_heatmap = self.visualizer.create_heatmap(df_signals)
            heatmap_path = self.base_data_path / 'visualizations' / f'heatmap_{timestamp}.html'
            self.visualizer.save_visualization(fig_heatmap, str(heatmap_path))
            
            self.logger.info(f"✅ Visualization complete")
            return [viz_path, stream_path, heatmap_path]
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return None
    
    def run_pipeline(self, **scraping_kwargs):
        """Run complete pipeline"""
        start_time = datetime.now()
        self.logger.info(f"🚀 Starting pipeline at {start_time}")
        
        results = {}
        
        # 1. Collection
        df_tweets = self.run_collection(**scraping_kwargs)
        if df_tweets is None:
            self.logger.error("Pipeline stopped: Collection failed")
            return results
        results['collection'] = {'tweet_count': len(df_tweets)}
        
        # 2. Processing
        df_clean = self.run_processing(df_tweets)
        if df_clean is None:
            self.logger.error("Pipeline stopped: Processing failed")
            return results
        results['processing'] = {'cleaned_count': len(df_clean)}
        
        # 3. Analysis
        df_signals = self.run_analysis(df_clean)
        if df_signals is None:
            self.logger.error("Pipeline stopped: Analysis failed")
            return results
        results['analysis'] = {'signal_windows': len(df_signals)}
        
        # 4. Visualization
        viz_paths = self.run_visualization(df_signals)
        if viz_paths:
            results['visualization'] = {'output_files': [str(p) for p in viz_paths]}
        
        # Calculate pipeline statistics
        end_time = datetime.now()
        duration = end_time - start_time
        
        results['pipeline'] = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'success': True
        }
        
        self.logger.info(f"✅ Pipeline completed in {duration}")
        self.logger.info(f"📊 Results: {results}")
        
        return results
    
    def run_incremental(self, hours: int = 1):
        """Run incremental pipeline for recent data"""
        self.logger.info(f"Running incremental pipeline for last {hours} hours")
        
        scraping_kwargs = {
            'hours': hours,
            'min_tweets': 200  # Lower threshold for incremental
        }
        
        return self.run_pipeline(**scraping_kwargs)
    
    def run_test(self):
        """Run test pipeline with sample data"""
        self.logger.info("Running test pipeline")
        
        # Create sample data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        sample_tweets = pd.DataFrame({
            'tweet_id': [f'test_{i}' for i in range(100)],
            'username': [f'user_{i}' for i in range(100)],
            'content': [f'Sample tweet about market {i}' for i in range(100)],
            'timestamp': dates,
            'like_count': np.random.randint(0, 100, 100),
            'retweet_count': np.random.randint(0, 50, 100),
            'reply_count': np.random.randint(0, 20, 100),
            'hashtags': [['#test', '#sample'] for _ in range(100)]
        })
        
        # Run pipeline with sample data
        df_clean = self.run_processing(sample_tweets)
        if df_clean is not None:
            df_signals = self.run_analysis(df_clean)
            if df_signals is not None:
                self.run_visualization(df_signals)
        
        self.logger.info("✅ Test pipeline completed")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Market Intelligence Pipeline')
    parser.add_argument('--mode', choices=['full', 'incremental', 'test', 'collect', 'process', 'analyze', 'visualize'],
                       default='full', help='Pipeline mode')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back (for collection)')
    parser.add_argument('--limit', type=int, default=2000, help='Minimum tweets to collect')
    parser.add_argument('--hashtags', type=str, help='Comma-separated hashtags')
    parser.add_argument('--input', type=str, help='Input file for specific stages')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PipelineRunner(args.config)
    
    # Parse hashtags
    hashtags = None
    if args.hashtags:
        hashtags = [h.strip() for h in args.hashtags.split(',')]
    
    # Run based on mode
    if args.mode == 'full':
        scraping_kwargs = {
            'hashtags': hashtags,
            'hours': args.hours,
            'min_tweets': args.limit
        }
        pipeline.run_pipeline(**scraping_kwargs)
    
    elif args.mode == 'incremental':
        pipeline.run_incremental(hours=args.hours)
    
    elif args.mode == 'test':
        pipeline.run_test()
    
    elif args.mode == 'collect':
        df_tweets = pipeline.run_collection(
            hashtags=hashtags,
            hours=args.hours,
            min_tweets=args.limit
        )
        if df_tweets is not None:
            print(f"✅ Collected {len(df_tweets)} tweets")
    
    elif args.mode == 'process' and args.input:
        df_tweets = pd.read_parquet(args.input)
        df_clean = pipeline.run_processing(df_tweets)
        if df_clean is not None:
            print(f"✅ Processed {len(df_clean)} tweets")
    
    elif args.mode == 'analyze' and args.input:
        df_clean = pd.read_parquet(args.input)
        df_signals = pipeline.run_analysis(df_clean)
        if df_signals is not None:
            print(f"✅ Generated {len(df_signals)} signal windows")
    
    elif args.mode == 'visualize' and args.input:
        df_signals = pd.read_parquet(args.input)
        viz_paths = pipeline.run_visualization(df_signals)
        if viz_paths:
            print(f"✅ Created visualizations: {viz_paths}")
    
    else:
        print("❌ Invalid arguments. Use --help for usage information")


if __name__ == "__main__":
    main()