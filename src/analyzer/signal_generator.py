"""
Generate quantitative trading signals from tweet text.
Uses TF-IDF and custom feature engineering.
"""

import logging
import pickle
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from ..processor.data_cleaner import DataCleaner

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals from text data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.analysis_config = config['analysis']
        
        # Initialize feature extractors
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.feature_pipeline = None
        
        # Market-specific keywords
        self.bullish_keywords = [
            'bullish', 'buy', 'long', 'up', 'rise', 'gain', 'profit', 
            'positive', 'strong', 'growth', 'outperform', 'rally'
        ]
        
        self.bearish_keywords = [
            'bearish', 'sell', 'short', 'down', 'fall', 'drop', 'loss',
            'negative', 'weak', 'decline', 'underperform', 'crash'
        ]
        
        # Technical indicators mentioned
        self.technical_indicators = [
            'support', 'resistance', 'breakout', 'breakdown', 'moving average',
            'rsi', 'macd', 'bollinger', 'volume', 'trend', 'pattern'
        ]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from tweet DataFrame.
        
        Args:
            df: Cleaned tweet DataFrame
            
        Returns:
            DataFrame with generated signals
        """
        if df.empty:
            logger.warning("Empty DataFrame, no signals generated")
            return pd.DataFrame()
        
        logger.info(f"Generating signals for {len(df)} tweets")
        
        # Create copy to avoid modifying original
        df_signals = df.copy()
        
        # Extract text features
        logger.info("Extracting text features...")
        text_features = self.extract_text_features(df_signals['content_clean'].tolist())
        
        # Add features to DataFrame
        for i, (tfidf_features, embeddings) in enumerate(text_features):
            df_signals.loc[i, 'tfidf_vector'] = pickle.dumps(tfidf_features)
            
            # Add aggregated features
            df_signals.loc[i, 'tfidf_sum'] = np.sum(tfidf_features)
            df_signals.loc[i, 'tfidf_mean'] = np.mean(tfidf_features)
            df_signals.loc[i, 'tfidf_std'] = np.std(tfidf_features)
        
        # Generate individual signals
        logger.info("Generating individual signals...")
        df_signals = self._generate_individual_signals(df_signals)
        
        # Aggregate signals over time windows
        logger.info("Aggregating signals...")
        df_aggregated = self.aggregate_signals(df_signals)
        
        # Calculate confidence intervals
        logger.info("Calculating confidence intervals...")
        df_aggregated = self.calculate_confidence_intervals(df_aggregated)
        
        # Create composite signal
        df_aggregated['composite_signal'] = self.create_composite_signal(df_aggregated)
        
        logger.info(f"Generated signals for {len(df_aggregated)} time windows")
        
        return df_aggregated
    
    def extract_text_features(self, texts: List[str]) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Extract features from text using TF-IDF and optional embeddings.
        
        Args:
            texts: List of cleaned tweet texts
            
        Returns:
            List of (tfidf_features, embeddings) tuples
        """
        method = self.analysis_config['feature_extraction']['method']
        
        # Initialize TF-IDF vectorizer if not already done
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.analysis_config['feature_extraction']['tfidf_max_features'],
                ngram_range=tuple(self.analysis_config['feature_extraction']['tfidf_ngram_range']),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            
            # Fit on all texts
            self.tfidf_vectorizer.fit(texts)
        
        # Transform texts
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        # Convert to dense arrays for storage
        features = []
        for i in range(tfidf_matrix.shape[0]):
            tfidf_features = tfidf_matrix[i].toarray().flatten()
            
            # Optional: Add embeddings
            embeddings = None
            if method in ['embeddings', 'both']:
                embeddings = self._get_embeddings(texts[i])
            
            features.append((tfidf_features, embeddings))
        
        return features
    
    def _get_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Get sentence embeddings for text"""
        try:
            # Using sentence-transformers if installed
            from sentence_transformers import SentenceTransformer
            
            if not hasattr(self, 'embedding_model'):
                model_name = self.analysis_config['feature_extraction']['embedding_model']
                self.embedding_model = SentenceTransformer(model_name)
            
            return self.embedding_model.encode([text])[0]
            
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping embeddings")
            return None
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return None
    
    def _generate_individual_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate individual signal components for each tweet"""
        
        # Sentiment-based signals
        df['sentiment_signal'] = df['sentiment_score'].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        
        # Engagement-weighted sentiment
        if 'engagement_score' in df.columns:
            df['weighted_sentiment'] = df['sentiment_signal'] * df['engagement_score']
        else:
            # Create simple engagement proxy
            df['engagement_proxy'] = (
                df['like_count'] + df['retweet_count'] * 2 + df['reply_count'] * 3
            ) / 100
            df['weighted_sentiment'] = df['sentiment_signal'] * df['engagement_proxy']
        
        # Urgency signal (based on exclamation/question marks)
        df['urgency_signal'] = (
            df['exclamation_count'] * 0.5 + 
            df['question_count'] * 0.3 +
            (df['uppercase_ratio'] > 0.3).astype(int) * 0.2
        )
        
        # Market term concentration signal
        if 'market_term_count' in df.columns:
            df['market_focus_signal'] = np.tanh(df['market_term_count'] / 10)
        
        # Technical indicator signal
        df['technical_mention_signal'] = df['content_clean'].apply(
            lambda x: 1 if any(indicator in x.lower() for indicator in self.technical_indicators) else 0
        )
        
        # Hashtag-based signals
        df['has_nifty_hashtag'] = df['hashtags'].apply(
            lambda x: 1 if any('nifty' in str(tag).lower() for tag in x) else 0
        )
        df['has_sensex_hashtag'] = df['hashtags'].apply(
            lambda x: 1 if any('sensex' in str(tag).lower() for tag in x) else 0
        )
        
        # Time-based signal (recent tweets more important)
        if 'timestamp' in df.columns:
            max_time = df['timestamp'].max()
            df['time_decay'] = 1 - (max_time - df['timestamp']).dt.total_seconds() / (24 * 3600)
            df['time_decay'] = df['time_decay'].clip(0, 1)
        
        return df
    
    def aggregate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate individual signals over time windows.
        
        Args:
            df: DataFrame with individual signals
            
        Returns:
            Aggregated signals by time window
        """
        if df.empty or 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        # Set timestamp as index for resampling
        df_time = df.set_index('timestamp').sort_index()
        
        # Define aggregation window (default: 1 hour)
        window_size = self.analysis_config['signal_generation']['window_size']
        window = f'{window_size}min'
        
        # Aggregate signals
        aggregation_rules = {
            'sentiment_signal': 'mean',
            'weighted_sentiment': 'sum',
            'urgency_signal': 'mean',
            'market_focus_signal': 'mean',
            'technical_mention_signal': 'sum',
            'has_nifty_hashtag': 'sum',
            'has_sensex_hashtag': 'sum',
            'like_count': 'sum',
            'retweet_count': 'sum',
            'reply_count': 'sum',
            'tweet_count': 'size'  # Count tweets in window
        }
        
        # Resample and aggregate
        df_aggregated = df_time.resample(window).agg(aggregation_rules)
        
        # Calculate additional aggregated metrics
        df_aggregated['sentiment_strength'] = df_aggregated['sentiment_signal'].abs()
        df_aggregated['engagement_total'] = (
            df_aggregated['like_count'] + 
            df_aggregated['retweet_count'] + 
            df_aggregated['reply_count']
        )
        
        # Normalize by tweet count
        df_aggregated['avg_sentiment'] = df_aggregated['sentiment_signal']
        df_aggregated['sentiment_volume'] = df_aggregated['sentiment_strength'] * df_aggregated['tweet_count']
        
        # Reset index
        df_aggregated = df_aggregated.reset_index()
        
        return df_aggregated
    
    def calculate_confidence_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confidence intervals for signals.
        
        Args:
            df: Aggregated signal DataFrame
            
        Returns:
            DataFrame with confidence intervals
        """
        if df.empty:
            return df
        
        confidence_level = self.analysis_config['signal_generation']['confidence_interval']
        
        # For sentiment signal confidence
        if 'tweet_count' in df.columns and 'sentiment_signal' in df.columns:
            # Simple binomial proportion confidence interval
            import scipy.stats as stats
            
            def calculate_ci(row):
                n = row['tweet_count']
                if n == 0:
                    return 0, 0
                
                # Proportion of positive sentiment (simplified)
                p = (row['sentiment_signal'] + 1) / 2  # Convert from [-1, 1] to [0, 1]
                p = np.clip(p, 0, 1)
                
                # Wilson score interval
                z = stats.norm.ppf((1 + confidence_level) / 2)
                denominator = 1 + z**2 / n
                centre = (p + z**2 / (2 * n)) / denominator
                half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
                
                lower = centre - half_width
                upper = centre + half_width
                
                # Convert back to [-1, 1] scale
                lower = 2 * lower - 1
                upper = 2 * upper - 1
                
                return lower, upper
            
            ci_results = df.apply(calculate_ci, axis=1, result_type='expand')
            df['sentiment_ci_lower'], df['sentiment_ci_upper'] = ci_results[0], ci_results[1]
            df['sentiment_ci_width'] = df['sentiment_ci_upper'] - df['sentiment_ci_lower']
        
        return df
    
    def create_composite_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite trading signal from multiple features.
        
        Args:
            df: DataFrame with individual signals
            
        Returns:
            Composite signal series
        """
        if df.empty:
            return pd.Series()
        
        # Define weights for different signal components
        weights = {
            'avg_sentiment': 0.4,
            'sentiment_volume': 0.2,
            'urgency_signal': 0.1,
            'technical_mention_signal': 0.1,
            'engagement_total': 0.1,
            'market_focus_signal': 0.1
        }
        
        # Normalize each component
        composite = pd.Series(0, index=df.index)
        
        for component, weight in weights.items():
            if component in df.columns:
                # Z-score normalization
                normalized = (df[component] - df[component].mean()) / (df[component].std() + 1e-10)
                composite += normalized * weight
        
        # Apply tanh activation to bound between -1 and 1
        composite = np.tanh(composite)
        
        return composite
    
    def save_model(self, output_path: str):
        """Save trained models and vectorizers"""
        if self.tfidf_vectorizer is not None:
            model_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'scaler': self.scaler,
                'config': self.config
            }
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(model_data, output_path)
            logger.info(f"Saved model to {output_path}")
    
    def load_model(self, model_path: str):
        """Load trained models"""
        try:
            model_data = joblib.load(model_path)
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.scaler = model_data['scaler']
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")


def main():
    """Command-line interface"""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Trading Signals from Tweets')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input cleaned tweets Parquet file')
    parser.add_argument('--output', type=str, default='data/signals/signals.parquet',
                       help='Output signals file path')
    parser.add_argument('--model-output', type=str, default='data/models/signal_model.joblib',
                       help='Output model file path')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_parquet(args.input)
    print(f"📥 Loaded {len(df)} cleaned tweets from {args.input}")
    
    # Generate signals
    generator = SignalGenerator(config)
    df_signals = generator.generate_signals(df)
    
    if not df_signals.empty:
        # Save signals
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_signals.to_parquet(args.output, engine='pyarrow', compression='snappy')
        
        # Save model
        generator.save_model(args.model_output)
        
        print(f"✅ Generated {len(df_signals)} signal windows")
        print(f"📊 Saved signals to {args.output}")
        print(f"🤖 Saved model to {args.model_output}")
        
        # Print sample signals
        print(f"\nSample signals (last 5 windows):")
        print(df_signals[['timestamp', 'composite_signal', 'avg_sentiment', 'tweet_count']].tail())
    else:
        print("❌ No signals generated")


if __name__ == "__main__":
    main()