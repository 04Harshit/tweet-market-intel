"""
Tests for signal generator module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.analyzer.signal_generator import SignalGenerator


class TestSignalGenerator(unittest.TestCase):
    """Test SignalGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'analysis': {
                'feature_extraction': {
                    'method': 'tfidf',
                    'tfidf_max_features': 100,
                    'tfidf_ngram_range': [1, 2],
                    'embedding_model': 'paraphrase-MiniLM-L6-v2'
                },
                'signal_generation': {
                    'window_size': 60,
                    'aggregation_method': 'weighted_average',
                    'confidence_interval': 0.95,
                    'min_samples_for_signal': 10
                }
            }
        }
        
        self.generator = SignalGenerator(self.test_config)
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'content_clean': ['Test tweet ' + str(i) for i in range(100)],
            'sentiment_score': np.random.uniform(-1, 1, 100),
            'like_count': np.random.randint(0, 100, 100),
            'retweet_count': np.random.randint(0, 50, 100),
            'reply_count': np.random.randint(0, 20, 100),
            'exclamation_count': np.random.randint(0, 3, 100),
            'question_count': np.random.randint(0, 2, 100),
            'uppercase_ratio': np.random.uniform(0, 0.5, 100),
            'market_term_count': np.random.randint(0, 5, 100),
            'hashtags': [['#test'] for _ in range(100)]
        })
    
    def test_extract_text_features(self):
        """Test text feature extraction"""
        test_texts = [
            "This is a bullish tweet about Nifty 50",
            "Bearish sentiment on Sensex today",
            "Market showing mixed signals"
        ]
        
        features = self.generator.extract_text_features(test_texts)
        
        self.assertEqual(len(features), len(test_texts))
        
        # Check structure
        for tfidf_features, embeddings in features:
            self.assertIsInstance(tfidf_features, np.ndarray)
            # TF-IDF features should have the right dimension
            self.assertEqual(len(tfidf_features), self.test_config['analysis']['feature_extraction']['tfidf_max_features'])
    
    @patch('src.analyzer.signal_generator.SentenceTransformer')
    def test_get_embeddings(self, mock_transformer):
        """Test getting sentence embeddings"""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(384)
        mock_transformer.return_value = mock_model
        
        text = "Test tweet for embeddings"
        embeddings = self.generator._get_embeddings(text)
        
        self.assertIsInstance(embeddings, np.ndarray)
        mock_model.encode.assert_called_once_with([text])
    
    def test_generate_individual_signals(self):
        """Test generating individual signals"""
        df_with_signals = self.generator._generate_individual_signals(self.test_df.copy())
        
        # Check signal columns created
        expected_columns = [
            'sentiment_signal',
            'weighted_sentiment',
            'urgency_signal',
            'market_focus_signal',
            'technical_mention_signal',
            'has_nifty_hashtag',
            'has_sensex_hashtag'
        ]
        
        for col in expected_columns:
            self.assertIn(col, df_with_signals.columns)
        
        # Check signal values
        self.assertTrue(all(df_with_signals['sentiment_signal'].isin([-1, 0, 1])))
        self.assertTrue(df_with_signals['urgency_signal'].min() >= 0)
        
        # Check hashtag signals
        self.assertTrue(all(df_with_signals['has_nifty_hashtag'].isin([0, 1])))
        self.assertTrue(all(df_with_signals['has_sensex_hashtag'].isin([0, 1])))
    
    def test_aggregate_signals(self):
        """Test signal aggregation"""
        # Add required columns
        df = self.test_df.copy()
        df['sentiment_signal'] = np.random.choice([-1, 0, 1], 100)
        df['weighted_sentiment'] = np.random.uniform(-1, 1, 100)
        df['urgency_signal'] = np.random.uniform(0, 1, 100)
        df['market_focus_signal'] = np.random.uniform(0, 1, 100)
        df['technical_mention_signal'] = np.random.randint(0, 2, 100)
        df['has_nifty_hashtag'] = np.random.randint(0, 2, 100)
        df['has_sensex_hashtag'] = np.random.randint(0, 2, 100)
        
        aggregated = self.generator.aggregate_signals(df)
        
        # Check aggregation results
        self.assertIn('tweet_count', aggregated.columns)
        self.assertIn('avg_sentiment', aggregated.columns)
        self.assertIn('sentiment_volume', aggregated.columns)
        
        # Check that aggregation reduced number of rows
        self.assertLess(len(aggregated), len(df))
        
        # Check tweet count aggregation
        total_tweets = aggregated['tweet_count'].sum()
        self.assertEqual(total_tweets, len(df))
    
    def test_calculate_confidence_intervals(self):
        """Test confidence interval calculation"""
        test_data = {
            'tweet_count': [100, 50, 30, 20],
            'sentiment_signal': [0.8, 0.2, -0.3, -0.7]
        }
        
        df = pd.DataFrame(test_data)
        df_with_ci = self.generator.calculate_confidence_intervals(df)
        
        # Check CI columns created
        ci_columns = ['sentiment_ci_lower', 'sentiment_ci_upper', 'sentiment_ci_width']
        
        for col in ci_columns:
            self.assertIn(col, df_with_ci.columns)
        
        # Check CI bounds
        for _, row in df_with_ci.iterrows():
            if row['tweet_count'] > 0:
                self.assertLessEqual(row['sentiment_ci_lower'], row['sentiment_signal'])
                self.assertGreaterEqual(row['sentiment_ci_upper'], row['sentiment_signal'])
                self.assertGreaterEqual(row['sentiment_ci_width'], 0)
    
    def test_create_composite_signal(self):
        """Test composite signal creation"""
        test_data = {
            'avg_sentiment': np.random.uniform(-1, 1, 50),
            'sentiment_volume': np.random.uniform(0, 10, 50),
            'urgency_signal': np.random.uniform(0, 1, 50),
            'technical_mention_signal': np.random.uniform(0, 1, 50),
            'engagement_total': np.random.uniform(0, 1000, 50),
            'market_focus_signal': np.random.uniform(0, 1, 50)
        }
        
        df = pd.DataFrame(test_data)
        composite_signal = self.generator.create_composite_signal(df)
        
        # Check signal properties
        self.assertEqual(len(composite_signal), len(df))
        self.assertTrue(all(composite_signal.between(-1, 1)))
        
        # Check signal statistics
        signal_mean = composite_signal.mean()
        signal_std = composite_signal.std()
        
        self.assertGreater(signal_std, 0)  # Should have some variation
    
    def test_generate_signals_integration(self):
        """Test complete signal generation integration"""
        # Mock TF-IDF to avoid actual training
        with patch.object(self.generator, 'extract_text_features') as mock_extract:
            # Mock return value
            mock_features = []
            for i in range(len(self.test_df)):
                tfidf = np.random.randn(self.test_config['analysis']['feature_extraction']['tfidf_max_features'])
                mock_features.append((tfidf, None))
            
            mock_extract.return_value = mock_features
            
            # Generate signals
            df_signals = self.generator.generate_signals(self.test_df)
            
            # Check results
            self.assertIsInstance(df_signals, pd.DataFrame)
            self.assertGreater(len(df_signals), 0)
            
            # Check signal columns
            expected_signal_cols = [
                'composite_signal',
                'avg_sentiment',
                'sentiment_volume',
                'tweet_count'
            ]
            
            for col in expected_signal_cols:
                self.assertIn(col, df_signals.columns)


if __name__ == '__main__':
    unittest.main()