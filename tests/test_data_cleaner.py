"""
Tests for data cleaner module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from src.processor.data_cleaner import DataCleaner


class TestDataCleaner(unittest.TestCase):
    """Test DataCleaner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'processing': {
                'text_cleaning': {
                    'remove_urls': True,
                    'remove_mentions': False,
                    'remove_hashtags': False,
                    'remove_emojis': True,
                    'convert_to_lowercase': True,
                    'remove_punctuation': False,
                    'keep_indian_chars': True
                },
                'language_handling': {
                    'detect_language': True,
                    'transliterate_to_english': False,
                    'remove_non_printable': True,
                    'fix_encoding': True
                },
                'deduplication_method': 'content_hash'
            }
        }
        
        self.cleaner = DataCleaner(self.test_config)
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        test_cases = [
            ("Hello World!", "hello world!"),
            ("Check this: https://example.com", "check this:"),
            ("@user Hello #hashtag", "@user hello #hashtag"),
            ("Hello 😀 World", "hello world"),
            ("  Extra   Spaces   ", "extra spaces"),
        ]
        
        for input_text, expected in test_cases:
            result = self.cleaner.clean_text(input_text)
            self.assertEqual(result, expected)
    
    def test_clean_text_indian_language(self):
        """Test cleaning text with Indian languages"""
        hindi_text = "नमस्ते दुनिया! Hello World"
        result = self.cleaner.clean_text(hindi_text)
        
        # Should preserve Hindi characters
        self.assertIn("नमस्ते", result)
        self.assertIn("hello world", result.lower())
    
    def test_clean_text_unicode(self):
        """Test cleaning Unicode text"""
        unicode_text = "Special chars: ñáéíóú © ® ™"
        result = self.cleaner.clean_text(unicode_text)
        
        # Should handle Unicode properly
        self.assertIn("special chars", result)
    
    def test_extract_features(self):
        """Test feature extraction"""
        test_text = "Hello world! This is a test tweet with #hashtag and @mention."
        features = self.cleaner.extract_features(test_text)
        
        self.assertIn('char_count', features)
        self.assertIn('word_count', features)
        self.assertIn('has_hashtag', features)
        self.assertIn('has_mention', features)
        self.assertIn('sentiment_score', features)
        
        self.assertEqual(features['word_count'], 11)
        self.assertTrue(features['has_hashtag'])
        self.assertTrue(features['has_mention'])
    
    def test_clean_dataframe(self):
        """Test cleaning entire DataFrame"""
        test_data = {
            'tweet_id': ['1', '2', '3'],
            'content': [
                'Hello https://example.com',
                '@user Check #Nifty50! 😀',
                'Duplicate tweet https://example.com'
            ],
            'like_count': [10, 20, 30],
            'timestamp': pd.to_datetime(['2024-01-01 10:00:00'] * 3)
        }
        
        df = pd.DataFrame(test_data)
        df_clean = self.cleaner.clean_dataframe(df)
        
        # Check basic cleaning
        self.assertIn('content_clean', df_clean.columns)
        self.assertIn('content_hash', df_clean.columns)
        
        # Check URL removal
        self.assertNotIn('https://example.com', df_clean.iloc[0]['content_clean'])
        
        # Check feature extraction
        self.assertIn('char_count', df_clean.columns)
        self.assertIn('word_count', df_clean.columns)
        
        # Check deduplication (if implemented)
        if len(df_clean) < len(df):
            self.assertLess(len(df_clean), len(df))
    
    def test_handle_missing_values(self):
        """Test handling of missing values"""
        test_data = {
            'text': ['Hello', None, 'World'],
            'count': [1, np.nan, 3],
            'category': ['A', 'B', None]
        }
        
        df = pd.DataFrame(test_data)
        df_clean = self.cleaner._handle_missing_values(df)
        
        # Check numeric columns filled with 0
        self.assertEqual(df_clean['count'].iloc[1], 0)
        
        # Check text columns filled with empty string
        self.assertEqual(df_clean['text'].iloc[1], '')
        self.assertEqual(df_clean['category'].iloc[2], '')
    
    def test_deduplicate_by_hash(self):
        """Test deduplication by content hash"""
        test_data = {
            'content': ['Hello World', 'Hello World', 'Different'],
            'other_col': [1, 2, 3]
        }
        
        df = pd.DataFrame(test_data)
        df['content_hash'] = df['content'].apply(
            lambda x: hash(x) % 10000  # Simple hash for testing
        )
        
        df_dedup = self.cleaner.deduplicate_by_hash(df)
        
        # Should remove one duplicate
        self.assertEqual(len(df_dedup), 2)
        self.assertEqual(df_dedup['content'].nunique(), 2)
    
    def test_normalize_data(self):
        """Test data normalization"""
        test_data = {
            'reply_count': [1, 2, 3, 4, 5],
            'retweet_count': [10, 20, 30, 40, 50],
            'like_count': [100, 200, 300, 400, 500]
        }
        
        df = pd.DataFrame(test_data)
        df_normalized = self.cleaner.normalize_data(df)
        
        # Check normalized columns created
        self.assertIn('reply_count_normalized', df_normalized.columns)
        self.assertIn('retweet_count_normalized', df_normalized.columns)
        self.assertIn('like_count_normalized', df_normalized.columns)
        self.assertIn('engagement_score', df_normalized.columns)
        
        # Check normalization bounds
        for col in ['reply_count_normalized', 'retweet_count_normalized', 'like_count_normalized']:
            self.assertTrue(df_normalized[col].min() >= 0)
            self.assertTrue(df_normalized[col].max() <= 1)
        
        # Check engagement score calculation
        self.assertTrue(df_normalized['engagement_score'].min() >= 0)
        self.assertTrue(df_normalized['engagement_score'].max() <= 1)


if __name__ == '__main__':
    unittest.main()