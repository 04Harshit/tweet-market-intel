"""
Tests for Twitter scraper module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta

from src.collector.twitter_scraper import TwitterScraper


class TestTwitterScraper(unittest.TestCase):
    """Test TwitterScraper class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'scraping': {
                'hashtags': ['#test1', '#test2'],
                'search_terms': ['test term'],
                'time_window_hours': 24,
                'min_tweets_target': 100,
                'max_tweets_per_hashtag': 50,
                'language_filter': ['en']
            },
            'rate_limiting': {
                'max_requests_per_minute': 10,
                'max_requests_per_hour': 100
            },
            'security': {
                'use_proxy': False,
                'proxy_list': []
            },
            'storage': {
                'partitioning': {
                    'enabled': True
                }
            }
        }
        
        self.temp_dir = tempfile.mkdtemp()
        self.scraper = TwitterScraper(self.test_config)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.collector.twitter_scraper.subprocess.run')
    def test_scrape_hashtag_success(self, mock_subprocess):
        """Test successful hashtag scraping"""
        # Mock subprocess output
        mock_output = Mock()
        mock_output.returncode = 0
        mock_output.stdout = '{"id": "123", "content": "Test tweet", "user": {"username": "test_user"}}\n'
        mock_subprocess.return_value = mock_output
        
        result = self.scraper._scrape_hashtag('#test', '2024-01-01')
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], '123')
        self.assertEqual(result[0]['content'], 'Test tweet')
    
    @patch('src.collector.twitter_scraper.subprocess.run')
    def test_scrape_hashtag_failure(self, mock_subprocess):
        """Test hashtag scraping failure"""
        mock_output = Mock()
        mock_output.returncode = 1
        mock_output.stderr = 'Error occurred'
        mock_subprocess.return_value = mock_output
        
        result = self.scraper._scrape_hashtag('#test', '2024-01-01')
        
        self.assertEqual(result, [])
    
    def test_tweets_to_dataframe(self):
        """Test tweet data conversion to DataFrame"""
        test_tweets = [
            {
                'id': '123',
                'content': 'Test tweet 1',
                'user': {'username': 'user1', 'id': 'u1'},
                'date': '2024-01-01T10:00:00',
                'lang': 'en',
                'hashtags': ['#test'],
                'mentionedUsers': [],
                'urls': [],
                'replyCount': 1,
                'retweetCount': 2,
                'likeCount': 3,
                'quoteCount': 0,
                'viewCount': 100,
                'sourceLabel': 'Twitter Web App',
                'media': [],
                'possiblySensitive': False
            },
            {
                'id': '124',
                'content': 'Test tweet 2',
                'user': {'username': 'user2', 'id': 'u2'},
                'date': '2024-01-01T11:00:00',
                'lang': 'en',
                'hashtags': ['#test', '#example'],
                'mentionedUsers': [{'username': 'user1'}],
                'urls': ['https://example.com'],
                'replyCount': 0,
                'retweetCount': 1,
                'likeCount': 2,
                'quoteCount': 1,
                'viewCount': 50,
                'sourceLabel': 'Twitter for iPhone',
                'media': [{'type': 'photo'}],
                'possiblySensitive': True
            }
        ]
        
        df = self.scraper._tweets_to_dataframe(test_tweets)
        
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['tweet_id'], '123')
        self.assertEqual(df.iloc[1]['tweet_id'], '124')
        self.assertEqual(df.iloc[1]['media_count'], 1)
        self.assertEqual(df.iloc[1]['is_sensitive'], True)
        
        # Check column types
        self.assertIn('timestamp', df.columns)
        self.assertIn('content', df.columns)
        self.assertIn('username', df.columns)
    
    def test_save_tweets(self):
        """Test saving tweets to Parquet"""
        # Create test DataFrame
        test_data = {
            'tweet_id': ['123', '124'],
            'username': ['user1', 'user2'],
            'content': ['Test 1', 'Test 2'],
            'timestamp': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']),
            'engagement_score': [1.0, 2.0]
        }
        df = pd.DataFrame(test_data)
        
        # Add partition columns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        
        output_path = Path(self.temp_dir) / 'test_tweets.parquet'
        
        # Save tweets
        self.scraper.save_tweets(df, str(output_path))
        
        # Verify file was created
        self.assertTrue(output_path.exists())
        
        # Load and verify
        loaded_df = pd.read_parquet(output_path)
        self.assertEqual(len(loaded_df), 2)
        self.assertEqual(loaded_df.iloc[0]['tweet_id'], '123')
    
    @patch.object(TwitterScraper, '_scrape_hashtag')
    @patch.object(TwitterScraper, '_scrape_search_term')
    def test_scrape_tweets_integration(self, mock_search, mock_hashtag):
        """Test complete scraping integration"""
        # Mock return values
        mock_hashtag.return_value = [
            {'id': '123', 'content': 'Test', 'user': {'username': 'test'}, 'date': '2024-01-01'}
        ]
        mock_search.return_value = [
            {'id': '124', 'content': 'Search result', 'user': {'username': 'test2'}, 'date': '2024-01-01'}
        ]
        
        df = self.scraper.scrape_tweets(
            hashtags=['#test'],
            search_terms=['test'],
            hours=24,
            min_tweets=10
        )
        
        self.assertGreater(len(df), 0)
        mock_hashtag.assert_called()
        mock_search.assert_called()
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame()
        
        # Test saving empty DataFrame
        output_path = Path(self.temp_dir) / 'empty.parquet'
        self.scraper.save_tweets(empty_df, str(output_path))
        
        # File should not be created
        self.assertFalse(output_path.exists())


if __name__ == '__main__':
    unittest.main()
