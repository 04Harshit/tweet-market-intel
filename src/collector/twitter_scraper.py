"""
Twitter/X Scraper for collecting tweets with Indian stock market hashtags.
Uses snscrape to avoid Twitter API restrictions.
"""

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Generator
import time
import random

import pandas as pd
from tqdm import tqdm

from ..utils.rate_limiter import RateLimiter
from ..utils.proxy_manager import ProxyManager

logger = logging.getLogger(__name__)


class TwitterScraper:
    """Scrapes tweets using snscrape without Twitter API"""
    
    def __init__(self, config: dict):
        self.config = config
        self.rate_limiter = RateLimiter(
            max_calls_per_minute=config['rate_limiting']['max_requests_per_minute'],
            max_calls_per_hour=config['rate_limiting']['max_requests_per_hour']
        )
        self.proxy_manager = ProxyManager(config['security']['proxy_list'])
        self.scraped_count = 0
        
    def scrape_tweets(
        self,
        hashtags: List[str] = None,
        search_terms: List[str] = None,
        hours: int = 24,
        min_tweets: int = 2000
    ) -> pd.DataFrame:
        """
        Scrape tweets for given hashtags and search terms.
        
        Args:
            hashtags: List of hashtags to search
            search_terms: List of search terms
            hours: Hours to look back
            min_tweets: Minimum number of tweets to collect
            
        Returns:
            DataFrame with scraped tweets
        """
        if hashtags is None:
            hashtags = self.config['scraping']['hashtags']
        if search_terms is None:
            search_terms = self.config['scraping']['search_terms']
        
        logger.info(f"Starting tweet collection for {len(hashtags)} hashtags")
        logger.info(f"Target: {min_tweets} tweets from last {hours} hours")
        
        all_tweets = []
        start_time = datetime.now()
        since_date = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d')
        
        # Scrape hashtags
        for hashtag in tqdm(hashtags, desc="Scraping hashtags"):
            try:
                tweets = self._scrape_hashtag(hashtag, since_date)
                all_tweets.extend(tweets)
                self.scraped_count += len(tweets)
                
                logger.info(f"Collected {len(tweets)} tweets for {hashtag}")
                
                # Check if we've reached target
                if self.scraped_count >= min_tweets:
                    logger.info(f"Reached target of {min_tweets} tweets")
                    break
                    
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                time.sleep(random.uniform(1, 3))  # Anti-bot delay
                
            except Exception as e:
                logger.error(f"Error scraping {hashtag}: {e}")
                continue
        
        # Scrape search terms if needed
        if self.scraped_count < min_tweets and search_terms:
            logger.info(f"Collecting additional tweets from search terms")
            for term in tqdm(search_terms, desc="Scraping search terms"):
                try:
                    tweets = self._scrape_search_term(term, since_date)
                    all_tweets.extend(tweets)
                    self.scraped_count += len(tweets)
                    
                    if self.scraped_count >= min_tweets:
                        break
                        
                    self.rate_limiter.wait_if_needed()
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    logger.error(f"Error searching for {term}: {e}")
                    continue
        
        # Convert to DataFrame
        df = self._tweets_to_dataframe(all_tweets)
        
        elapsed = datetime.now() - start_time
        logger.info(f"Total collected: {len(df)} tweets in {elapsed}")
        
        return df
    
    def _scrape_hashtag(self, hashtag: str, since_date: str) -> List[Dict]:
        """Scrape tweets for a specific hashtag"""
        max_results = self.config['scraping']['max_tweets_per_hashtag']
        
        # Build snscrape command
        cmd = [
            'snscrape',
            '--jsonl',
            '--max-results', str(max_results),
            'twitter-hashtag',
            f'{hashtag} since:{since_date}'
        ]
        
        return self._execute_scrape_command(cmd, hashtag)
    
    def _scrape_search_term(self, term: str, since_date: str) -> List[Dict]:
        """Scrape tweets for a search term"""
        max_results = self.config['scraping']['max_tweets_per_hashtag'] // 2
        
        cmd = [
            'snscrape',
            '--jsonl',
            '--max-results', str(max_results),
            'twitter-search',
            f'"{term}" since:{since_date} lang:en'
        ]
        
        return self._execute_scrape_command(cmd, term)
    
    def _execute_scrape_command(self, cmd: List[str], source: str) -> List[Dict]:
        """Execute snscrape command and parse results"""
        try:
            # Add proxy if configured
            if self.config['security']['use_proxy']:
                proxy = self.proxy_manager.get_proxy()
                if proxy:
                    cmd = ['proxychains'] + cmd  # Linux/Mac
                    # For Windows, would need different proxy handling
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"snscrape returned {result.returncode} for {source}")
                logger.warning(f"Error: {result.stderr[:200]}")
                return []
            
            # Parse JSONL output
            tweets = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        tweet_data = json.loads(line)
                        tweets.append(tweet_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error: {e}")
                        continue
            
            return tweets
            
        except subprocess.TimeoutExpired:
            logger.error(f"Scraping timeout for {source}")
            return []
        except Exception as e:
            logger.error(f"Error executing command for {source}: {e}")
            return []
    
    def _tweets_to_dataframe(self, tweets: List[Dict]) -> pd.DataFrame:
        """Convert raw tweet data to structured DataFrame"""
        processed_tweets = []
        
        for tweet in tweets:
            try:
                processed = {
                    'tweet_id': tweet.get('id', ''),
                    'username': tweet.get('user', {}).get('username', ''),
                    'user_id': tweet.get('user', {}).get('id', ''),
                    'content': tweet.get('content', ''),
                    'timestamp': tweet.get('date', ''),
                    'language': tweet.get('lang', ''),
                    'hashtags': tweet.get('hashtags', []),
                    'mentions': tweet.get('mentionedUsers', []),
                    'urls': tweet.get('urls', []),
                    'reply_count': tweet.get('replyCount', 0),
                    'retweet_count': tweet.get('retweetCount', 0),
                    'like_count': tweet.get('likeCount', 0),
                    'quote_count': tweet.get('quoteCount', 0),
                    'view_count': tweet.get('viewCount', 0),
                    'source': tweet.get('sourceLabel', ''),
                    'media_count': len(tweet.get('media', [])),
                    'is_sensitive': tweet.get('possiblySensitive', False),
                    'is_reply': tweet.get('inReplyToTweetId') is not None,
                    'collected_at': datetime.now().isoformat(),
                    'raw_data': json.dumps(tweet)  # Keep raw data for debugging
                }
                processed_tweets.append(processed)
                
            except Exception as e:
                logger.warning(f"Error processing tweet: {e}")
                continue
        
        df = pd.DataFrame(processed_tweets)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    async def scrape_async(self, hashtags: List[str], hours: int = 24) -> pd.DataFrame:
        """Async version of scrape method"""
        # Implementation for async scraping
        pass
    
    def save_tweets(self, df: pd.DataFrame, output_path: str):
        """Save tweets to Parquet file"""
        if df.empty:
            logger.warning("No tweets to save")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add partition columns
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Save to Parquet
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            partition_cols=['date', 'hour'] if self.config['storage']['partitioning']['enabled'] else None
        )
        
        logger.info(f"Saved {len(df)} tweets to {output_path}")


def main():
    """Command-line interface"""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Twitter Scraper for Market Intelligence')
    parser.add_argument('--hashtags', type=str, help='Comma-separated hashtags')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back')
    parser.add_argument('--limit', type=int, default=2000, help='Minimum tweets to collect')
    parser.add_argument('--output', type=str, default='data/raw/tweets.parquet', 
                       help='Output file path')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse hashtags
    if args.hashtags:
        hashtags = [h.strip() for h in args.hashtags.split(',')]
    else:
        hashtags = config['scraping']['hashtags']
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create scraper and collect tweets
    scraper = TwitterScraper(config)
    df = scraper.scrape_tweets(
        hashtags=hashtags,
        hours=args.hours,
        min_tweets=args.limit
    )
    
    # Save results
    if not df.empty:
        scraper.save_tweets(df, args.output)
        print(f"✅ Collected {len(df)} tweets")
        print(f"📊 Sample tweet: {df.iloc[0]['content'][:100]}...")
    else:
        print("❌ No tweets collected")


if __name__ == "__main__":
    main()