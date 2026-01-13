"""
Data cleaning and preprocessing for tweets.
Handles Unicode, Indian languages, and data normalization.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

import pandas as pd
import numpy as np
from ftfy import fix_text
import emoji
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and preprocesses tweet data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.clean_config = config['processing']['text_cleaning']
        self.lang_config = config['processing']['language_handling']
        
        # Regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.emoji_pattern = emoji.get_emoji_regexp()
        
        # Indian language patterns
        self.hindi_pattern = re.compile(r'[\u0900-\u097F]+')
        self.tamil_pattern = re.compile(r'[\u0B80-\u0BFF]+')
        self.telugu_pattern = re.compile(r'[\u0C00-\u0C7F]+')
        self.kannada_pattern = re.compile(r'[\u0C80-\u0CFF]+')
        self.malayalam_pattern = re.compile(r'[\u0D00-\u0D7F]+')
        
        # Non-printable characters
        self.non_printable_pattern = re.compile(r'[^\x20-\x7E\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]+')
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean entire DataFrame of tweets.
        
        Args:
            df: Raw tweet DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        logger.info(f"Starting data cleaning for {len(df)} tweets")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Apply cleaning to each tweet
        tqdm.pandas(desc="Cleaning tweets")
        df_clean['content_clean'] = df_clean['content'].progress_apply(self.clean_text)
        
        # Extract features
        df_clean['features'] = df_clean['content_clean'].apply(self.extract_features)
        
        # Expand features into columns
        features_df = pd.json_normalize(df_clean['features'])
        df_clean = pd.concat([df_clean, features_df], axis=1)
        
        # Drop the temporary columns
        df_clean = df_clean.drop(columns=['features'])
        
        # Add content hash for deduplication
        df_clean['content_hash'] = df_clean['content_clean'].apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()
        )
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Convert data types
        df_clean = self._convert_data_types(df_clean)
        
        # Deduplicate
        if self.config['processing']['deduplication_method'] == 'content_hash':
            df_clean = self.deduplicate_by_hash(df_clean)
        
        logger.info(f"Cleaned {len(df_clean)} tweets")
        
        return df_clean
    
    def clean_text(self, text: str) -> str:
        """
        Clean individual tweet text.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Fix encoding issues
        if self.lang_config['fix_encoding']:
            text = fix_text(text)
        
        # Remove URLs
        if self.clean_config['remove_urls']:
            text = self.url_pattern.sub('', text)
        
        # Remove mentions (optional)
        if self.clean_config['remove_mentions']:
            text = self.mention_pattern.sub('', text)
        else:
            # Extract mentions as features but keep in text
            pass
        
        # Remove hashtags (optional)
        if self.clean_config['remove_hashtags']:
            text = self.hashtag_pattern.sub('', text)
        
        # Remove emojis (optional)
        if self.clean_config['remove_emojis']:
            text = self.emoji_pattern.sub('', text)
        
        # Convert to lowercase
        if self.clean_config['convert_to_lowercase']:
            text = text.lower()
        
        # Remove non-printable characters
        if self.lang_config['remove_non_printable']:
            if self.lang_config['keep_indian_chars']:
                # Keep Indian language characters
                text = self.non_printable_pattern.sub('', text)
            else:
                # Remove all non-ASCII
                text = text.encode('ascii', 'ignore').decode()
        
        # Transliterate Indian languages to English (optional)
        if self.lang_config['transliterate_to_english']:
            text = self._transliterate_indian_text(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from cleaned text.
        
        Args:
            text: Cleaned tweet text
            
        Returns:
            Dictionary of features
        """
        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'has_emoji': bool(self.emoji_pattern.search(text)),
            'has_mention': bool(self.mention_pattern.search(text)),
            'has_hashtag': bool(self.hashtag_pattern.search(text)),
            'has_url': bool(self.url_pattern.search(text)),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        }
        
        # Language detection features
        if self.lang_config['detect_language']:
            lang_features = self._detect_indian_language(text)
            features.update(lang_features)
        
        # Sentiment indicators (simple)
        positive_words = ['bullish', 'buy', 'up', 'rise', 'gain', 'profit', 'positive']
        negative_words = ['bearish', 'sell', 'down', 'fall', 'loss', 'negative', 'crash']
        
        features['positive_word_count'] = sum(1 for word in positive_words if word in text.lower())
        features['negative_word_count'] = sum(1 for word in negative_words if word in text.lower())
        features['sentiment_score'] = features['positive_word_count'] - features['negative_word_count']
        
        # Market-specific indicators
        market_terms = ['nifty', 'sensex', 'stock', 'market', 'trade', 'investment', 'portfolio']
        features['market_term_count'] = sum(1 for term in market_terms if term in text.lower())
        
        return features
    
    def _detect_indian_language(self, text: str) -> Dict[str, bool]:
        """Detect presence of Indian languages in text"""
        return {
            'has_hindi': bool(self.hindi_pattern.search(text)),
            'has_tamil': bool(self.tamil_pattern.search(text)),
            'has_telugu': bool(self.telugu_pattern.search(text)),
            'has_kannada': bool(self.kannada_pattern.search(text)),
            'has_malayalam': bool(self.malayalam_pattern.search(text)),
            'has_any_indian_language': any([
                bool(self.hindi_pattern.search(text)),
                bool(self.tamil_pattern.search(text)),
                bool(self.telugu_pattern.search(text)),
                bool(self.kannada_pattern.search(text)),
                bool(self.malayalam_pattern.search(text))
            ])
        }
    
    def _transliterate_indian_text(self, text: str) -> str:
        """Transliterate Indian language text to English"""
        try:
            # Hindi
            if self.hindi_pattern.search(text):
                text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
            # Tamil
            if self.tamil_pattern.search(text):
                text = transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
            # Add other languages as needed
        except Exception as e:
            logger.warning(f"Transliteration error: {e}")
        
        return text
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame"""
        # Fill numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Fill text columns with empty string
        text_cols = df.select_dtypes(include=[object]).columns
        df[text_cols] = df[text_cols].fillna('')
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types"""
        # Convert timestamp columns
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        if 'collected_at' in df.columns:
            df['collected_at'] = pd.to_datetime(df['collected_at'], errors='coerce')
        
        # Convert numeric columns
        engagement_cols = ['reply_count', 'retweet_count', 'like_count', 
                          'quote_count', 'view_count', 'media_count']
        
        for col in engagement_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        return df
    
    def deduplicate_by_hash(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate tweets based on content hash"""
        initial_count = len(df)
        df_dedup = df.drop_duplicates(subset=['content_hash'], keep='first')
        
        removed = initial_count - len(df_dedup)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate tweets")
        
        return df_dedup
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features"""
        # Normalize engagement metrics
        engagement_cols = ['reply_count', 'retweet_count', 'like_count', 
                          'quote_count', 'view_count']
        
        for col in engagement_cols:
            if col in df.columns and df[col].max() > 0:
                df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Create composite engagement score
        if all(f'{col}_normalized' in df.columns for col in engagement_cols):
            weights = {'reply_count': 0.2, 'retweet_count': 0.3, 
                      'like_count': 0.3, 'quote_count': 0.1, 'view_count': 0.1}
            
            df['engagement_score'] = sum(
                df[f'{col}_normalized'] * weight 
                for col, weight in weights.items() 
                if f'{col}_normalized' in df.columns
            )
        
        return df


def main():
    """Command-line interface"""
    import yaml
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Data Cleaner for Tweets')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input Parquet file or directory')
    parser.add_argument('--output', type=str, default='data/processed/cleaned_tweets.parquet',
                       help='Output file path')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    input_path = Path(args.input)
    if input_path.is_dir():
        # Load all parquet files in directory
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_parquet(input_path)
    
    print(f"📥 Loaded {len(df)} tweets from {args.input}")
    
    # Clean data
    cleaner = DataCleaner(config)
    df_clean = cleaner.clean_dataframe(df)
    
    # Save cleaned data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_clean.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        partition_cols=['date', 'hour'] if config['storage']['partitioning']['enabled'] else None
    )
    
    print(f"✅ Cleaned {len(df_clean)} tweets")
    print(f"📊 Saved to {args.output}")
    
    # Print sample
    if not df_clean.empty:
        print(f"\nSample cleaned tweet:")
        print(f"Original: {df.iloc[0]['content'][:100]}...")
        print(f"Cleaned: {df_clean.iloc[0]['content_clean'][:100]}...")


if __name__ == "__main__":
    main()