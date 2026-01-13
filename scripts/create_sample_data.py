#!/usr/bin/env python3
"""
Create sample data for testing and demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json


def create_sample_tweets(n_tweets=100):
    """Create sample tweet data"""
    
    # Sample usernames
    usernames = ['trader_raj', 'market_wizard', 'nifty_expert', 
                 'sensex_analyst', 'intraday_king', 'stock_guru']
    
    # Sample content templates
    templates = [
        "{} looking bullish today! #Nifty50 #StockMarket",
        "Bearish signals on {} watch out! #Sensex",
        "{} showing strong support at {} levels",
        "Breaking: {} crosses {} mark! #BankNifty",
        "{} volatility increasing, trade carefully #Intraday",
        "Positive news for {} sector #Stocks"
    ]
    
    # Sample hashtags
    hashtag_groups = [
        ['#Nifty50', '#Sensex', '#Stocks'],
        ['#BankNifty', '#Trading', '#Market'],
        ['#Intraday', '#Investment', '#Portfolio'],
        ['#BSE', '#NSE', '#StockMarket'],
        ['#Bullish', '#Bearish', '#Trend']
    ]
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_tweets)
    
    tweets = []
    for i in range(n_tweets):
        username = np.random.choice(usernames)
        template = np.random.choice(templates)
        stock = np.random.choice(['Nifty', 'Sensex', 'Bank Nifty', 'Reliance', 'TCS', 'HDFC'])
        level = np.random.randint(18000, 22000)
        
        content = template.format(stock, level)
        
        # Add random hashtags
        hashtags = np.random.choice(hashtag_groups[np.random.randint(0, len(hashtag_groups))], 
                                   size=np.random.randint(1, 4), replace=False).tolist()
        
        tweet = {
            'tweet_id': f'sample_{i:04d}',
            'username': username,
            'user_id': f'user_{np.random.randint(1000, 9999)}',
            'content': content,
            'timestamp': dates[i],
            'language': 'en',
            'hashtags': hashtags,
            'mentions': [f'@{np.random.choice(usernames)}'] if np.random.random() > 0.7 else [],
            'urls': [] if np.random.random() > 0.8 else ['https://example.com/link'],
            'reply_count': np.random.randint(0, 10),
            'retweet_count': np.random.randint(0, 50),
            'like_count': np.random.randint(0, 100),
            'quote_count': np.random.randint(0, 5),
            'view_count': np.random.randint(100, 10000),
            'source': np.random.choice(['Twitter Web App', 'Twitter for iPhone', 'TweetDeck']),
            'media_count': 0 if np.random.random() > 0.9 else 1,
            'is_sensitive': False,
            'is_reply': np.random.random() > 0.8,
            'collected_at': datetime.now().isoformat()
        }
        tweets.append(tweet)
    
    df = pd.DataFrame(tweets)
    
    # Add partition columns
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    
    return df


def create_sample_signals(n_windows=50):
    """Create sample signal data"""
    
    start_date = datetime.now() - timedelta(days=2)
    dates = pd.date_range(start=start_date, periods=n_windows, freq='1H')
    
    # Generate realistic signal patterns
    base_signal = np.sin(np.linspace(0, 4*np.pi, n_windows))
    noise = np.random.normal(0, 0.1, n_windows)
    
    signals = []
    for i in range(n_windows):
        signal = {
            'timestamp': dates[i],
            'composite_signal': base_signal[i] + noise[i],
            'avg_sentiment': base_signal[i] * 0.8 + np.random.normal(0, 0.05),
            'sentiment_volume': np.random.uniform(0.5, 1.5),
            'tweet_count': np.random.randint(10, 100),
            'engagement_total': np.random.randint(100, 1000),
            'sentiment_ci_lower': base_signal[i] - 0.2,
            'sentiment_ci_upper': base_signal[i] + 0.2,
            'sentiment_ci_width': 0.4,
            'urgency_signal': np.random.uniform(0, 1),
            'market_focus_signal': np.random.uniform(0.3, 0.9),
            'technical_mention_signal': np.random.randint(0, 5),
            'has_nifty_hashtag': np.random.randint(0, 10),
            'has_sensex_hashtag': np.random.randint(0, 5)
        }
        signals.append(signal)
    
    df = pd.DataFrame(signals)
    return df


def main():
    """Main function to create sample data"""
    
    print("📊 Creating sample data...")
    
    # Create directories
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # 1. Create sample tweets
    print("  Creating sample tweets...")
    df_tweets = create_sample_tweets(100)
    tweet_path = data_dir / 'raw' / 'sample_tweets.parquet'
    tweet_path.parent.mkdir(exist_ok=True)
    df_tweets.to_parquet(tweet_path, engine='pyarrow', compression='snappy')
    
    # 2. Create sample cleaned data (simplified)
    print("  Creating sample cleaned data...")
    df_clean = df_tweets.copy()
    df_clean['content_clean'] = df_clean['content'].str.lower()
    df_clean['content_hash'] = df_clean['content'].apply(hash)
    df_clean['char_count'] = df_clean['content'].str.len()
    df_clean['word_count'] = df_clean['content'].str.split().str.len()
    
    clean_path = data_dir / 'processed' / 'sample_cleaned.parquet'
    clean_path.parent.mkdir(exist_ok=True)
    df_clean.to_parquet(clean_path, engine='pyarrow', compression='snappy')
    
    # 3. Create sample signals
    print("  Creating sample signals...")
    df_signals = create_sample_signals(50)
    signal_path = data_dir / 'signals' / 'sample_signals.parquet'
    signal_path.parent.mkdir(exist_ok=True)
    df_signals.to_parquet(signal_path, engine='pyarrow', compression='snappy')
    
    # 4. Create sample analysis results
    print("  Creating sample analysis results...")
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'total_tweets': len(df_tweets),
            'signal_windows': len(df_signals),
            'avg_signal_strength': df_signals['composite_signal'].abs().mean(),
            'signal_volatility': df_signals['composite_signal'].std(),
            'positive_signal_percentage': (df_signals['composite_signal'] > 0).mean() * 100
        },
        'signal_summary': {
            'mean': df_signals['composite_signal'].mean(),
            'std': df_signals['composite_signal'].std(),
            'min': df_signals['composite_signal'].min(),
            'max': df_signals['composite_signal'].max(),
            'q25': df_signals['composite_signal'].quantile(0.25),
            'median': df_signals['composite_signal'].median(),
            'q75': df_signals['composite_signal'].quantile(0.75)
        }
    }
    
    analysis_path = data_dir / 'analysis' / 'sample_results.json'
    analysis_path.parent.mkdir(exist_ok=True)
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"✅ Sample data created:")
    print(f"   - Tweets: {tweet_path} ({len(df_tweets)} rows)")
    print(f"   - Cleaned: {clean_path} ({len(df_clean)} rows)")
    print(f"   - Signals: {signal_path} ({len(df_signals)} rows)")
    print(f"   - Analysis: {analysis_path}")
    
    # 5. Create a simple visualization
    try:
        import plotly.graph_objects as go
        
        print("  Creating sample visualization...")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_signals['timestamp'],
            y=df_signals['composite_signal'],
            mode='lines',
            name='Composite Signal'
        ))
        fig.add_trace(go.Bar(
            x=df_signals['timestamp'],
            y=df_signals['tweet_count'],
            name='Tweet Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        fig.update_layout(
            title='Sample Market Signals',
            yaxis=dict(title='Signal Value'),
            yaxis2=dict(title='Tweet Volume', overlaying='y', side='right'),
            hovermode='x unified'
        )
        
        viz_path = data_dir / 'visualizations' / 'sample_dashboard.html'
        viz_path.parent.mkdir(exist_ok=True)
        fig.write_html(str(viz_path))
        print(f"   - Visualization: {viz_path}")
        
    except ImportError:
        print("   Note: Plotly not installed, skipping visualization")
    
    print("\n🎉 Sample data creation complete!")
    print("You can now run: python scripts/run_pipeline.py --mode test")


if __name__ == "__main__":
    main()