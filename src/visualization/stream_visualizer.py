"""
Low-memory visualization for large datasets.
Uses streaming, sampling, and efficient plotting techniques.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

# Set default template
pio.templates.default = "plotly_dark"

logger = logging.getLogger(__name__)


class StreamVisualizer:
    """Creates low-memory visualizations for large datasets"""
    
    def __init__(self, config: dict):
        self.config = config
        self.viz_config = config['visualization']
        self.max_points = self.viz_config['max_data_points']
        self.sampling_method = self.viz_config['sampling_method']
        
    def create_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            df: DataFrame with signals and metrics
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            logger.warning("Empty DataFrame, creating empty dashboard")
            return go.Figure()
        
        # Sample data if too large
        df_sampled = self.sample_data(df)
        
        # Create subplot grid
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Composite Trading Signal',
                'Sentiment Trend',
                'Tweet Volume Over Time',
                'Engagement Metrics',
                'Signal Confidence Intervals',
                'Hashtag Distribution'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Composite Signal
        if 'composite_signal' in df_sampled.columns and 'timestamp' in df_sampled.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_sampled['timestamp'],
                    y=df_sampled['composite_signal'],
                    mode='lines',
                    name='Composite Signal',
                    line=dict(color='cyan', width=2),
                    hovertemplate='Time: %{x}<br>Signal: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Sentiment Trend
        if 'avg_sentiment' in df_sampled.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_sampled['timestamp'],
                    y=df_sampled['avg_sentiment'],
                    mode='lines',
                    name='Avg Sentiment',
                    line=dict(color='lime', width=2),
                    hovertemplate='Time: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add confidence intervals if available
            if all(col in df_sampled.columns for col in ['sentiment_ci_lower', 'sentiment_ci_upper']):
                fig.add_trace(
                    go.Scatter(
                        x=df_sampled['timestamp'],
                        y=df_sampled['sentiment_ci_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_sampled['timestamp'],
                        y=df_sampled['sentiment_ci_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 0, 0.2)',
                        name='95% CI',
                        hoverinfo='skip'
                    ),
                    row=1, col=2
                )
        
        # 3. Tweet Volume
        if 'tweet_count' in df_sampled.columns:
            fig.add_trace(
                go.Bar(
                    x=df_sampled['timestamp'],
                    y=df_sampled['tweet_count'],
                    name='Tweet Volume',
                    marker_color='royalblue',
                    hovertemplate='Time: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Engagement Metrics
        engagement_cols = ['like_count', 'retweet_count', 'reply_count']
        available_engagement = [col for col in engagement_cols if col in df_sampled.columns]
        
        if available_engagement:
            for col in available_engagement:
                fig.add_trace(
                    go.Scatter(
                        x=df_sampled['timestamp'],
                        y=df_sampled[col],
                        mode='lines',
                        name=col.replace('_', ' ').title(),
                        hovertemplate='Time: %{x}<br>Count: %{y}<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        # 5. Signal Distribution (Histogram)
        if 'composite_signal' in df_sampled.columns:
            fig.add_trace(
                go.Histogram(
                    x=df_sampled['composite_signal'],
                    nbinsx=50,
                    name='Signal Distribution',
                    marker_color='orange',
                    opacity=0.7
                ),
                row=3, col=1
            )
        
        # 6. Hashtag Distribution (if raw data available)
        # This would need access to raw tweet data
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="Market Intelligence Dashboard",
            title_font_size=20,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_xaxes(title_text="Signal Value", row=3, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)
        
        return fig
    
    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample data to reduce number of points for visualization.
        
        Args:
            df: Original DataFrame
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= self.max_points:
            return df
        
        logger.info(f"Sampling {len(df)} points down to {self.max_points}")
        
        if self.sampling_method == 'random':
            # Random sampling
            return df.sample(n=self.max_points, random_state=42).sort_index()
        
        elif self.sampling_method == 'systematic':
            # Systematic sampling
            step = len(df) // self.max_points
            indices = list(range(0, len(df), step))[:self.max_points]
            return df.iloc[indices]
        
        elif self.sampling_method == 'reservoir':
            # Reservoir sampling
            reservoir = df.iloc[:self.max_points].copy()
            
            for i in range(self.max_points, len(df)):
                j = np.random.randint(0, i + 1)
                if j < self.max_points:
                    reservoir.iloc[j] = df.iloc[i]
            
            return reservoir.sort_index()
        
        elif self.sampling_method == 'time_based':
            # Time-based aggregation
            if 'timestamp' not in df.columns:
                return self.sample_data(df)  # Fallback
            
            # Resample to reduce points
            df_time = df.set_index('timestamp')
            resample_rule = self._get_resample_rule(len(df))
            
            # Aggregate numeric columns
            numeric_cols = df_time.select_dtypes(include=[np.number]).columns
            df_resampled = df_time[numeric_cols].resample(resample_rule).mean()
            
            # Reset index
            df_resampled = df_resampled.reset_index()
            
            return df_resampled
        
        else:
            # Default: random sampling
            return df.sample(n=min(self.max_points, len(df)), random_state=42).sort_index()
    
    def _get_resample_rule(self, data_points: int) -> str:
        """Determine resample rule based on data size"""
        if data_points > 10000:
            return '1H'  # Hourly
        elif data_points > 5000:
            return '30T'  # 30 minutes
        elif data_points > 2000:
            return '15T'  # 15 minutes
        elif data_points > 1000:
            return '10T'  # 10 minutes
        else:
            return '5T'  # 5 minutes
    
    def create_streaming_plot(self, df: pd.DataFrame, window_hours: int = 24) -> go.Figure:
        """
        Create a streaming plot that shows recent data.
        
        Args:
            df: DataFrame with timestamp index
            window_hours: Number of hours to show
            
        Returns:
            Streaming plot figure
        """
        if df.empty or 'timestamp' not in df.columns:
            return go.Figure()
        
        # Filter to recent data
        cutoff = datetime.now() - timedelta(hours=window_hours)
        df_recent = df[df['timestamp'] >= cutoff].copy()
        
        if df_recent.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add signals
        signal_cols = ['composite_signal', 'avg_sentiment', 'weighted_sentiment']
        colors = ['cyan', 'lime', 'yellow']
        
        for col, color in zip(signal_cols, colors):
            if col in df_recent.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_recent['timestamp'],
                        y=df_recent[col],
                        mode='lines',
                        name=col.replace('_', ' ').title(),
                        line=dict(color=color, width=2),
                        hovertemplate='Time: %{x}<br>Value: %{y:.3f}<extra></extra>'
                    )
                )
        
        # Add volume as background bars
        if 'tweet_count' in df_recent.columns:
            fig.add_trace(
                go.Bar(
                    x=df_recent['timestamp'],
                    y=df_recent['tweet_count'],
                    name='Tweet Volume',
                    marker_color='rgba(100, 100, 255, 0.3)',
                    yaxis='y2',
                    hovertemplate='Time: %{x}<br>Count: %{y}<extra></extra>'
                )
            )
        
        # Update layout for dual axes
        fig.update_layout(
            title=f"Real-time Signals (Last {window_hours} hours)",
            xaxis_title="Time",
            yaxis_title="Signal Value",
            yaxis2=dict(
                title="Tweet Volume",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_heatmap(self, df: pd.DataFrame, value_col: str = 'composite_signal') -> go.Figure:
        """
        Create heatmap of signals by hour and day.
        
        Args:
            df: DataFrame with timestamp
            value_col: Column to visualize
            
        Returns:
            Heatmap figure
        """
        if df.empty or 'timestamp' not in df.columns or value_col not in df.columns:
            return go.Figure()
        
        # Extract hour and day
        df_heat = df.copy()
        df_heat['hour'] = df_heat['timestamp'].dt.hour
        df_heat['day'] = df_heat['timestamp'].dt.day_name()
        
        # Pivot for heatmap
        heat_data = df_heat.pivot_table(
            values=value_col,
            index='day',
            columns='hour',
            aggfunc='mean'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heat_data = heat_data.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=heat_data.values,
            x=list(range(24)),
            y=heat_data.index,
            colorscale='RdYlGn',
            zmid=0,
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Signal: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Heatmap of {value_col.replace('_', ' ').title()} by Hour and Day",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week"
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, output_path: str):
        """
        Save visualization to file.
        
        Args:
            fig: Plotly figure
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension
        if output_path.suffix == '.html':
            fig.write_html(output_path)
        elif output_path.suffix == '.png':
            fig.write_image(output_path, width=1600, height=900)
        elif output_path.suffix == '.pdf':
            fig.write_image(output_path, format='pdf', width=1600, height=900)
        else:
            # Default to HTML
            fig.write_html(str(output_path.with_suffix('.html')))
        
        logger.info(f"Saved visualization to {output_path}")


def main():
    """Command-line interface"""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Visualization')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input signals Parquet file')
    parser.add_argument('--output', type=str, default='visualizations/dashboard.html',
                       help='Output visualization file path')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Configuration file path')
    parser.add_argument('--type', choices=['dashboard', 'streaming', 'heatmap'],
                       default='dashboard', help='Visualization type')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_parquet(args.input)
    print(f"📥 Loaded {len(df)} signal points from {args.input}")
    
    # Create visualizer
    visualizer = StreamVisualizer(config)
    
    # Create visualization
    if args.type == 'dashboard':
        fig = visualizer.create_dashboard(df)
    elif args.type == 'streaming':
        fig = visualizer.create_streaming_plot(df)
    elif args.type == 'heatmap':
        fig = visualizer.create_heatmap(df)
    else:
        fig = visualizer.create_dashboard(df)
    
    # Save visualization
    visualizer.save_visualization(fig, args.output)
    
    print(f"✅ Created {args.type} visualization")
    print(f"📊 Saved to {args.output}")
    
    # Show in browser if interactive
    try:
        import sys
        if '--show' in sys.argv:
            fig.show()
    except:
        pass


if __name__ == "__main__":
    main()