"""
Efficient Parquet storage handler with partitioning and compression.
"""

import logging
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ParquetHandler:
    """Handles Parquet file operations with optimization"""
    
    def __init__(self, config: dict):
        self.config = config
        self.storage_config = config['storage']
        self.compression = self.storage_config['compression']
        self.partitioning_enabled = self.storage_config['partitioning']['enabled']
        self.partition_cols = self.storage_config['partitioning']['columns']
        
    def save_dataframe(
        self,
        df: pd.DataFrame,
        base_path: str,
        table_name: str,
        mode: str = 'append',
        partition_cols: Optional[List[str]] = None
    ) -> str:
        """
        Save DataFrame to Parquet format with optimization.
        
        Args:
            df: DataFrame to save
            base_path: Base directory path
            table_name: Name of the table/dataset
            mode: 'append', 'overwrite', or 'error'
            partition_cols: Columns to partition by
            
        Returns:
            Path where data was saved
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to save")
            return ""
        
        # Convert to Arrow table
        table = pa.Table.from_pandas(df)
        
        # Determine output path
        output_path = Path(base_path) / table_name
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine partitioning columns
        if partition_cols is None and self.partitioning_enabled:
            partition_cols = self._get_valid_partition_cols(df, self.partition_cols)
        
        if partition_cols:
            # Write partitioned dataset
            self._write_partitioned_dataset(table, output_path, partition_cols, mode)
        else:
            # Write single file
            self._write_single_file(table, output_path, mode)
        
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return str(output_path)
    
    def _write_partitioned_dataset(
        self,
        table: pa.Table,
        output_path: Path,
        partition_cols: List[str],
        mode: str
    ):
        """Write partitioned Parquet dataset"""
        # Ensure partition columns exist in table
        existing_cols = set(table.column_names)
        valid_partition_cols = [col for col in partition_cols if col in existing_cols]
        
        if not valid_partition_cols:
            logger.warning("No valid partition columns, writing as single file")
            self._write_single_file(table, output_path, mode)
            return
        
        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=valid_partition_cols,
            compression=self.compression,
            existing_data_behavior='overwrite_or_ignore' if mode == 'append' else 'delete_matching'
        )
    
    def _write_single_file(
        self,
        table: pa.Table,
        output_path: Path,
        mode: str
    ):
        """Write single Parquet file"""
        if mode == 'append' and output_path.exists():
            # Read existing data
            existing_table = pq.read_table(output_path)
            # Combine with new data
            combined_table = pa.concat_tables([existing_table, table])
            # Write combined data
            pq.write_table(
                combined_table,
                output_path,
                compression=self.compression
            )
        else:
            # Write new file
            pq.write_table(
                table,
                output_path,
                compression=self.compression
            )
    
    def _get_valid_partition_cols(
        self,
        df: pd.DataFrame,
        candidate_cols: List[str]
    ) -> List[str]:
        """Get valid partition columns from DataFrame"""
        valid_cols = []
        
        for col in candidate_cols:
            if col in df.columns:
                # Check if column has reasonable cardinality for partitioning
                unique_count = df[col].nunique()
                if 1 < unique_count < 100:  # Reasonable partition sizes
                    valid_cols.append(col)
                else:
                    logger.debug(f"Column {col} has {unique_count} unique values, skipping for partitioning")
        
        return valid_cols
    
    def load_dataframe(
        self,
        input_path: str,
        filters: Optional[List[tuple]] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data from Parquet file/directory.
        
        Args:
            input_path: Path to Parquet file or directory
            filters: PyArrow filters for predicate pushdown
            columns: Columns to load (None for all)
            
        Returns:
            Loaded DataFrame
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Path does not exist: {input_path}")
            return pd.DataFrame()
        
        try:
            if input_path.is_dir():
                # Load partitioned dataset
                dataset = pq.ParquetDataset(
                    input_path,
                    filters=filters,
                    use_legacy_dataset=False
                )
                table = dataset.read(columns=columns)
            else:
                # Load single file
                table = pq.read_table(
                    input_path,
                    filters=filters,
                    columns=columns
                )
            
            df = table.to_pandas()
            logger.info(f"Loaded {len(df)} rows from {input_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {input_path}: {e}")
            return pd.DataFrame()
    
    def query_data(
        self,
        input_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        hashtags: Optional[List[str]] = None,
        min_engagement: int = 0
    ) -> pd.DataFrame:
        """
        Query data with filters for efficient reading.
        
        Args:
            input_path: Path to Parquet data
            start_date: Start date filter
            end_date: End date filter
            hashtags: List of hashtags to filter
            min_engagement: Minimum engagement score
            
        Returns:
            Filtered DataFrame
        """
        filters = []
        
        # Date filters (if timestamp column exists and partitioning includes date)
        if start_date or end_date:
            date_filters = []
            
            if start_date:
                date_filters.append(('timestamp', '>=', start_date))
            if end_date:
                date_filters.append(('timestamp', '<=', end_date))
            
            filters.extend(date_filters)
        
        # Hashtag filter (if supported by schema)
        if hashtags:
            # This assumes hashtags are stored as an array
            pass
        
        # Engagement filter
        if min_engagement > 0:
            filters.append(('engagement_score', '>=', min_engagement))
        
        return self.load_dataframe(input_path, filters=filters if filters else None)
    
    def optimize_dataset(
        self,
        input_path: str,
        output_path: str,
        target_row_group_size: int = 100000
    ):
        """
        Optimize Parquet dataset by repartitioning and recompressing.
        
        Args:
            input_path: Input dataset path
            output_path: Output optimized dataset path
            target_row_group_size: Target rows per row group
        """
        logger.info(f"Optimizing dataset from {input_path} to {output_path}")
        
        # Read dataset
        dataset = pq.ParquetDataset(input_path, use_legacy_dataset=False)
        
        # Rewrite with optimization
        pq.write_table(
            dataset.read(),
            output_path,
            compression=self.compression,
            row_group_size=target_row_group_size
        )
        
        logger.info(f"Optimized dataset saved to {output_path}")
    
    def get_dataset_stats(self, input_path: str) -> Dict[str, Any]:
        """Get statistics about Parquet dataset"""
        stats = {
            'total_rows': 0,
            'num_files': 0,
            'total_size_mb': 0,
            'columns': [],
            'partition_info': {}
        }
        
        input_path = Path(input_path)
        
        if not input_path.exists():
            return stats
        
        try:
            if input_path.is_dir():
                # Get all Parquet files
                parquet_files = list(input_path.rglob("*.parquet"))
                stats['num_files'] = len(parquet_files)
                
                # Read first file for schema
                if parquet_files:
                    first_file = parquet_files[0]
                    schema = pq.read_schema(first_file)
                    stats['columns'] = schema.names
                    
                    # Get partitioning info
                    dataset = pq.ParquetDataset(input_path, use_legacy_dataset=False)
                    if hasattr(dataset, 'partitioning'):
                        stats['partition_info'] = {
                            'schema': str(dataset.partitioning.schema),
                            'depth': dataset.partitioning.partitioning_dictionary_depth
                        }
                
                # Calculate total size and rows
                total_size = 0
                total_rows = 0
                
                for file in parquet_files:
                    file_size = file.stat().st_size
                    total_size += file_size
                    
                    # Read metadata to get row count
                    metadata = pq.read_metadata(file)
                    total_rows += metadata.num_rows
                
                stats['total_rows'] = total_rows
                stats['total_size_mb'] = total_size / (1024 * 1024)
                
            else:
                # Single file
                stats['num_files'] = 1
                file_size = input_path.stat().st_size
                stats['total_size_mb'] = file_size / (1024 * 1024)
                
                metadata = pq.read_metadata(input_path)
                stats['total_rows'] = metadata.num_rows
                stats['columns'] = metadata.schema.names
        
        except Exception as e:
            logger.error(f"Error getting dataset stats: {e}")
        
        return stats
    
    def cleanup_old_data(
        self,
        data_path: str,
        retention_days: int = 30
    ):
        """
        Clean up old data files.
        
        Args:
            data_path: Path to data directory
            retention_days: Number of days to retain
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        data_path = Path(data_path)
        
        if not data_path.exists():
            return
        
        logger.info(f"Cleaning up data older than {cutoff_date}")
        
        # Find and delete old partitioned data
        for date_dir in data_path.glob("*"):
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date < cutoff_date:
                    import shutil
                    shutil.rmtree(date_dir)
                    logger.info(f"Removed old directory: {date_dir}")
            except ValueError:
                # Not a date directory
                continue


def main():
    """Command-line interface"""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Parquet Storage Handler')
    parser.add_argument('--action', choices=['save', 'load', 'query', 'stats', 'optimize', 'cleanup'],
                       required=True, help='Action to perform')
    parser.add_argument('--input', type=str, help='Input file/directory path')
    parser.add_argument('--output', type=str, help='Output file/directory path')
    parser.add_argument('--data', type=str, help='Data file to save (for save action)')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    handler = ParquetHandler(config)
    
    if args.action == 'save' and args.data and args.output:
        # Load data
        df = pd.read_parquet(args.data)
        handler.save_dataframe(df, args.output, 'tweets')
        print(f"✅ Saved data to {args.output}")
    
    elif args.action == 'load' and args.input:
        df = handler.load_dataframe(args.input)
        print(f"📊 Loaded {len(df)} rows")
        if not df.empty:
            print(df.head())
    
    elif args.action == 'query' and args.input:
        df = handler.query_data(args.input)
        print(f"📊 Queried {len(df)} rows")
    
    elif args.action == 'stats' and args.input:
        stats = handler.get_dataset_stats(args.input)
        print("📈 Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.action == 'optimize' and args.input and args.output:
        handler.optimize_dataset(args.input, args.output)
        print(f"✅ Optimized dataset saved to {args.output}")
    
    elif args.action == 'cleanup' and args.input:
        handler.cleanup_old_data(args.input)
        print("✅ Cleanup completed")
    
    else:
        print("❌ Invalid arguments")


if __name__ == "__main__":
    main()