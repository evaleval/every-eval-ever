#!/usr/bin/env python3
"""
Generate comprehensive statistics from all uploaded data by processing shards incrementally.
This avoids memory issues by processing one shard at a time and maintaining running statistics.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from huggingface_hub import HfApi, list_repo_files
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IncrementalStatsCalculator:
    """Incrementally calculate statistics across multiple parquet shards."""
    
    def __init__(self):
        self.running_stats = {}  # benchmark_model_key -> stats dict
        self.processed_shards = 0
        self.total_records = 0
    
    def update_with_shard(self, shard_df: pd.DataFrame, source_name: str) -> None:
        """Update running statistics with data from a new shard."""
        logger.info(f"📊 Processing shard with {len(shard_df):,} records")
        
        # Group by benchmark and model
        grouped = shard_df.groupby(['dataset_name', 'model_name'])
        
        for (dataset_name, model_name), group in grouped:
            key = f"{dataset_name}|{model_name}"
            
            # Extract current batch statistics
            scores = group['evaluation_score'].dropna()
            if len(scores) == 0:
                continue
            
            batch_stats = {
                'count': len(scores),
                'sum': scores.sum(),
                'sum_squares': (scores ** 2).sum(),
                'min_score': scores.min(),
                'max_score': scores.max(),
                'model_family': group['model_family'].iloc[0] if 'model_family' in group else 'unknown',
                'evaluation_method': group['evaluation_method_name'].iloc[0] if 'evaluation_method_name' in group else 'unknown',
                'source': source_name,
                'dataset_name': dataset_name,
                'model_name': model_name
            }
            
            # Update running statistics
            if key in self.running_stats:
                # Merge with existing stats
                existing = self.running_stats[key]
                
                # Combine counts and sums
                new_count = existing['count'] + batch_stats['count']
                new_sum = existing['sum'] + batch_stats['sum']
                new_sum_squares = existing['sum_squares'] + batch_stats['sum_squares']
                
                # Update min/max
                new_min = min(existing['min_score'], batch_stats['min_score'])
                new_max = max(existing['max_score'], batch_stats['max_score'])
                
                self.running_stats[key] = {
                    'count': new_count,
                    'sum': new_sum,
                    'sum_squares': new_sum_squares,
                    'min_score': new_min,
                    'max_score': new_max,
                    'model_family': batch_stats['model_family'],
                    'evaluation_method': batch_stats['evaluation_method'],
                    'source': batch_stats['source'],
                    'dataset_name': batch_stats['dataset_name'],
                    'model_name': batch_stats['model_name']
                }
            else:
                # First time seeing this combination
                self.running_stats[key] = batch_stats
        
        self.processed_shards += 1
        self.total_records += len(shard_df)
        logger.info(f"✅ Processed shard. Running totals: {len(self.running_stats)} combinations, {self.total_records:,} records")
    
    def finalize_statistics(self) -> pd.DataFrame:
        """Convert running statistics to final DataFrame with calculated metrics."""
        logger.info(f"📊 Finalizing statistics for {len(self.running_stats)} benchmark-model combinations")
        
        final_stats = []
        
        for key, stats in self.running_stats.items():
            count = stats['count']
            mean_score = stats['sum'] / count
            
            # Calculate standard deviation
            variance = (stats['sum_squares'] / count) - (mean_score ** 2)
            std_score = np.sqrt(max(0, variance))  # Ensure non-negative
            
            final_stats.append({
                'source': stats['source'],
                'dataset_name': stats['dataset_name'],
                'model_name': stats['model_name'],
                'model_family': stats['model_family'],
                'evaluation_method_name': stats['evaluation_method'],
                'total_samples': count,
                'accuracy': mean_score,  # For binary tasks, mean score = accuracy
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': stats['min_score'],
                'max_score': stats['max_score'],
                'processed_at': datetime.now().isoformat(),
                'comprehensive_stats_generated_at': datetime.now().isoformat(),
                'pipeline_stage': 'comprehensive_statistics',
                'data_freshness_hours': 0,  # Will be calculated based on source data timestamps
            })
        
        df = pd.DataFrame(final_stats)
        
        # Sort by dataset and accuracy for better readability
        df = df.sort_values(['dataset_name', 'accuracy'], ascending=[True, False])
        
        logger.info(f"✅ Generated comprehensive statistics:")
        logger.info(f"   📊 {len(df)} benchmark-model combinations")
        logger.info(f"   🎯 {df['dataset_name'].nunique()} unique benchmarks")
        logger.info(f"   🤖 {df['model_name'].nunique()} unique models")
        logger.info(f"   📈 Processed {self.total_records:,} total evaluation records")
        
        return df


def get_data_shards(api: HfApi, repo_id: str, source_name: str) -> list:
    """Get list of data shard files from the repository."""
    try:
        files = list_repo_files(repo_id=repo_id, repo_type='dataset')
        
        # Filter for source-specific parquet files, excluding stats files
        shard_files = [
            f for f in files 
            if f.endswith('.parquet') 
            and source_name in f 
            and not f.startswith('stats-')
        ]
        
        logger.info(f"📁 Found {len(shard_files)} data shards for source '{source_name}'")
        for f in sorted(shard_files):
            logger.info(f"   📄 {f}")
        
        return sorted(shard_files)
    
    except Exception as e:
        logger.error(f"❌ Failed to list repository files: {e}")
        return []


def process_shard_incrementally(api: HfApi, repo_id: str, shard_file: str, 
                               calculator: IncrementalStatsCalculator, source_name: str) -> bool:
    """Download and process a single shard, updating running statistics."""
    try:
        logger.info(f"⬇️ Processing shard: {shard_file}")
        
        # Download shard to temporary file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            api.hf_hub_download(
                repo_id=repo_id,
                repo_type='dataset',
                filename=shard_file,
                local_dir_use_symlinks=False,
                local_dir=str(Path(tmp_path).parent),
                local_filename=Path(tmp_path).name
            )
            
            # Load and process shard
            shard_df = pd.read_parquet(tmp_path)
            calculator.update_with_shard(shard_df, source_name)
            
            return True
            
        finally:
            # Clean up temporary file
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
                
    except Exception as e:
        logger.error(f"❌ Failed to process shard {shard_file}: {e}")
        return False


def upload_comprehensive_stats(api: HfApi, stats_df: pd.DataFrame, stats_repo_id: str, 
                              source_name: str, token: str) -> None:
    """Upload comprehensive statistics to the stats repository."""
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_filename = f"comprehensive_stats_{source_name}_{timestamp}.parquet"
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        stats_df.to_parquet(tmp_path, index=False)
        
        # Get file size for logging
        file_size = Path(tmp_path).stat().st_size / (1024 * 1024)
        
        logger.info(f"☁️ Uploading comprehensive statistics: {stats_filename} ({file_size:.1f} MB)")
        
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=stats_filename,
            repo_id=stats_repo_id,
            repo_type='dataset',
            token=token,
        )
        
        logger.info(f"✅ Uploaded comprehensive statistics to {stats_repo_id}")
        
        # Show top performers
        if len(stats_df) > 0:
            logger.info("🏆 Top 5 performers:")
            top_5 = stats_df.nlargest(5, 'accuracy')
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                logger.info(f"   {i}. {row['model_name']} on {row['dataset_name']}: {row['accuracy']*100:.1f}%")
        
    finally:
        # Clean up temporary file
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive statistics from all uploaded data")
    parser.add_argument("--main-repo-id", required=True, help="Main dataset repository ID")
    parser.add_argument("--stats-repo-id", required=True, help="Statistics dataset repository ID")
    parser.add_argument("--source-name", required=True, help="Source name to process (e.g., 'helm')")
    
    args = parser.parse_args()
    
    # Get HF token
    token = os.environ.get('HF_TOKEN')
    if not token:
        logger.error("❌ HF_TOKEN environment variable not set")
        return 1
    
    try:
        logger.info("🚀 Starting comprehensive statistics generation...")
        logger.info(f"📊 Source: {args.source_name}")
        logger.info(f"📁 Main repo: {args.main_repo_id}")
        logger.info(f"📈 Stats repo: {args.stats_repo_id}")
        
        # Setup HuggingFace API
        api = HfApi()
        
        # Get list of data shards
        shard_files = get_data_shards(api, args.main_repo_id, args.source_name)
        
        if not shard_files:
            logger.warning("⚠️ No data shards found. Nothing to process.")
            return 0
        
        # Initialize incremental calculator
        calculator = IncrementalStatsCalculator()
        
        # Process each shard incrementally
        successful_shards = 0
        for shard_file in shard_files:
            if process_shard_incrementally(api, args.main_repo_id, shard_file, calculator, args.source_name):
                successful_shards += 1
            else:
                logger.warning(f"⚠️ Skipped failed shard: {shard_file}")
        
        if successful_shards == 0:
            logger.error("❌ No shards processed successfully")
            return 1
        
        logger.info(f"✅ Processed {successful_shards}/{len(shard_files)} shards successfully")
        
        # Finalize statistics
        comprehensive_stats = calculator.finalize_statistics()
        
        # Upload comprehensive statistics
        upload_comprehensive_stats(api, comprehensive_stats, args.stats_repo_id, args.source_name, token)
        
        logger.info("🎉 Comprehensive statistics generation complete!")
        return 0
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
