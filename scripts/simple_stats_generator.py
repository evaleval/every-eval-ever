#!/usr/bin/env python3
"""
Generate comprehensive statistics using the datasets library directly.
This is much simpler and more reliable than manually downloading shards.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from huggingface_hub import HfApi
import json

# Try to import datasets library
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logging.error("❌ datasets library not available. Install with: pip install datasets")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_comprehensive_stats(repo_id: str, stats_repo_id: str, source_name: str = "helm") -> Dict[str, Any]:
    """Generate comprehensive statistics using datasets library."""
    
    logger.info(f"📊 Loading dataset from {repo_id}")
    
    try:
        # Load the entire dataset using datasets library
        dataset = load_dataset(repo_id, split='train')
        logger.info(f"✅ Loaded {len(dataset):,} records")
        
        # Convert to pandas for easier analysis
        df = dataset.to_pandas()
        logger.info(f"📊 Converting to pandas DataFrame: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return {}
    
    # Generate comprehensive statistics
    stats = {}
    
    # Basic statistics
    stats['overview'] = {
        'total_records': len(df),
        'total_unique_tasks': df['task_id'].nunique() if 'task_id' in df.columns else 0,
        'total_models': df['model_name'].nunique() if 'model_name' in df.columns else 0,
        'total_datasets': df['dataset_name'].nunique() if 'dataset_name' in df.columns else 0,
        'data_sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
        'benchmarks': df['benchmark'].value_counts().to_dict() if 'benchmark' in df.columns else {},
        'generation_timestamp': datetime.now().isoformat(),
        'source_repo': repo_id
    }
    
    logger.info(f"📈 Basic stats: {stats['overview']['total_records']:,} records, "
                f"{stats['overview']['total_models']:,} models, "
                f"{stats['overview']['total_datasets']:,} datasets")
    
    # Model performance statistics
    if 'model_name' in df.columns and 'evaluation_score' in df.columns:
        model_stats = []
        
        for model_name, model_group in df.groupby('model_name'):
            scores = model_group['evaluation_score'].dropna()
            if len(scores) > 0:
                model_stat = {
                    'model_name': model_name,
                    'total_evaluations': len(scores),
                    'mean_score': float(scores.mean()),
                    'median_score': float(scores.median()),
                    'std_score': float(scores.std()) if len(scores) > 1 else 0.0,
                    'min_score': float(scores.min()),
                    'max_score': float(scores.max()),
                    'datasets_evaluated': model_group['dataset_name'].nunique() if 'dataset_name' in model_group.columns else 0
                }
                model_stats.append(model_stat)
        
        # Sort by mean score
        model_stats.sort(key=lambda x: x['mean_score'], reverse=True)
        stats['model_performance'] = model_stats
        logger.info(f"📊 Generated performance stats for {len(model_stats)} models")
    
    # Dataset statistics
    if 'dataset_name' in df.columns and 'evaluation_score' in df.columns:
        dataset_stats = []
        
        for dataset_name, dataset_group in df.groupby('dataset_name'):
            scores = dataset_group['evaluation_score'].dropna()
            if len(scores) > 0:
                dataset_stat = {
                    'dataset_name': dataset_name,
                    'total_evaluations': len(scores),
                    'mean_score': float(scores.mean()),
                    'median_score': float(scores.median()),
                    'std_score': float(scores.std()) if len(scores) > 1 else 0.0,
                    'min_score': float(scores.min()),
                    'max_score': float(scores.max()),
                    'models_evaluated': dataset_group['model_name'].nunique() if 'model_name' in dataset_group.columns else 0
                }
                dataset_stats.append(dataset_stat)
        
        # Sort by number of evaluations
        dataset_stats.sort(key=lambda x: x['total_evaluations'], reverse=True)
        stats['dataset_statistics'] = dataset_stats
        logger.info(f"📊 Generated dataset stats for {len(dataset_stats)} datasets")
    
    # Benchmark-specific statistics
    if 'benchmark' in df.columns:
        benchmark_stats = []
        
        for benchmark, benchmark_group in df.groupby('benchmark'):
            benchmark_stat = {
                'benchmark': benchmark,
                'total_records': len(benchmark_group),
                'unique_models': benchmark_group['model_name'].nunique() if 'model_name' in benchmark_group.columns else 0,
                'unique_datasets': benchmark_group['dataset_name'].nunique() if 'dataset_name' in benchmark_group.columns else 0,
            }
            
            if 'evaluation_score' in benchmark_group.columns:
                scores = benchmark_group['evaluation_score'].dropna()
                if len(scores) > 0:
                    benchmark_stat.update({
                        'mean_score': float(scores.mean()),
                        'median_score': float(scores.median()),
                        'std_score': float(scores.std()) if len(scores) > 1 else 0.0
                    })
            
            benchmark_stats.append(benchmark_stat)
        
        stats['benchmark_statistics'] = benchmark_stats
        logger.info(f"📊 Generated benchmark stats for {len(benchmark_stats)} benchmarks")
    
    # Top performing model-dataset combinations
    if 'model_name' in df.columns and 'dataset_name' in df.columns and 'evaluation_score' in df.columns:
        top_combinations = []
        
        for (model, dataset), group in df.groupby(['model_name', 'dataset_name']):
            scores = group['evaluation_score'].dropna()
            if len(scores) > 0:
                combination = {
                    'model_name': model,
                    'dataset_name': dataset,
                    'mean_score': float(scores.mean()),
                    'evaluations': len(scores)
                }
                top_combinations.append(combination)
        
        # Sort by mean score and take top 100
        top_combinations.sort(key=lambda x: x['mean_score'], reverse=True)
        stats['top_combinations'] = top_combinations[:100]
        logger.info(f"📊 Generated top 100 model-dataset combinations")
    
    return stats


def create_summary_dataset(stats: Dict[str, Any], stats_repo_id: str, api: HfApi) -> bool:
    """Create a summary dataset from the statistics."""
    
    try:
        # Create model performance DataFrame
        if 'model_performance' in stats:
            model_df = pd.DataFrame(stats['model_performance'])
            model_df['record_type'] = 'model_performance'
            model_df['generation_timestamp'] = stats['overview']['generation_timestamp']
        else:
            model_df = pd.DataFrame()
        
        # Create dataset statistics DataFrame  
        if 'dataset_statistics' in stats:
            dataset_df = pd.DataFrame(stats['dataset_statistics'])
            dataset_df['record_type'] = 'dataset_statistics'
            dataset_df['generation_timestamp'] = stats['overview']['generation_timestamp']
        else:
            dataset_df = pd.DataFrame()
        
        # Create top combinations DataFrame
        if 'top_combinations' in stats:
            combinations_df = pd.DataFrame(stats['top_combinations'])
            combinations_df['record_type'] = 'top_combinations'
            combinations_df['generation_timestamp'] = stats['overview']['generation_timestamp']
        else:
            combinations_df = pd.DataFrame()
        
        # Combine all DataFrames
        all_dfs = [df for df in [model_df, dataset_df, combinations_df] if not df.empty]
        
        if not all_dfs:
            logger.warning("⚠️ No statistics data to upload")
            return False
        
        combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        
        # Save to parquet
        output_file = Path("summary_statistics.parquet")
        combined_df.to_parquet(output_file, compression='snappy', index=False)
        
        logger.info(f"📤 Uploading {len(combined_df):,} summary records to {stats_repo_id}")
        
        # Upload to HuggingFace
        api.upload_file(
            path_or_fileobj=str(output_file),
            path_in_repo="statistics.parquet",
            repo_id=stats_repo_id,
            repo_type="dataset"
        )
        
        # Upload metadata as well
        metadata_file = Path("summary_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump({
                'overview': stats['overview'],
                'benchmark_statistics': stats.get('benchmark_statistics', [])
            }, f, indent=2)
        
        api.upload_file(
            path_or_fileobj=str(metadata_file),
            path_in_repo="metadata.json",
            repo_id=stats_repo_id,
            repo_type="dataset"
        )
        
        # Cleanup
        output_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
        
        logger.info(f"✅ Successfully uploaded summary statistics to {stats_repo_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create summary dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive statistics using datasets library")
    parser.add_argument("--repo-id", type=str, required=True, help="Source HuggingFace dataset repository ID")
    parser.add_argument("--stats-repo-id", type=str, required=True, help="Statistics HuggingFace dataset repository ID")
    parser.add_argument("--source-name", type=str, default="helm", help="Source name")
    
    args = parser.parse_args()
    
    # Setup HuggingFace API
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("❌ HF_TOKEN environment variable not set")
        return False
    
    try:
        api = HfApi(token=hf_token)
        
        # Create stats repo if it doesn't exist
        try:
            api.repo_info(args.stats_repo_id, repo_type="dataset")
            logger.info(f"📁 Statistics repository exists: {args.stats_repo_id}")
        except Exception:
            logger.info(f"📁 Creating statistics repository: {args.stats_repo_id}")
            api.create_repo(args.stats_repo_id, repo_type="dataset", exist_ok=True)
        
    except Exception as e:
        logger.error(f"❌ Failed to setup HuggingFace API: {e}")
        return False
    
    logger.info("🚀 Starting comprehensive statistics generation")
    logger.info(f"📊 Source: {args.repo_id}")
    logger.info(f"📤 Target: {args.stats_repo_id}")
    
    # Generate statistics
    stats = generate_comprehensive_stats(args.repo_id, args.stats_repo_id, args.source_name)
    
    if not stats:
        logger.error("❌ Failed to generate statistics")
        return False
    
    # Create and upload summary dataset
    success = create_summary_dataset(stats, args.stats_repo_id, api)
    
    if success:
        logger.info("🎉 Statistics generation completed successfully!")
        logger.info(f"📊 Processed {stats['overview']['total_records']:,} records")
        logger.info(f"🤖 {stats['overview']['total_models']:,} models analyzed")
        logger.info(f"📋 {stats['overview']['total_datasets']:,} datasets analyzed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
