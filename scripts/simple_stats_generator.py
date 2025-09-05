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
    
    # Main statistics: per benchmark+dataset+model combination
    detailed_stats = []
    
    if all(col in df.columns for col in ['benchmark', 'dataset_name', 'model_name', 'evaluation_score']):
        logger.info("📊 Generating detailed benchmark+dataset+model statistics...")
        
        # Group by benchmark, dataset, and model
        grouped = df.groupby(['benchmark', 'dataset_name', 'model_name'])
        
        for (benchmark, dataset_name, model_name), group in grouped:
            scores = group['evaluation_score'].dropna()
            
            if len(scores) > 0:
                # Calculate statistics for this specific combination
                stat_record = {
                    'benchmark': benchmark,
                    'dataset_name': dataset_name,
                    'model_name': model_name,
                    'evaluation_count': len(scores),
                    'mean_score': float(scores.mean()),
                    'median_score': float(scores.median()),
                    'std_score': float(scores.std()) if len(scores) > 1 else 0.0,
                    'min_score': float(scores.min()),
                    'max_score': float(scores.max()),
                    'first_evaluation': group['timestamp'].min() if 'timestamp' in group.columns else None,
                    'last_evaluation': group['timestamp'].max() if 'timestamp' in group.columns else None,
                    'source': group['source'].iloc[0] if 'source' in group.columns else source_name,
                    'unique_tasks': group['task_id'].nunique() if 'task_id' in group.columns else len(group),
                    'generation_timestamp': datetime.now().isoformat()
                }
                
                # Add percentiles if we have enough data
                if len(scores) >= 4:
                    stat_record['p25_score'] = float(scores.quantile(0.25))
                    stat_record['p75_score'] = float(scores.quantile(0.75))
                else:
                    stat_record['p25_score'] = stat_record['min_score']
                    stat_record['p75_score'] = stat_record['max_score']
                
                detailed_stats.append(stat_record)
        
        logger.info(f"📊 Generated {len(detailed_stats)} benchmark+dataset+model combinations")
    else:
        logger.warning("⚠️ Missing required columns for detailed statistics")
    
    stats['detailed_performance'] = detailed_stats
    
    return stats


def create_summary_dataset(stats: Dict[str, Any], stats_repo_id: str, api: HfApi) -> bool:
    """Create a summary dataset from the statistics."""
    
    try:
        # Create the main detailed performance DataFrame
        if 'detailed_performance' in stats and stats['detailed_performance']:
            detailed_df = pd.DataFrame(stats['detailed_performance'])
            logger.info(f"📊 Created detailed performance DataFrame with {len(detailed_df):,} rows")
        else:
            logger.warning("⚠️ No detailed performance data to upload")
            return False
        
        # Save to parquet
        output_file = Path("detailed_statistics.parquet")
        detailed_df.to_parquet(output_file, compression='snappy', index=False)
        
        logger.info(f"📤 Uploading {len(detailed_df):,} detailed performance records to {stats_repo_id}")
        
        # Upload to HuggingFace
        api.upload_file(
            path_or_fileobj=str(output_file),
            path_in_repo="detailed_statistics.parquet",
            repo_id=stats_repo_id,
            repo_type="dataset"
        )
        
        # Create and upload overview metadata
        metadata = {
            'overview': stats['overview'],
            'detailed_stats_summary': {
                'total_combinations': len(detailed_df),
                'benchmarks': sorted(detailed_df['benchmark'].unique().tolist()) if 'benchmark' in detailed_df.columns else [],
                'models': detailed_df['model_name'].nunique() if 'model_name' in detailed_df.columns else 0,
                'datasets': detailed_df['dataset_name'].nunique() if 'dataset_name' in detailed_df.columns else 0,
                'top_performing_combinations': detailed_df.nlargest(10, 'mean_score')[
                    ['benchmark', 'dataset_name', 'model_name', 'mean_score']
                ].to_dict('records') if len(detailed_df) > 0 else [],
                'worst_performing_combinations': detailed_df.nsmallest(10, 'mean_score')[
                    ['benchmark', 'dataset_name', 'model_name', 'mean_score']
                ].to_dict('records') if len(detailed_df) > 0 else []
            }
        }
        
        metadata_file = Path("detailed_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        api.upload_file(
            path_or_fileobj=str(metadata_file),
            path_in_repo="detailed_metadata.json",
            repo_id=stats_repo_id,
            repo_type="dataset"
        )
        
        # Also create a README for the stats repository
        readme_content = f"""# Evaluation Statistics Dataset

This dataset contains detailed performance statistics for AI model evaluations.

## Structure

### Main Data File: `detailed_statistics.parquet`
Each row represents a unique combination of:
- **benchmark**: The evaluation benchmark (lite, mmlu, classic)
- **dataset_name**: The specific dataset within the benchmark
- **model_name**: The AI model being evaluated

### Columns:
- `benchmark`: Evaluation benchmark name
- `dataset_name`: Dataset name
- `model_name`: Model name  
- `evaluation_count`: Number of evaluations for this combination
- `mean_score`: Average evaluation score
- `median_score`: Median evaluation score
- `std_score`: Standard deviation of scores
- `min_score`: Minimum score
- `max_score`: Maximum score
- `p25_score`: 25th percentile score
- `p75_score`: 75th percentile score
- `first_evaluation`: Timestamp of first evaluation
- `last_evaluation`: Timestamp of most recent evaluation
- `source`: Data source (e.g., "helm")
- `unique_tasks`: Number of unique tasks evaluated
- `generation_timestamp`: When this statistics record was generated

## Summary
- **Total combinations**: {metadata['detailed_stats_summary']['total_combinations']:,}
- **Unique models**: {metadata['detailed_stats_summary']['models']:,}
- **Unique datasets**: {metadata['detailed_stats_summary']['datasets']:,}
- **Benchmarks**: {', '.join(metadata['detailed_stats_summary']['benchmarks'])}
- **Generated**: {metadata['overview']['generation_timestamp']}

## Usage Example

```python
from datasets import load_dataset
import pandas as pd

# Load the statistics dataset
dataset = load_dataset("evaleval/every_eval_score_ever")
stats_df = pd.DataFrame(dataset['train'])

# Find top performing models on a specific dataset
mmlu_results = stats_df[
    (stats_df['benchmark'] == 'mmlu') & 
    (stats_df['dataset_name'] == 'some_dataset')
].sort_values('mean_score', ascending=False)

# Compare model performance across benchmarks
model_comparison = stats_df[
    stats_df['model_name'] == 'some_model'
].groupby('benchmark')['mean_score'].mean()
```

Updated: {datetime.now().isoformat()}
"""
        
        readme_file = Path("README.md")
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj=str(readme_file),
            path_in_repo="README.md",
            repo_id=stats_repo_id,
            repo_type="dataset"
        )
        
        # Cleanup
        output_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
        readme_file.unlink(missing_ok=True)
        
        logger.info(f"✅ Successfully uploaded detailed statistics to {stats_repo_id}")
        logger.info(f"📊 {len(detailed_df):,} benchmark+dataset+model combinations")
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
