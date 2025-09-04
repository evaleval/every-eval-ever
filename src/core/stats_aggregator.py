"""
Statistics aggregator for evaluation data.

Creates summary statistics per benchmark per model from detailed evaluation data.
Outputs aggregated statistics suitable for leaderboards and analysis.
"""

import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config.settings import AGGREGATED_DATA_DIR


def calculate_benchmark_stats(df: pd.DataFrame) -> Dict:
    """Calculate statistics for a benchmark-model combination."""
    total_samples = len(df)
    
    if total_samples == 0:
        return {
            'total_samples': 0,
            'accuracy': None,
            'mean_score': None,
            'std_score': None,
            'min_score': None,
            'max_score': None,
        }
    
    # Calculate accuracy (assuming scores are 0/1 for most tasks)
    scores = df['evaluation_score'].dropna()
    
    if len(scores) == 0:
        return {
            'total_samples': total_samples,
            'accuracy': None,
            'mean_score': None,
            'std_score': None,
            'min_score': None,
            'max_score': None,
        }
    
    return {
        'total_samples': total_samples,
        'accuracy': scores.mean(),  # For 0/1 scores, mean = accuracy
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'min_score': scores.min(),
        'max_score': scores.max(),
    }


def aggregate_stats_from_parquet(
    input_file: Path,
    output_file: Path,
    source_name: str = "helm"
) -> Path:
    """
    Create aggregated statistics from detailed evaluation parquet file.
    
    Args:
        input_file: Path to detailed evaluation parquet file
        output_file: Path to save aggregated statistics parquet file
        source_name: Name of the evaluation source
        
    Returns:
        Path to the created statistics file
    """
    print(f"📊 Loading detailed evaluation data from {input_file}")
    
    # Read the detailed evaluation data
    df = pd.read_parquet(input_file)
    
    print(f"📈 Loaded {len(df):,} evaluation records")
    print(f"🔍 Found columns: {list(df.columns)}")
    
    # Group by dataset_name and model_name to calculate stats
    print(f"📋 Calculating statistics per benchmark per model...")
    
    stats_records = []
    
    # Group by dataset and model
    grouped = df.groupby(['dataset_name', 'model_name'])
    
    for (dataset_name, model_name), group_df in grouped:
        # Get model family if available
        model_family = group_df['model_family'].iloc[0] if 'model_family' in group_df.columns else None
        
        # Get evaluation method if available
        eval_methods = group_df['evaluation_method_name'].unique()
        primary_eval_method = eval_methods[0] if len(eval_methods) > 0 else None
        
        # Calculate statistics
        stats = calculate_benchmark_stats(group_df)
        
        # Create statistics record
        record = {
            'source': source_name,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'model_family': model_family,
            'evaluation_method_name': primary_eval_method,
            'total_samples': stats['total_samples'],
            'accuracy': stats['accuracy'],
            'mean_score': stats['mean_score'],
            'std_score': stats['std_score'],
            'min_score': stats['min_score'],
            'max_score': stats['max_score'],
            'processed_at': datetime.now().isoformat(),
            'statistics_generated_at': datetime.now().isoformat(),
            'pipeline_stage': 'statistics_aggregation',
        }
        
        stats_records.append(record)
    
    # Create DataFrame from statistics records
    stats_df = pd.DataFrame(stats_records)
    
    print(f"📊 Generated statistics for {len(stats_df)} benchmark-model combinations")
    print(f"📊 Unique datasets: {stats_df['dataset_name'].nunique()}")
    print(f"📊 Unique models: {stats_df['model_name'].nunique()}")
    
    # Save to parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(output_file, index=False)
    
    print(f"✅ Statistics saved to {output_file}")
    
    # Show sample of the statistics
    print(f"\n📋 Sample statistics:")
    sample_df = stats_df.head(10)[['dataset_name', 'model_name', 'total_samples', 'accuracy', 'mean_score']]
    print(sample_df.to_string(index=False))
    
    return output_file


def aggregate_multiple_sources(
    input_dir: Path = AGGREGATED_DATA_DIR,
    output_file: Optional[Path] = None,
    pattern: str = "*_aggregated.parquet"
) -> Path:
    """
    Aggregate statistics from multiple source parquet files.
    
    Args:
        input_dir: Directory containing detailed evaluation parquet files
        output_file: Path to save combined statistics (optional)
        pattern: Glob pattern to match input files
        
    Returns:
        Path to the created combined statistics file
    """
    if output_file is None:
        output_file = input_dir / "evaluation_statistics.parquet"
    
    print(f"🔍 Looking for evaluation files in {input_dir} with pattern '{pattern}'")
    
    # Find all aggregated parquet files
    input_files = list(input_dir.glob(pattern))
    
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern '{pattern}' in {input_dir}")
    
    print(f"📁 Found {len(input_files)} evaluation files:")
    for file in input_files:
        print(f"   - {file.name}")
    
    all_stats = []
    
    for input_file in input_files:
        # Extract source name from filename (e.g., "helm_lite_aggregated.parquet" -> "helm")
        source_name = input_file.stem.split('_')[0]
        
        print(f"\n🔄 Processing {input_file.name} (source: {source_name})")
        
        # Create temporary stats file
        temp_stats_file = input_dir / f"temp_stats_{input_file.stem}.parquet"
        
        try:
            aggregate_stats_from_parquet(input_file, temp_stats_file, source_name)
            
            # Load the stats and add to collection
            stats_df = pd.read_parquet(temp_stats_file)
            all_stats.append(stats_df)
            
            # Clean up temp file
            temp_stats_file.unlink()
            
        except Exception as e:
            print(f"⚠️ Error processing {input_file.name}: {e}")
            if temp_stats_file.exists():
                temp_stats_file.unlink()
            continue
    
    if not all_stats:
        raise RuntimeError("No statistics could be generated from any input files")
    
    # Combine all statistics
    print(f"\n🔄 Combining statistics from {len(all_stats)} sources...")
    combined_stats = pd.concat(all_stats, ignore_index=True)
    
    # Sort by accuracy (descending) for leaderboard-style ordering
    combined_stats = combined_stats.sort_values(['dataset_name', 'accuracy'], ascending=[True, False])
    
    # Save combined statistics
    combined_stats.to_parquet(output_file, index=False)
    
    print(f"✅ Combined statistics saved to {output_file}")
    print(f"📊 Total benchmark-model combinations: {len(combined_stats)}")
    print(f"📊 Unique datasets: {combined_stats['dataset_name'].nunique()}")
    print(f"📊 Unique models: {combined_stats['model_name'].nunique()}")
    print(f"📊 Sources: {combined_stats['source'].unique().tolist()}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate evaluation statistics from detailed data")
    parser.add_argument("--input-file", type=str, help="Input detailed evaluation parquet file")
    parser.add_argument("--input-dir", type=str, default=str(AGGREGATED_DATA_DIR),
                       help=f"Input directory containing evaluation files (default: {AGGREGATED_DATA_DIR})")
    parser.add_argument("--output-file", type=str, help="Output statistics parquet file")
    parser.add_argument("--source-name", type=str, default="helm", help="Name of the evaluation source")
    parser.add_argument("--pattern", type=str, default="*_aggregated.parquet",
                       help="Glob pattern to match input files when processing directory")
    parser.add_argument("--multi-source", action="store_true",
                       help="Aggregate statistics from multiple sources in input directory")
    
    args = parser.parse_args()
    
    if args.multi_source:
        # Process multiple sources from directory
        input_dir = Path(args.input_dir)
        output_file = Path(args.output_file) if args.output_file else None
        
        aggregate_multiple_sources(
            input_dir=input_dir,
            output_file=output_file,
            pattern=args.pattern
        )
        
    elif args.input_file:
        # Process single file
        input_file = Path(args.input_file)
        output_file = Path(args.output_file) if args.output_file else input_file.parent / f"{input_file.stem}_stats.parquet"
        
        aggregate_stats_from_parquet(
            input_file=input_file,
            output_file=output_file,
            source_name=args.source_name
        )
        
    else:
        print("❌ Error: Please specify either --input-file or --multi-source")
        parser.print_help()
