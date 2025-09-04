#!/usr/bin/env python3
"""
Generate example comprehensive statistics files that would be produced by 
scripts/generate_comprehensive_stats.py for the scores repository.

Only comprehensive stats files are generated - no summary stats files are needed.
Each row represents one model-benchmark-dataset combination with full statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def generate_comprehensive_stats_from_chunks():
    """Generate comprehensive statistics from the actual chunk files."""
    
    # Read the example data files
    examples_dir = Path(__file__).parent / "examples"
    parquet_files = list(examples_dir.glob("chunk_*.parquet"))
    
    if not parquet_files:
        print("❌ No chunk parquet files found to process")
        return False
    
    print(f"📊 Processing {len(parquet_files)} chunk files to generate comprehensive stats")
    
    # Combine all data
    all_data = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        all_data.append(df)
        print(f"  📄 Loaded {file.name}: {len(df)} rows")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Combined dataset: {len(combined_df)} total rows")
    
    # Generate comprehensive statistics - one row per model-benchmark-dataset combination
    stats_rows = []
    
    # Group by the key dimensions: source, benchmark, dataset, model
    grouped = combined_df.groupby(['source', 'benchmark', 'dataset_name', 'model_name'])
    
    for (source, benchmark, dataset_name, model_name), group in grouped:
        # Calculate statistics for this specific combination
        scores = group['evaluation_score'].dropna()
        
        if len(scores) == 0:
            continue;
            
        # Extract model family
        model_family = group['model_family'].iloc[0] if 'model_family' in group.columns else model_name.split('/')[0] if '/' in model_name else 'unknown'
        
        # Calculate comprehensive statistics
        stats_row = {
            'source': source,
            'benchmark': benchmark, 
            'dataset_name': dataset_name,
            'model_name': model_name,
            'model_family': model_family,
            'total_samples': len(scores),
            'accuracy': scores.mean(),
            'mean_score': scores.mean(),  # Same as accuracy for most cases
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'median_score': scores.median(),
            'q25_score': scores.quantile(0.25),
            'q75_score': scores.quantile(0.75),
            'processing_date': group['processing_date'].iloc[0],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'stats_version': '1.0'
        }
        
        stats_rows.append(stats_row)
    
    # Create comprehensive stats DataFrame
    comprehensive_stats = pd.DataFrame(stats_rows)
    
    # Sort by benchmark, dataset, then by accuracy descending
    comprehensive_stats = comprehensive_stats.sort_values([
        'benchmark', 'dataset_name', 'accuracy'
    ], ascending=[True, True, False])
    
    print(f"📊 Generated comprehensive stats: {len(comprehensive_stats)} model-benchmark-dataset combinations")
    
    # Create the output file with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d')
    output_file = examples_dir / f"comprehensive_stats_from_chunks_{timestamp}.parquet"
    
    comprehensive_stats.to_parquet(output_file, index=False)
    
    print(f"✅ Created comprehensive stats file: {output_file.name}")
    print(f"📊 File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Show sample of the comprehensive stats
    print(f"\n📋 Sample comprehensive statistics:")
    display_cols = ['benchmark', 'dataset_name', 'model_name', 'total_samples', 'accuracy']
    print(comprehensive_stats[display_cols].head(10).to_string(index=False))
    
    # Show summary by benchmark
    print(f"\n📊 Summary by benchmark:")
    if len(comprehensive_stats) > 0:
        benchmark_summary = comprehensive_stats.groupby('benchmark').agg({
            'dataset_name': 'nunique',
            'model_name': 'nunique', 
            'total_samples': 'sum',
            'accuracy': 'mean'
        }).round(3)
        benchmark_summary.columns = ['unique_datasets', 'unique_models', 'total_samples', 'avg_accuracy']
        print(benchmark_summary.to_string())
    
    return True;


def generate_comprehensive_stats_example():
    """Generate example comprehensive statistics parquet file.
    
    Only comprehensive stats are generated - one row per model-benchmark-dataset 
    combination. This is what gets uploaded to the scores repository.
    """
    
    # Comprehensive statistics data - one row per model-benchmark-dataset combination
    # This reflects what would actually be uploaded to the scores repository
    comprehensive_data = [
        # OpenBookQA + lite benchmark (from our real chunks)
        ("helm", "lite", "openbookqa", "01-ai/yi-34b", "01-ai", 500, 0.920, 0.271, 0.0, 1.0, 1.0, 1.0, 1.0),
        ("helm", "lite", "openbookqa", "01-ai/yi-6b", "01-ai", 500, 0.800, 0.400, 0.0, 1.0, 1.0, 1.0, 1.0),
        ("helm", "lite", "openbookqa", "AlephAlpha/luminous-base", "AlephAlpha", 500, 0.286, 0.452, 0.0, 1.0, 0.0, 0.0, 1.0),
        ("helm", "lite", "openbookqa", "AlephAlpha/luminous-extended", "AlephAlpha", 500, 0.272, 0.445, 0.0, 1.0, 0.0, 0.0, 1.0),
        ("helm", "lite", "openbookqa", "anthropic/claude-v1.3", "Anthropic", 500, 0.908, 0.289, 0.0, 1.0, 1.0, 1.0, 1.0),
        
        # Additional synthetic examples for MMLU
        ("helm", "mmlu", "abstract_algebra", "01-ai/yi-34b", "01-ai", 100, 0.74, 0.439, 0.0, 1.0, 1.0, 1.0, 1.0),
        ("helm", "mmlu", "abstract_algebra", "anthropic/claude-v1.3", "Anthropic", 100, 0.83, 0.376, 0.0, 1.0, 1.0, 1.0, 1.0),
        ("helm", "mmlu", "abstract_algebra", "openai/gpt-4", "OpenAI", 100, 0.86, 0.347, 0.0, 1.0, 1.0, 1.0, 1.0),
        ("helm", "mmlu", "anatomy", "01-ai/yi-34b", "01-ai", 135, 0.71, 0.455, 0.0, 1.0, 1.0, 1.0, 1.0),
        ("helm", "mmlu", "anatomy", "anthropic/claude-v1.3", "Anthropic", 135, 0.78, 0.415, 0.0, 1.0, 1.0, 1.0, 1.0),
        ("helm", "mmlu", "anatomy", "openai/gpt-4", "OpenAI", 135, 0.85, 0.358, 0.0, 1.0, 1.0, 1.0, 1.0),
        
        # HellaSwag examples
        ("helm", "classic", "hellaswag", "01-ai/yi-34b", "01-ai", 1000, 0.867, 0.340, 0.0, 1.0, 1.0, 1.0, 1.0),
        ("helm", "classic", "hellaswag", "anthropic/claude-v1.3", "Anthropic", 1000, 0.892, 0.310, 0.0, 1.0, 1.0, 1.0, 1.0),
        ("helm", "classic", "hellaswag", "openai/gpt-4", "OpenAI", 1000, 0.923, 0.266, 0.0, 1.0, 1.0, 1.0, 1.0),
    ]
    
    # Create comprehensive statistics DataFrame
    stats_data = []
    base_time = datetime.now()
    
    for i, (source, benchmark, dataset, model, family, samples, accuracy, std, min_s, max_s, q25, median, q75) in enumerate(comprehensive_data):
        
        stats_data.append({
            'source': source,
            'benchmark': benchmark,
            'dataset_name': dataset,
            'model_name': model,
            'model_family': family,
            'total_samples': samples,
            'accuracy': accuracy,
            'mean_score': accuracy,  # For classification tasks, these are the same
            'std_score': std,
            'min_score': min_s,
            'max_score': max_s,
            'median_score': median,
            'q25_score': q25,
            'q75_score': q75,
            'processing_date': (base_time - timedelta(hours=i)).strftime('%Y-%m-%d'),
            'last_updated': base_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'stats_version': '1.0'
        })
    
    # Create DataFrame and sort by benchmark, dataset, then accuracy
    comprehensive_df = pd.DataFrame(stats_data)
    comprehensive_df = comprehensive_df.sort_values(['benchmark', 'dataset_name', 'accuracy'], ascending=[True, True, False])
    
    # Generate realistic timestamp for filename
    timestamp = base_time.strftime("%Y-%m-%d")
    
    # Save comprehensive stats file (as would be uploaded to scores repo)
    examples_dir = Path(__file__).parent / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    comprehensive_file = examples_dir / f"comprehensive_stats_{timestamp}.parquet"
    comprehensive_df.to_parquet(comprehensive_file, index=False)
    
    print(f"✅ Generated comprehensive statistics file: {comprehensive_file.name}")
    print(f"📊 File size: {comprehensive_file.stat().st_size / 1024:.1f} KB")
    print(f"📋 Contains {len(comprehensive_df)} benchmark-model combinations")
    print(f"🎯 {comprehensive_df['dataset_name'].nunique()} unique benchmarks")
    print(f"🤖 {comprehensive_df['model_name'].nunique()} unique models")
    print(f"📈 {comprehensive_df['total_samples'].sum():,} total evaluation samples")
    print(f"📤 When uploaded to scores repo, this would become: helm-00001.parquet (smart indexing)")
    
    # Show top performers by benchmark
    print(f"\n🏆 Top performers by benchmark:")
    for dataset in comprehensive_df['dataset_name'].unique():
        dataset_df = comprehensive_df[comprehensive_df['dataset_name'] == dataset]
        top_model = dataset_df.iloc[0]  # Already sorted by accuracy desc
        print(f"  📊 {dataset}: {top_model['model_name']} ({top_model['accuracy']*100:.1f}%)")
    
    return comprehensive_file

def show_file_structures(comprehensive_file):
    """Show the structure of generated files."""
    
    print(f"\n" + "="*80)
    print("📋 COMPREHENSIVE STATS FILE STRUCTURE")
    print("="*80)
    
    df = pd.read_parquet(comprehensive_file)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns:")
    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].iloc[0] if len(df) > 0 else "N/A"
        print(f"  • {col}: {dtype} (e.g., '{sample}')")
    
    print(f"\nSample rows:")
    key_cols = ['dataset_name', 'model_name', 'accuracy', 'total_samples', 'std_score']
    print(df[key_cols].head(3).to_string(index=False))

if __name__ == "__main__":
    comprehensive_file = generate_comprehensive_stats_example()
    show_file_structures(comprehensive_file)
    
    print(f"\n🎉 Example comprehensive statistics file generated!")
    print(f"📁 File saved in: tests/examples/")
    print(f"💡 This represents the type of file uploaded to evaleval/every_eval_score_ever")
    print(f"📋 Upload pattern: One file per source with smart indexing (e.g., helm-00001.parquet)")
    print(f"🔄 Smart indexing ensures no conflicts and maintains chronological order")
