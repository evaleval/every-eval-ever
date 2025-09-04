#!/usr/bin/env python3
"""
Demo script showing how the comprehensive stats generator would work.
This simulates the HuggingFace integration without requiring tokens.
"""

import logging
from pathlib import Path
import sys
sys.path.append('scripts')
from generate_comprehensive_stats import IncrementalStatsCalculator
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_comprehensive_stats():
    """Demonstrate the comprehensive statistics generation process."""
    
    logger.info("🚀 Demo: Comprehensive Statistics Generation")
    logger.info("=" * 60)
    
    # Simulate multiple shards (we'll use the same file split into chunks)
    test_file = Path("data/aggregated/helm_lite_aggregated.parquet")
    
    if not test_file.exists():
        logger.error("❌ Test data not found. Run HELM processing first.")
        return False
    
    # Load data and split into "shards"
    full_data = pd.read_parquet(test_file)
    logger.info(f"📊 Simulating processing of data: {len(full_data):,} total records")
    
    # Split into chunks to simulate multiple uploaded shards
    chunk_size = 15000
    shards = [full_data[i:i+chunk_size] for i in range(0, len(full_data), chunk_size)]
    
    logger.info(f"📁 Simulating {len(shards)} shards (would be downloaded from HuggingFace):")
    for i, shard in enumerate(shards):
        logger.info(f"   📄 helm_lite_part_{i+1:03d}.parquet: {len(shard):,} records")
    
    # Initialize incremental calculator
    calculator = IncrementalStatsCalculator()
    
    logger.info("")
    logger.info("🔄 Processing shards incrementally...")
    
    # Process each "shard" 
    for i, shard in enumerate(shards):
        logger.info(f"⬇️ Processing shard {i+1}/{len(shards)}: helm_lite_part_{i+1:03d}.parquet")
        calculator.update_with_shard(shard, "helm")
        
        # Show memory usage would be constant
        current_combinations = len(calculator.running_stats)
        logger.info(f"   📊 Running totals: {current_combinations} combinations in memory")
    
    # Finalize comprehensive statistics
    logger.info("")
    logger.info("📊 Finalizing comprehensive statistics...")
    comprehensive_stats = calculator.finalize_statistics()
    
    # Show results
    logger.info("")
    logger.info("📈 COMPREHENSIVE STATISTICS RESULTS")
    logger.info(f"   📊 Total combinations: {len(comprehensive_stats)}")
    logger.info(f"   🎯 Unique benchmarks: {comprehensive_stats['dataset_name'].nunique()}")
    logger.info(f"   🤖 Unique models: {comprehensive_stats['model_name'].nunique()}")
    logger.info(f"   📈 Total evaluation records processed: {calculator.total_records:,}")
    
    # Show top performers
    logger.info("")
    logger.info("🏆 TOP 10 PERFORMERS (COMPREHENSIVE LEADERBOARD)")
    top_10 = comprehensive_stats.nlargest(10, 'accuracy')
    
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        logger.info(f"   {i:2d}. {row['model_name']:<25} on {row['dataset_name']:<15}: {row['accuracy']*100:5.1f}% ({row['total_samples']:,} samples)")
    
    # Show benchmark difficulty analysis
    logger.info("")
    logger.info("📊 BENCHMARK DIFFICULTY ANALYSIS")
    benchmark_difficulty = comprehensive_stats.groupby('dataset_name')['accuracy'].agg(['mean', 'std', 'count']).round(4)
    benchmark_difficulty = benchmark_difficulty.sort_values('mean', ascending=False)
    
    logger.info("   Average accuracy across all models:")
    for benchmark, stats in benchmark_difficulty.iterrows():
        logger.info(f"   {benchmark:<15}: {stats['mean']*100:5.1f}% ± {stats['std']*100:4.1f}% ({stats['count']} models)")
    
    # Show memory efficiency
    logger.info("")
    logger.info("💡 MEMORY EFFICIENCY DEMONSTRATION")
    logger.info(f"   📊 Traditional approach: Load all {len(full_data):,} records = ~{len(full_data)*200/1024/1024:.1f} MB")
    logger.info(f"   🧠 Incremental approach: Max {chunk_size:,} records = ~{chunk_size*200/1024/1024:.1f} MB")
    logger.info(f"   ⚡ Memory reduction: {len(full_data)/chunk_size:.1f}x less memory usage")
    logger.info(f"   🔄 Final statistics: {len(comprehensive_stats)} rows = ~{len(comprehensive_stats)*100/1024:.1f} KB")
    
    # Simulate upload
    logger.info("")
    logger.info("☁️ SIMULATED UPLOAD")
    output_file = Path("data/aggregated/comprehensive_stats_helm_demo.parquet")
    comprehensive_stats.to_parquet(output_file, index=False)
    file_size = output_file.stat().st_size / (1024 * 1024)
    
    logger.info(f"   📁 Saved: {output_file.name}")
    logger.info(f"   📏 Size: {file_size:.2f} MB")
    logger.info(f"   ☁️ Would upload to: evaleval/every_eval_score_ever")
    logger.info(f"   🧹 Would clean up temporary files")
    
    logger.info("")
    logger.info("🎉 Demo complete! This process would:")
    logger.info("   ✅ Handle datasets of any size without memory issues")
    logger.info("   ✅ Process shards incrementally as they're uploaded")
    logger.info("   ✅ Generate a single comprehensive leaderboard")
    logger.info("   ✅ Update statistics automatically after each upload batch")
    
    return True

if __name__ == "__main__":
    success = demo_comprehensive_stats()
    sys.exit(0 if success else 1)
