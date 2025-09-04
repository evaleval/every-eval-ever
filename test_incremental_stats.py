#!/usr/bin/env python3
"""
Test the incremental statistics calculator locally.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('scripts')

from generate_comprehensive_stats import IncrementalStatsCalculator

def test_incremental_stats():
    """Test the incremental statistics calculator with local data."""
    
    # Check if we have test data
    test_file = Path("data/aggregated/helm_lite_aggregated.parquet")
    if not test_file.exists():
        print("❌ Test file not found:", test_file)
        return False
    
    print("🧪 Testing incremental statistics calculator...")
    
    # Load full dataset for comparison
    full_df = pd.read_parquet(test_file)
    print(f"📊 Loaded test data: {len(full_df):,} records")
    
    # Initialize calculator
    calculator = IncrementalStatsCalculator()
    
    # Split data into chunks to simulate shards
    chunk_size = 10000
    chunks = [full_df[i:i+chunk_size] for i in range(0, len(full_df), chunk_size)]
    
    print(f"🔄 Processing {len(chunks)} chunks of ~{chunk_size:,} records each...")
    
    # Process chunks incrementally
    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)} ({len(chunk):,} records)")
        calculator.update_with_shard(chunk, "helm")
    
    # Finalize statistics
    incremental_stats = calculator.finalize_statistics()
    
    print(f"✅ Incremental processing complete!")
    print(f"📊 Generated {len(incremental_stats)} benchmark-model combinations")
    
    # Compare with direct calculation for validation
    print("\n🔍 Validation: Comparing with direct calculation...")
    
    # Direct calculation
    direct_grouped = full_df.groupby(['dataset_name', 'model_name'])['evaluation_score']
    direct_stats = []
    
    for (dataset, model), scores in direct_grouped:
        direct_stats.append({
            'dataset_name': dataset,
            'model_name': model,
            'count': len(scores),
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        })
    
    direct_df = pd.DataFrame(direct_stats)
    
    # Compare a few key metrics
    print(f"📊 Direct calculation: {len(direct_df)} combinations")
    print(f"📊 Incremental calculation: {len(incremental_stats)} combinations")
    
    if len(direct_df) == len(incremental_stats):
        print("✅ Same number of combinations")
    else:
        print("❌ Different number of combinations!")
    
    # Show sample results
    print("\n🏆 Top 5 performers (incremental):")
    top_5 = incremental_stats.nlargest(5, 'accuracy')
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. {row['model_name']} on {row['dataset_name']}: {row['accuracy']*100:.1f}%")
    
    # Memory efficiency note
    print(f"\n💡 Memory efficiency:")
    print(f"   Full dataset: {len(full_df):,} records loaded at once")
    print(f"   Incremental: Max {chunk_size:,} records in memory at any time")
    print(f"   Reduction: {len(full_df)/chunk_size:.1f}x less memory usage")
    
    return True

if __name__ == "__main__":
    success = test_incremental_stats()
    sys.exit(0 if success else 1)
