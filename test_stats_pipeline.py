#!/usr/bin/env python3
"""
Test script to verify the stats generation pipeline without uploading.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_stats_pipeline():
    """Test the complete pipeline: download -> process -> aggregate -> stats"""
    
    # Test benchmark
    benchmark = "lite"
    source_name = "helm"
    part_num = 999  # Test part number
    
    # Output paths
    aggregated_file = Path(f"data/aggregated/{source_name}_{benchmark}_part_{part_num:03d}.parquet")
    stats_file = Path(f"data/aggregated/stats-{source_name}_{benchmark}_part_{part_num:03d}.parquet")
    
    try:
        logger.info(f"🧪 Testing complete stats pipeline for {benchmark}")
        
        # Step 1: Process/download HELM data
        logger.info("📥 Step 1 - Processing HELM data...")
        process_cmd = [
            sys.executable, "-m", "src.sources.helm.processor",
            "--benchmark", benchmark,
            "--max-workers", "1"
        ]
        
        logger.info(f"Running: {' '.join(process_cmd)}")
        result = subprocess.run(process_cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"❌ Processing failed: {result.stderr}")
            return False
        
        logger.info("✅ HELM processing completed")
        
        # Step 2: Aggregate the data
        logger.info("🔄 Step 2 - Aggregating data...")
        agg_cmd = [
            sys.executable, "-m", "src.core.aggregator",
            "--source", source_name,
            "--benchmark", benchmark,
            "--output-file", str(aggregated_file)
        ]
        
        logger.info(f"Running: {' '.join(agg_cmd)}")
        result = subprocess.run(agg_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"❌ Aggregation failed: {result.stderr}")
            return False
        
        # Check if the aggregator created the file with different naming
        expected_agg_file = Path(f"data/aggregated/{source_name}_{benchmark}_aggregated.parquet")
        if expected_agg_file.exists():
            expected_agg_file.rename(aggregated_file)
            logger.info(f"✅ Renamed {expected_agg_file.name} to {aggregated_file.name}")
        
        if not aggregated_file.exists():
            logger.error(f"❌ Aggregated file not found: {aggregated_file}")
            return False
        
        logger.info(f"✅ Aggregation completed: {aggregated_file}")
        
        # Step 3: Generate statistics
        logger.info("📊 Step 3 - Generating statistics...")
        stats_cmd = [
            sys.executable, "-m", "src.core.stats_aggregator",
            "--input-file", str(aggregated_file),
            "--output-file", str(stats_file),
            "--source-name", source_name
        ]
        
        logger.info(f"Running: {' '.join(stats_cmd)}")
        result = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f"❌ Stats generation failed: {result.stderr}")
            return False
        
        if not stats_file.exists():
            logger.error(f"❌ Stats file not found: {stats_file}")
            return False
        
        logger.info(f"✅ Statistics generated: {stats_file}")
        
        # Show file sizes
        agg_size = aggregated_file.stat().st_size / (1024 * 1024)
        stats_size = stats_file.stat().st_size / (1024 * 1024)
        
        logger.info(f"📁 File sizes:")
        logger.info(f"   Detailed data: {aggregated_file.name} ({agg_size:.2f} MB)")
        logger.info(f"   Statistics: {stats_file.name} ({stats_size:.2f} MB)")
        
        # Quick peek at the stats
        import pandas as pd
        stats_df = pd.read_parquet(stats_file)
        logger.info(f"📊 Statistics summary:")
        logger.info(f"   Total benchmark-model combinations: {len(stats_df)}")
        logger.info(f"   Columns: {list(stats_df.columns)}")
        
        if len(stats_df) > 0:
            # Show top performers
            top_3 = stats_df.nlargest(3, 'accuracy')
            logger.info(f"   Top 3 performers:")
            for _, row in top_3.iterrows():
                logger.info(f"     {row['model_name']} on {row['benchmark']}: {row['accuracy']:.1f}%")
        
        logger.info("🎉 Complete pipeline test successful!")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("⏰ Pipeline test timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_stats_pipeline()
    sys.exit(0 if success else 1)
