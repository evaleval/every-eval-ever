#!/usr/bin/env python3
"""
Quick test to generate example parquet files to examine their structure.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from the test file
from tests.test_end_to_end import create_test_csv, mock_hf_api
from scripts.optimized_helm_processor import process_with_optimization
from unittest.mock import patch
import argparse

def generate_example_files():
    """Generate example parquet files without cleanup."""
    print("🧪 Generating example parquet files...")
    
    # Clean up any existing files first
    import shutil
    for cleanup_dir in ["data/benchmark_lines", "data/processed", "data/aggregated", "data/downloads"]:
        cleanup_path = Path(cleanup_dir)
        if cleanup_path.exists():
            shutil.rmtree(cleanup_path)
            print(f"🧹 Cleaned up: {cleanup_dir}")
    
    # Create test CSV
    csv_file = create_test_csv()
    print(f"✅ Created test CSV: {csv_file}")
    
    # Mock HF operations and environment
    mock_api = mock_hf_api()
    
    # Set fake HF token
    os.environ['HF_TOKEN'] = 'fake_token_for_testing'
    
    with patch('scripts.optimized_helm_processor.setup_hf_api', return_value=mock_api):
        with patch('scripts.optimized_helm_processor.get_existing_files', return_value=[]):
            # Run processor with no cleanup to preserve files
            args = argparse.Namespace(
                benchmark='lite',
                csv_file=str(csv_file),
                chunk_size=3,
                max_workers=1,
                upload_workers=1,
                timeout=5,
                repo_id='test/repo',
                source_name='helm',
                adapter_method=None,
                no_cleanup=True  # Preserve files
            )
            
            success = process_with_optimization(args)
    
    if success:
        print("✅ Generated example files successfully!")
        
        # List generated parquet files
        agg_dir = Path("data/aggregated")
        if agg_dir.exists():
            parquet_files = list(agg_dir.glob("*.parquet"))
            print(f"\n📊 Generated {len(parquet_files)} parquet files:")
            for file in parquet_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  📄 {file.name}: {size_mb:.2f}MB")
        
        return True
    else:
        print("❌ Failed to generate example files")
        return False

if __name__ == "__main__":
    generate_example_files()
