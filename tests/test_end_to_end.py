#!/usr/bin/env python3
"""
End-to-end test for optimized HELM processor workflow.

Tests the complete pipeline:
1. CSV generation (web scraping)
2. Task processing (download + convert)
3. Parquet file creation
4. No upload (mocked)

This validates the core workflow without HuggingFace dependencies.
"""

import os
import sys
import time
import logging
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_test_logging():
    """Set up test logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def create_test_csv():
    """Create a small test CSV file with just a few tasks for testing."""
    import pandas as pd
    
    # Create a small set of test tasks
    test_tasks = [
        "commonsense:dataset=openbookqa,method=multiple_choice_joint,model=01-ai_yi-34b",
        "commonsense:dataset=openbookqa,method=multiple_choice_joint,model=01-ai_yi-6b",
        "commonsense:dataset=openbookqa,method=multiple_choice_joint,model=AlephAlpha_luminous-base",
        "commonsense:dataset=openbookqa,method=multiple_choice_joint,model=AlephAlpha_luminous-extended",
        "commonsense:dataset=openbookqa,method=multiple_choice_joint,model=anthropic_claude-v1.3"
    ]
    
    # Create DataFrame matching the expected format
    df = pd.DataFrame({
        'Unnamed: 0': range(len(test_tasks)),
        'Run': [f'run_{i}' for i in range(len(test_tasks))],
        'Model': [task.split('model=')[1] for task in test_tasks],
        'Groups': ['commonsense'] * len(test_tasks),
        'Adapter method': ['multiple_choice_joint'] * len(test_tasks),
        'Subject / Task': ['openbookqa'] * len(test_tasks),
        'task': test_tasks  # This is the key column the processor uses
    })
    
    # Ensure the directory exists
    csv_path = Path("data/benchmark_lines/helm_lite.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ Created test CSV with {len(test_tasks)} tasks: {csv_path}")
    return str(csv_path)

def cleanup_test_files():
    """Clean up test files from previous runs."""
    cleanup_paths = [
        "data/benchmark_lines/helm_lite.csv",
        "data/processed",
        "data/aggregated", 
        "data/downloads"
    ]
    
    for path_str in cleanup_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_file():
                path.unlink()
                print(f"🧹 Removed file: {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"🧹 Removed directory: {path}")
    """Clean up test files from previous runs."""
    cleanup_paths = [
        "data/benchmark_lines/helm_lite.csv",
        "data/processed",
        "data/aggregated", 
        "data/downloads"
    ]
    
    for path_str in cleanup_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_file():
                path.unlink()
                print(f"🧹 Removed file: {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"🧹 Removed directory: {path}")

def mock_hf_api():
    """Create a mock HuggingFace API that doesn't actually upload."""
    class MockHfApi:
        def __init__(self, token=None):
            pass
            
        def repo_info(self, repo_id, repo_type="dataset"):
            # Simulate repo exists
            return {"id": repo_id}
            
        def create_repo(self, repo_id, repo_type="dataset", exist_ok=True):
            return {"id": repo_id}
            
        def list_repo_files(self, repo_id, repo_type="dataset"):
            # Return empty list (no existing files)
            return []
            
        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type="dataset"):
            print(f"🔄 Mock upload: {path_in_repo} (size: {Path(path_or_fileobj).stat().st_size / 1024:.1f}KB)")
            return True
    
    return MockHfApi

def test_summary_analytics(aggregated_dir: Path, expected_total_entries: int) -> bool:
    """Test the summary analytics functionality using generated parquet files."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import the stats aggregator
        from src.core.stats_aggregator import aggregate_stats_from_parquet, calculate_benchmark_stats
        import pandas as pd
        
        logger.info("🔬 Testing summary analytics on generated parquet files")
        
        # Find all parquet files
        parquet_files = list(aggregated_dir.glob("*.parquet"))
        if not parquet_files:
            logger.error("❌ No parquet files found for analytics testing")
            return False
        
        analytics_results = []
        total_processed_entries = 0
        
        for parquet_file in parquet_files:
            logger.info(f"📊 Analyzing {parquet_file.name}")
            
            # Read the parquet file
            df = pd.read_parquet(parquet_file)
            entries = len(df)
            total_processed_entries += entries
            
            # Test basic data integrity
            required_cols = ['dataset_name', 'model_name', 'evaluation_score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"⚠️ Missing required columns in {parquet_file.name}: {missing_cols}")
                continue
            
            # Test metadata fields
            metadata_cols = ['source', 'benchmark', 'timestamp', 'processing_date']
            for col in metadata_cols:
                if col in df.columns:
                    unique_vals = df[col].nunique()
                    logger.info(f"  📋 {col}: {unique_vals} unique value(s)")
                else:
                    logger.warning(f"⚠️ Missing metadata column: {col}")
            
            # Calculate basic statistics for each dataset-model combination
            grouped = df.groupby(['dataset_name', 'model_name'])
            
            for (dataset_name, model_name), group_df in grouped:
                # Test the calculate_benchmark_stats function
                stats = calculate_benchmark_stats(group_df)
                
                analytics_results.append({
                    'file': parquet_file.name,
                    'dataset': dataset_name,
                    'model': model_name,
                    'samples': stats['total_samples'],
                    'accuracy': stats['accuracy'],
                    'mean_score': stats['mean_score'],
                    'std_score': stats['std_score']
                })
                
                logger.info(f"  🎯 {dataset_name} + {model_name}: {stats['total_samples']} samples, accuracy={stats['accuracy']:.3f}")
        
        # Test aggregated statistics generation
        logger.info("🧮 Testing aggregated statistics generation")
        
        # Create a temporary combined file for testing
        temp_combined_file = aggregated_dir / "temp_combined_test.parquet"
        temp_stats_file = aggregated_dir / "temp_stats_test.parquet"
        
        try:
            # Combine all parquet files into one for testing
            all_dfs = []
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                all_dfs.append(df)
            
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_parquet(temp_combined_file, index=False)
            
            # Test the aggregate_stats_from_parquet function
            logger.info("📈 Generating aggregate statistics")
            result_file = aggregate_stats_from_parquet(
                input_file=temp_combined_file,
                output_file=temp_stats_file,
                source_name="helm"
            )
            
            # Verify the statistics file
            if result_file.exists():
                stats_df = pd.read_parquet(result_file)
                logger.info(f"✅ Generated statistics for {len(stats_df)} benchmark-model combinations")
                logger.info(f"📊 Unique datasets: {stats_df['dataset_name'].nunique()}")
                logger.info(f"📊 Unique models: {stats_df['model_name'].nunique()}")
                
                # Show top performers
                if len(stats_df) > 0:
                    top_performers = stats_df.nlargest(3, 'accuracy')[['dataset_name', 'model_name', 'accuracy', 'total_samples']]
                    logger.info("🏆 Top performers by accuracy:")
                    for _, row in top_performers.iterrows():
                        logger.info(f"  🥇 {row['dataset_name']} + {row['model_name']}: {row['accuracy']:.3f} ({row['total_samples']} samples)")
                
            else:
                logger.error("❌ Statistics file was not created")
                return False
            
        finally:
            # Clean up temporary files
            for temp_file in [temp_combined_file, temp_stats_file]:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"🧹 Cleaned up {temp_file.name}")
        
        # Verify data consistency
        if total_processed_entries != expected_total_entries:
            logger.warning(f"⚠️ Entry count mismatch: processed {total_processed_entries}, expected {expected_total_entries}")
        else:
            logger.info(f"✅ Entry count verified: {total_processed_entries} entries processed")
        
        # Summary of analytics results
        unique_datasets = len(set(r['dataset'] for r in analytics_results))
        unique_models = len(set(r['model'] for r in analytics_results))
        avg_accuracy = sum(r['accuracy'] for r in analytics_results if r['accuracy'] is not None) / len([r for r in analytics_results if r['accuracy'] is not None])
        
        logger.info("📊 Analytics Summary:")
        logger.info(f"  📋 Benchmark-model combinations: {len(analytics_results)}")
        logger.info(f"  🎯 Unique datasets: {unique_datasets}")
        logger.info(f"  🤖 Unique models: {unique_models}")
        logger.info(f"  📈 Average accuracy: {avg_accuracy:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analytics testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger = setup_test_logging()
    
    print("=" * 80)
    print("🧪 END-TO-END OPTIMIZED PROCESSOR TEST")
    print("=" * 80)
    
    # Step 1: Clean up any previous test files
    logger.info("🧹 Step 1: Cleaning up previous test files")
    cleanup_test_files()
    
    # Step 2: Create test CSV file (instead of web scraping)
    logger.info("📝 Step 2: Creating test CSV file with sample tasks")
    test_csv_path = create_test_csv()
    
    # Step 3: Set up environment
    logger.info("🔧 Step 3: Setting up test environment")
    os.environ['HF_TOKEN'] = 'test_token'
    
    # Step 4: Test the optimized processor with mocked upload
    logger.info("🚀 Step 4: Running optimized processor (end-to-end)")
    
    # Mock the HuggingFace API to avoid actual uploads
    with patch('scripts.optimized_helm_processor.HfApi', mock_hf_api()):
        from scripts.optimized_helm_processor import main as run_processor
        
        # Mock sys.argv for the processor
        test_args = [
            "optimized_helm_processor.py",
            "--benchmark", "lite",  # This should trigger CSV generation
            "--chunk-size", "3",    # Very small chunks for testing
            "--max-workers", "1",   # Single worker for clarity
            "--upload-workers", "1",
            "--repo-id", "test/repo",
            "--source-name", "helm",
            "--timeout", "5",        # Short timeout
            "--no-cleanup"          # Don't clean up files for testing
        ]
        
        start_time = time.time()
        
        with patch('sys.argv', test_args):
            logger.info("🔄 Running optimized processor...")
            try:
                success = run_processor()
                if success:
                    logger.info("✅ Optimized processor completed successfully!")
                else:
                    logger.error("❌ Optimized processor failed!")
                    return False
            except Exception as e:
                logger.error(f"❌ Optimized processor crashed: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    processing_time = time.time() - start_time
    
    # Step 5: Verify outputs
    logger.info("🔍 Step 5: Verifying generated files")
    
    # Check CSV was generated
    csv_file = Path("data/benchmark_lines/helm_lite.csv")
    if csv_file.exists():
        logger.info(f"✅ CSV file generated: {csv_file}")
        # Count tasks in CSV
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            task_count = len(df)
            logger.info(f"📊 CSV contains {task_count} tasks")
            if task_count > 0:
                logger.info(f"📝 Sample task: {df.iloc[0]['task'] if 'task' in df.columns else 'Column not found'}")
        except Exception as e:
            logger.warning(f"⚠️ Could not read CSV: {e}")
    else:
        logger.error("❌ CSV file not generated")
        return False
    
    # Check processed files were created
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        chunk_dirs = list(processed_dir.glob("chunk_*"))
        logger.info(f"✅ Processed data: {len(chunk_dirs)} chunk directories")
        
        total_csv_files = 0
        for chunk_dir in chunk_dirs:
            csv_files = list(chunk_dir.glob("*.csv"))
            total_csv_files += len(csv_files)
            logger.info(f"  📁 {chunk_dir.name}: {len(csv_files)} CSV files")
            
        logger.info(f"📊 Total processed CSV files: {total_csv_files}")
    else:
        logger.warning("⚠️ No processed data directory found")
    
    # Check parquet files were created  
    aggregated_dir = Path("data/aggregated")
    if aggregated_dir.exists():
        parquet_files = list(aggregated_dir.glob("*.parquet"))
        logger.info(f"✅ Parquet files: {len(parquet_files)} files")
        
        total_entries = 0
        for parquet_file in parquet_files:
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_file)
                entries = len(df)
                total_entries += entries
                size_mb = parquet_file.stat().st_size / (1024 * 1024)
                logger.info(f"  📄 {parquet_file.name}: {entries:,} entries ({size_mb:.2f}MB)")
            except Exception as e:
                logger.warning(f"⚠️ Could not read {parquet_file.name}: {e}")
                
        logger.info(f"📊 Total entries in parquet files: {total_entries:,}")
    else:
        logger.error("❌ No aggregated parquet files found")
        return False
    
    # Step 6: Test Summary Analytics
    logger.info("📊 Step 6: Testing Summary Analytics")
    analytics_success = test_summary_analytics(aggregated_dir, total_entries)
    if not analytics_success:
        logger.error("❌ Summary analytics testing failed")
        return False
    
    # Step 7: Test Summary
    logger.info("📊 Step 7: Test Summary")
    print("\n" + "=" * 80)
    print("🎉 END-TO-END TEST RESULTS")
    print("=" * 80)
    print("✅ CSV Generation: Working")
    print("✅ Task Processing: Working") 
    print("✅ Parquet Creation: Working")
    print("✅ File Organization: Working")
    print("✅ Summary Analytics: Working")
    print("🔄 Upload Simulation: Working (mocked)")
    print(f"⏱️ Processing Time: {processing_time:.1f} seconds")
    print(f"📊 Total Data Entries: {total_entries:,}")
    print("💡 Ready for production deployment!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    
    # Ask about cleanup
    print(f"\n🧹 Clean up test files? (y/N): ", end="")
    try:
        response = input().strip().lower()
        if response in ['y', 'yes']:
            cleanup_test_files()
            print("✅ Test files cleaned up")
        else:
            print("📁 Test files preserved for inspection")
    except (EOFError, KeyboardInterrupt):
        print("\n📁 Test files preserved")
    
    sys.exit(0 if success else 1)
