#!/usr/bin/env python3
"""
Test script to verify all three benchmarks (lite, mmlu, classic) can complete end-to-end.
This will test with a small subset of tasks to identify any issues.
"""

import asyncio
import sys
import subprocess
from pathlib import Path
import logging
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_benchmark_setup(benchmark: str) -> bool:
    """Test if a benchmark can be set up properly (CSV generation, initial steps)"""
    try:
        logger.info(f"🧪 Testing benchmark setup: {benchmark}")
        
        # Check if CSV file exists or can be generated
        csv_path = Path(f"data/benchmark_lines/helm_{benchmark}.csv")
        
        if not csv_path.exists():
            logger.info(f"📋 CSV not found for {benchmark}, attempting to generate...")
            
            # Try to generate CSV using web scraper
            try:
                from src.sources.helm.web_scraper import main as create_csv_main
                asyncio.run(create_csv_main(benchmark=benchmark, output_dir=str(csv_path.parent)))
                logger.info(f"✅ Generated CSV for {benchmark}")
            except Exception as e:
                logger.error(f"❌ Failed to generate CSV for {benchmark}: {e}")
                return False
        
        # Check if CSV is readable and has content
        if csv_path.exists():
            try:
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                    logger.info(f"📊 CSV for {benchmark} has {len(lines)} lines")
                    if len(lines) > 1:  # Header + at least one task
                        return True
                    else:
                        logger.error(f"❌ CSV for {benchmark} is empty or only has header")
                        return False
            except Exception as e:
                logger.error(f"❌ Error reading CSV for {benchmark}: {e}")
                return False
        else:
            logger.error(f"❌ CSV file still doesn't exist for {benchmark}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Setup test failed for {benchmark}: {e}")
        return False

def test_benchmark_processing(benchmark: str, max_tasks: int = 2) -> bool:
    """Test processing a small number of tasks for a benchmark"""
    try:
        logger.info(f"🔄 Testing processing for benchmark: {benchmark} (max {max_tasks} tasks)")
        
        # Create a test script command
        cmd = [
            sys.executable, "scripts/optimized_helm_processor.py",
            "--benchmark", benchmark,
            "--chunk-size", str(max_tasks),
            "--max-workers", "2",
            "--repo-id", "evaleval/every_eval_ever",
            "--source-name", "helm",
            "--timeout", "5",  # Short timeout for testing
            "--no-cleanup"  # Keep files for inspection
        ]
        
        logger.info(f"🚀 Running command: {' '.join(cmd)}")
        
        # Run the command with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.getcwd()
        )
        
        logger.info(f"📤 Command exit code: {result.returncode}")
        
        if result.stdout:
            logger.info("📋 STDOUT:")
            for line in result.stdout.split('\n')[-20:]:  # Show last 20 lines
                if line.strip():
                    logger.info(f"   {line}")
        
        if result.stderr:
            logger.info("📋 STDERR:")
            for line in result.stderr.split('\n')[-10:]:  # Show last 10 lines
                if line.strip():
                    logger.info(f"   {line}")
        
        if result.returncode == 0:
            logger.info(f"✅ Processing test passed for {benchmark}")
            return True
        else:
            logger.error(f"❌ Processing test failed for {benchmark} (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Processing test timed out for {benchmark}")
        return False
    except Exception as e:
        logger.error(f"❌ Processing test failed for {benchmark}: {e}")
        return False

def main():
    """Run tests for all benchmarks"""
    logger.info("🧪 Starting comprehensive benchmark testing...")
    
    benchmarks = ["lite", "mmlu", "classic"]
    results = {}
    
    # Test setup for each benchmark
    logger.info("\n" + "="*60)
    logger.info("📋 PHASE 1: Testing benchmark setup...")
    logger.info("="*60)
    
    for benchmark in benchmarks:
        results[f"{benchmark}_setup"] = test_benchmark_setup(benchmark)
    
    # Test processing for benchmarks that passed setup
    logger.info("\n" + "="*60)
    logger.info("🔄 PHASE 2: Testing benchmark processing...")
    logger.info("="*60)
    
    for benchmark in benchmarks:
        if results.get(f"{benchmark}_setup", False):
            results[f"{benchmark}_processing"] = test_benchmark_processing(benchmark)
        else:
            logger.warning(f"⚠️ Skipping processing test for {benchmark} (setup failed)")
            results[f"{benchmark}_processing"] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    for benchmark in benchmarks:
        setup_ok = results.get(f"{benchmark}_setup", False)
        process_ok = results.get(f"{benchmark}_processing", False)
        
        setup_status = "✅" if setup_ok else "❌"
        process_status = "✅" if process_ok else "❌"
        
        logger.info(f"{benchmark:8} | Setup: {setup_status} | Processing: {process_status}")
    
    # Overall result
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! All benchmarks should work end-to-end.")
    else:
        failed_tests = [k for k, v in results.items() if not v]
        logger.error(f"\n❌ SOME TESTS FAILED: {', '.join(failed_tests)}")
        
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
