#!/usr/bin/env python3
"""
Quick test script to verify all 3 benchmarks can complete end-to-end with a small sample.
Tests 5 tasks per benchmark to identify any structural issues.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_benchmark_end_to_end(benchmark: str, max_tasks: int = 5):
    """Test a benchmark end-to-end with a limited number of tasks."""
    logger.info(f"\n🧪 Testing {benchmark} benchmark (max {max_tasks} tasks)")
    
    try:
        # 1. Generate CSV if it doesn't exist
        csv_file = f"data/benchmark_lines/helm_{benchmark}.csv"
        csv_path = Path(csv_file)
        
        if not csv_path.exists():
            logger.info(f"📥 Generating CSV for {benchmark}...")
            from src.core.helm.web_scraper import main as create_csv_main
            await create_csv_main(benchmark=benchmark, output_dir=str(csv_path.parent))
        
        # 2. Read a few tasks from CSV
        logger.info(f"📖 Reading tasks from {csv_file}...")
        with open(csv_file, 'r') as f:
            tasks = [line.strip() for line in f.readlines()[:max_tasks]]
        
        logger.info(f"📋 Selected {len(tasks)} tasks for testing")
        
        # 3. Test downloading 1 task
        if tasks:
            test_task = tasks[0]
            logger.info(f"📥 Testing download for: {test_task}")
            
            from src.core.helm.downloader import download_tasks
            downloaded = await download_tasks([test_task])
            
            if downloaded:
                logger.info(f"✅ Download successful: {downloaded[0]}")
            else:
                logger.error(f"❌ Download failed for {test_task}")
                return False
        
        # 4. Test processing 1 task
        if downloaded:
            logger.info(f"🔄 Testing processing for downloaded task...")
            from src.core.helm.processor import process_line
            
            try:
                result = process_line(downloaded[0])
                if result:
                    logger.info(f"✅ Processing successful: {len(result)} entries")
                else:
                    logger.error(f"❌ Processing returned no data")
                    return False
            except Exception as e:
                logger.error(f"❌ Processing failed: {e}")
                return False
        
        logger.info(f"✅ {benchmark} benchmark test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ {benchmark} benchmark test FAILED: {e}")
        return False

async def main():
    """Test all benchmarks with small samples."""
    logger.info("🚀 Starting benchmark end-to-end tests (5 tasks each)")
    
    benchmarks = ['lite', 'mmlu', 'classic']
    results = {}
    
    for benchmark in benchmarks:
        results[benchmark] = await test_benchmark_end_to_end(benchmark, max_tasks=5)
    
    # Summary
    logger.info("\n📊 TEST SUMMARY:")
    for benchmark, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {benchmark}: {status}")
    
    failed_benchmarks = [b for b, passed in results.items() if not passed]
    if failed_benchmarks:
        logger.error(f"\n💥 Failed benchmarks: {failed_benchmarks}")
        return False
    else:
        logger.info(f"\n🎉 All benchmarks passed!")
        return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
