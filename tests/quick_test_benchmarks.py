#!/usr/bin/env python3
"""
Quick test to identify issues with different HELM benchmarks.
Tests downloading and processing of sample tasks from lite, mmlu, and classic benchmarks.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Sample tasks for each benchmark (hardcoded to avoid CSV generation)
SAMPLE_TASKS = {
    'lite': [
        'commonsense:dataset=openbookqa,method=multiple_choice_joint,model=01-ai_yi-34b',
        'commonsense:dataset=hellaswag,method=multiple_choice_joint,model=01-ai_yi-34b',
        'reading_comprehension:dataset=boolq,method=multiple_choice_joint,model=01-ai_yi-34b',
        'sentiment:dataset=imdb,method=multiple_choice_joint,model=01-ai_yi-34b',
        'toxicity_detection:dataset=civil_comments,method=multiple_choice_joint,model=01-ai_yi-34b'
    ],
    'mmlu': [
        'mmlu:subject=clinical_knowledge,method=multiple_choice_joint,model=01-ai_yi-6b,eval_split=test,groups=mmlu_clinical_knowledge',
        'mmlu:subject=medical_genetics,method=multiple_choice_joint,model=01-ai_yi-6b,eval_split=test,groups=mmlu_medical_genetics',
        'mmlu:subject=anatomy,method=multiple_choice_joint,model=01-ai_yi-6b,eval_split=test,groups=mmlu_anatomy',
        'mmlu:subject=professional_medicine,method=multiple_choice_joint,model=01-ai_yi-6b,eval_split=test,groups=mmlu_professional_medicine',
        'mmlu:subject=college_biology,method=multiple_choice_joint,model=01-ai_yi-6b,eval_split=test,groups=mmlu_college_biology'
    ],
    'classic': [
        'natural_language_inference:dataset=snli,method=multiple_choice_joint,model=01-ai_yi-6b',
        'question_answering:dataset=natural_questions_closedbook,method=multiple_choice_joint,model=01-ai_yi-6b',
        'reading_comprehension:dataset=squad,method=multiple_choice_joint,model=01-ai_yi-6b',
        'sentiment:dataset=imdb,method=multiple_choice_joint,model=01-ai_yi-6b',
        'summarization:dataset=cnn_dailymail,method=multiple_choice_joint,model=01-ai_yi-6b'
    ]
}

async def test_task_download_and_processing(task: str, benchmark: str):
    """Test downloading and processing a single task."""
    try:
        logger.info(f"📥 Testing download for: {task}")
        
        # Test download with proper arguments
        from src.core.helm.downloader import download_tasks
        output_dir = "data/downloads"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run download_tasks in executor since it's not async
        downloaded = await asyncio.get_event_loop().run_in_executor(
            None, download_tasks, [task], output_dir, benchmark
        )
        
        if not downloaded:
            logger.error(f"❌ Download failed for {task}")
            return False
        
        downloaded_path = downloaded[0]
        logger.info(f"✅ Download successful: {downloaded_path}")
        
        # Check if file exists
        if not Path(downloaded_path).exists():
            logger.error(f"❌ Downloaded file doesn't exist: {downloaded_path}")
            return False
        
        # Test processing
        logger.info(f"🔄 Testing processing for downloaded task...")
        from src.core.helm.processor import process_line
        
        result = process_line(downloaded_path)
        if result:
            logger.info(f"✅ Processing successful: {len(result)} entries")
            return True
        else:
            logger.error(f"❌ Processing returned no data")
            return False
            
    except Exception as e:
        logger.error(f"❌ Task test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_benchmark(benchmark: str):
    """Test a benchmark with sample tasks."""
    logger.info(f"\n🧪 Testing {benchmark} benchmark")
    
    if benchmark not in SAMPLE_TASKS:
        logger.error(f"❌ No sample tasks defined for {benchmark}")
        return False
    
    tasks = SAMPLE_TASKS[benchmark]
    logger.info(f"📋 Testing {len(tasks)} sample tasks")
    
    # Test just the first task to identify structural issues
    success = await test_task_download_and_processing(tasks[0], benchmark)
    
    if success:
        logger.info(f"✅ {benchmark} benchmark test PASSED")
    else:
        logger.error(f"❌ {benchmark} benchmark test FAILED")
    
    return success

async def main():
    """Test all benchmarks with sample tasks."""
    logger.info("🚀 Starting quick benchmark tests (1 task per benchmark)")
    
    benchmarks = ['lite', 'mmlu', 'classic']
    results = {}
    
    for benchmark in benchmarks:
        results[benchmark] = await test_benchmark(benchmark)
    
    # Summary
    logger.info("\n📊 TEST SUMMARY:")
    for benchmark, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {benchmark}: {status}")
    
    failed_benchmarks = [b for b, passed in results.items() if not passed]
    if failed_benchmarks:
        logger.error(f"\n💥 Failed benchmarks: {failed_benchmarks}")
        logger.info("\n🔍 This indicates structural issues that prevent these benchmarks from working")
        return False
    else:
        logger.info(f"\n🎉 All benchmarks passed basic tests!")
        return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
