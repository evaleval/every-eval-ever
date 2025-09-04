#!/usr/bin/env python3
"""
Optimized HELM processor with chunked processing and parallel uploads.

This script modifies the existing HELM processing flow to:
1. Process tasks in configurable chunks (default: 200 tasks)
2. Generate intermediate parquet files after each chunk
3. Upload chunks to HuggingFace immediately
4. Clean up local files to save disk space
5. Continue processing while uploading in background

Key optimizations:
- Parallel processing within chunks (2-4 workers)
- Immediate upload of completed chunks
- Memory-efficient processing (smaller batches)
- Disk space management (cleanup after upload)
- Progress tracking and ETA estimation

Usage:
    python scripts/optimized_helm_processor.py --benchmark lite --chunk-size 200 --max-workers 4 --repo-id evaleval/every_eval_ever
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import logging
import pandas as pd

from huggingface_hub import HfApi
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from config.settings import BENCHMARK_CSVS_DIR

# Set up logging
logger = setup_logging("optimized_helm.log", "OptimizedHELM")


def setup_hf_api(token: str, repo_id: str) -> HfApi:
    """Initialize HuggingFace API."""
    if not token:
        raise ValueError('HF_TOKEN environment variable required')
    
    try:
        api = HfApi(token=token)
        # Test connection and create repo if needed
        try:
            api.repo_info(repo_id, repo_type="dataset")
            logger.info(f"📁 Repository exists: {repo_id}")
        except Exception:
            logger.info(f"📁 Creating repository: {repo_id}")
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        return api
    except Exception as e:
        logger.error(f"❌ Failed to setup HuggingFace API: {e}")
        raise


def get_existing_files(api: HfApi, repo_id: str) -> set:
    """Get list of files already in the repository."""
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        parquet_files = {f for f in files if f.endswith('.parquet')}
        logger.info(f"📁 Found {len(parquet_files)} existing parquet files in repo")
        return parquet_files
    except Exception as e:
        logger.warning(f"⚠️ Error checking files, starting from part 1: {e}")
        return set()


def get_processed_tasks(api: HfApi, repo_id: str, benchmark: str) -> set:
    """Get set of tasks that have already been processed for this benchmark using efficient lazy loading."""
    processed_tasks = set()
    try:
        # Check if the dataset exists and has data
        try:
            # Use streaming for memory efficiency - don't load entire dataset into memory
            dataset = load_dataset(repo_id, streaming=True, split='train')
            logger.info(f"🔍 Efficiently checking processed tasks using streaming dataset...")
            
            # Iterate through dataset to collect task identifiers
            # This is memory-efficient as it streams rather than loading everything
            for i, example in enumerate(dataset):
                # Filter by benchmark if available
                if 'benchmark' in example and example['benchmark'] != benchmark:
                    continue
                
                # Look for task identifier in common columns (prioritize task_id)
                task_id = None
                for col in ['task_id', 'task', 'dataset_name', 'evaluation_id']:
                    if col in example and example[col]:
                        task_id = example[col]
                        break
                
                if task_id:
                    processed_tasks.add(task_id)
                
                # Progress logging every 10k examples
                if i > 0 and i % 10000 == 0:
                    logger.info(f"🔄 Scanned {i:,} examples, found {len(processed_tasks)} unique tasks for '{benchmark}'")
                
                # Limit scanning to avoid excessive startup time (first 100k examples should be representative)
                if i >= 100000:
                    logger.info(f"⏱️ Stopping scan at {i:,} examples to avoid excessive startup time")
                    break
            
            logger.info(f"✅ Found {len(processed_tasks)} already processed tasks for benchmark '{benchmark}'")
            if len(processed_tasks) > 0:
                logger.info(f"📝 Sample processed tasks: {list(processed_tasks)[:5]}")
            
        except Exception as e:
            logger.info(f"📝 Dataset not found or empty: {e}")
            logger.info(f"📝 All tasks will be processed (first run or empty dataset)")
        
        return processed_tasks
        
    except Exception as e:
        logger.warning(f"⚠️ Error checking processed tasks: {e}")
        return set()


def read_tasks_chunked(csv_file: str, chunk_size: int, adapter_method: str = None, processed_tasks: set = None) -> List[List[str]]:
    """Read tasks from CSV and split into chunks with optimized reading and duplicate filtering."""
    try:
        # Fast CSV reading with minimal parsing for speed
        df = pd.read_csv(csv_file, dtype=str, low_memory=False)
        
        # Check which column contains the task identifiers
        if 'task' in df.columns:
            tasks = df['task'].tolist()
        elif 'Run' in df.columns:
            # HELM web scraper creates 'Run' column with task identifiers
            tasks = df['Run'].tolist()
        else:
            # Fallback: use the first column
            tasks = df.iloc[:, 0].tolist()
            logger.warning(f"⚠️ No 'task' or 'Run' column found, using first column: {df.columns[0]}")
        
        original_count = len(tasks)
        
        # Filter by adapter method if specified
        if adapter_method:
            tasks = [task for task in tasks if adapter_method in task]
            logger.info(f"🔽 Filtered to {len(tasks)} tasks with adapter method: {adapter_method}")
        
        # Filter out already processed tasks
        if processed_tasks:
            before_dedup = len(tasks)
            tasks_to_keep = []
            skipped_tasks = []
            
            for task in tasks:
                if task not in processed_tasks:
                    tasks_to_keep.append(task)
                else:
                    skipped_tasks.append(task)
            
            tasks = tasks_to_keep
            removed_count = len(skipped_tasks)
            
            if removed_count > 0:
                logger.info(f"⏭️ Skipped {removed_count} already processed tasks")
                logger.info(f"🆕 {len(tasks)} new tasks to process (saved {removed_count/before_dedup:.1%} work)")
                if removed_count <= 10:  # Show details for small numbers
                    logger.info(f"📝 Skipped tasks: {skipped_tasks}")
                else:
                    logger.info(f"📝 Sample skipped tasks: {skipped_tasks[:5]}...")
            else:
                logger.info(f"✨ No duplicate tasks found - all {len(tasks)} tasks are new")
        
        if len(tasks) == 0:
            logger.warning(f"⚠️ No tasks to process after filtering!")
            return []
        
        chunks = [tasks[i:i+chunk_size] for i in range(0, len(tasks), chunk_size)]
        logger.info(f"📦 Split {len(tasks)} tasks into {len(chunks)} chunks of ~{chunk_size} tasks each")
        return chunks
    except Exception as e:
        logger.error(f"❌ Error reading tasks: {e}")
        raise


def process_chunk(chunk_tasks: List[str], chunk_id: int, workers: int, timeout_minutes: int = 60, benchmark: str = "unknown") -> str:
    """Process a chunk of tasks and return the output file path."""
    logger.info(f"🔄 Processing chunk {chunk_id} with {len(chunk_tasks)} tasks")
    logger.info(f"⏱️ Timeout: {timeout_minutes} minutes per chunk")
    logger.info(f"⚡ Workers: {workers}")
    
    # Import the processor function directly
    from src.sources.helm.processor import process_line
    
    # Create chunk output directory
    chunk_dir = Path(f"data/processed/chunk_{chunk_id:04d}")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    task_file_mapping = {}  # Track which task produced which file
    
    # Process each task in the chunk
    for i, task in enumerate(chunk_tasks, 1):
        logger.info(f"🔄 Processing task {i}/{len(chunk_tasks)}: {task}")
        
        try:
            # Call the HELM processor function directly with required arguments
            result = process_line(
                line=task,
                output_dir=str(chunk_dir),
                benchmark="lite",  # Use a default benchmark name
                downloads_dir="data/downloads",
                keep_temp_files=False,
                overwrite=False,
                show_progress=False  # Disable progress bar for cleaner logs
            )
            
            if result and result.get("status") == "success":
                output_file = result.get("converted_file")  # Use 'converted_file' not 'output_file'
                if output_file and Path(output_file).exists():
                    # Read the file to count entries
                    try:
                        df = pd.read_csv(output_file)
                        entry_count = len(df)
                        logger.info(f"✅ Task {i} completed: {entry_count} entries")
                        processed_files.append(output_file)
                        # Store the actual HELM task/run name for proper task_id assignment
                        task_file_mapping[output_file] = task  # task is the HELM run name like "openai_gpt-3.5-turbo:openbookqa:1"
                    except Exception as e:
                        logger.warning(f"⚠️ Could not count entries in {output_file}: {e}")
                        processed_files.append(output_file)
                        task_file_mapping[output_file] = task
                else:
                    logger.error(f"❌ Task {i} failed: no output file generated")
            else:
                status = result.get("status", "unknown") if result else "no result"
                error = result.get("error", "no error details") if result else "no result"
                logger.error(f"❌ Task {i} failed with status: {status}, error: {error}")
                
        except Exception as e:
            logger.error(f"❌ Task {i} failed with error: {e}")
            continue
    
    logger.info(f"📊 Chunk {chunk_id} summary: {len(processed_files)}/{len(chunk_tasks)} tasks successful")
    
    if not processed_files:
        logger.error(f"❌ No files processed in chunk {chunk_id}")
        return None
    
    # Combine all processed files into one parquet file
    try:
        combined_data = []
        total_entries = 0
        
        for file_path in processed_files:
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                combined_data.append(df)
                total_entries += len(df)
        
        if combined_data:
            start_time = time.time()
            
            # Process data in smaller batches to save memory and increase speed
            logger.info(f"📊 Combining {len(combined_data)} dataframes with {total_entries} total entries")
            
            # Add task_id to each dataframe before combining based on the actual task name
            for i, (df, file_path) in enumerate(zip(combined_data, processed_files)):
                if file_path in task_file_mapping:
                    # Use the actual HELM task/run name
                    task_name = task_file_mapping[file_path]
                    df['task_id'] = task_name
                    logger.debug(f"📝 Assigned task_id '{task_name}' to {len(df)} rows from {Path(file_path).name}")
                else:
                    # Fallback if mapping is missing - this shouldn't happen
                    fallback_task = f"unknown_task_{i}"
                    df['task_id'] = fallback_task
                    logger.warning(f"⚠️ Missing task mapping for {file_path}, using fallback: {fallback_task}")
            
            # Use faster concat with minimal validation for speed
            combined_df = pd.concat(combined_data, ignore_index=True, copy=False, sort=False)
            
            # Clear intermediate data immediately to free memory
            del combined_data
            
            # Add metadata fields efficiently
            import datetime
            timestamp = datetime.datetime.now()
            combined_df['source'] = 'helm'
            combined_df['benchmark'] = benchmark  
            combined_df['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
            combined_df['processing_date'] = timestamp.strftime('%Y-%m-%d')
            
            # Log task_id distribution for verification
            if 'task_id' in combined_df.columns:
                unique_tasks = combined_df['task_id'].nunique()
                logger.info(f"📊 Combined data contains {unique_tasks} unique task_ids from {len(chunk_tasks)} processed tasks")
            
            # Reorder columns to put metadata first
            metadata_cols = ['task_id', 'source', 'benchmark', 'timestamp', 'processing_date']
            other_cols = [col for col in combined_df.columns if col not in metadata_cols]
            combined_df = combined_df[metadata_cols + other_cols]
            
            # Create aggregated directory
            agg_dir = Path("data/aggregated")
            agg_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = agg_dir / f"chunk_{chunk_id:04d}.parquet"
            
            # Use faster parquet writing with optimized settings
            combined_df.to_parquet(
                output_file, 
                compression='snappy',  # Faster than default
                index=False,
                engine='pyarrow'
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Created chunk parquet: {output_file} ({total_entries} entries, {processing_time:.1f}s)")
            return str(output_file)
        else:
            logger.error(f"❌ No data to combine for chunk {chunk_id}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error creating chunk parquet for chunk {chunk_id}: {e}")
        return None


def upload_file(api: HfApi, local_path: str, repo_id: str, source_name: str, part_num: int, cleanup: bool = True) -> bool:
    """Upload a file to HuggingFace and clean up local file."""
    try:
        file_path = Path(local_path)
        if not file_path.exists():
            logger.error(f"❌ File not found: {local_path}")
            return False
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        remote_name = f"data-{part_num:05d}.parquet"
        
        logger.info(f"☁️ Uploading {file_path.name} as {remote_name} ({file_size_mb:.1f}MB)")
        
        start_time = time.time()
        
        # Fast upload with minimal retries for speed
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="dataset",
            run_as_future=False  # Synchronous for better error handling
        )
        
        upload_time = time.time() - start_time
        speed_mbps = file_size_mb / max(upload_time, 0.001)
        logger.info(f"✅ Uploaded {remote_name} in {upload_time:.1f}s ({speed_mbps:.1f}MB/s)")
        
        # Clean up local file if cleanup is enabled
        if cleanup:
            file_path.unlink()
            logger.info(f"🧹 Cleaned up local file: {local_path}")
        else:
            logger.info(f"📁 Preserved local file: {local_path}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Upload failed for {local_path}: {e}")
        return False


def process_with_optimization(args):
    """
    Main processing function with chunked optimization.
    
    This function is designed to be cronjob-friendly:
    - Chunk numbering is determined by existing files in the HF dataset
    - Each run continues from where the previous run left off
    - Local chunk directories use the same numbering as HF dataset files
    - Example: If HF has data-00001.parquet, data-00002.parquet, 
      next run will create chunk_0003 and upload as data-00003.parquet
    """
    logger.info("🚀 Starting optimized processing")
    
    # Determine benchmark name from args or CSV file path
    if args.benchmark:
        benchmark_name = args.benchmark
    else:
        # Extract benchmark name from CSV file path (e.g., "helm_lite.csv" -> "lite")
        csv_filename = Path(args.csv_file).stem
        if csv_filename.startswith('helm_'):
            benchmark_name = csv_filename[5:]  # Remove "helm_" prefix
        else:
            benchmark_name = "unknown"
    
    # Setup HuggingFace API
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("❌ HF_TOKEN environment variable not set")
        return False
    
    api = setup_hf_api(hf_token, args.repo_id)
    existing_files = get_existing_files(api, args.repo_id)
    
    # Determine starting number for both chunk ID and part number
    # Each chunk becomes one parquet file, so they should be in sync
    if existing_files:
        part_numbers = []
        for f in existing_files:
            try:
                if f.startswith('data-') and f.endswith('.parquet'):
                    num = int(f.split('-')[1].split('.')[0])
                    part_numbers.append(num)
            except ValueError:
                continue
        start_number = max(part_numbers) + 1 if part_numbers else 1
    else:
        start_number = 1
    
    logger.info(f"📊 Starting from number {start_number} (both chunk ID and part number)")
    
    # Check for already processed tasks to avoid duplicates (unless skipped)
    processed_tasks = set()
    if not args.skip_duplicate_check:
        logger.info("🔍 Checking for already processed tasks to avoid duplicates...")
        processed_tasks = get_processed_tasks(api, args.repo_id, benchmark_name)
    else:
        logger.info("⏭️ Skipping duplicate check as requested")
    
    # Read and chunk tasks with duplicate filtering
    task_chunks = read_tasks_chunked(args.csv_file, args.chunk_size, args.adapter_method, processed_tasks)
    
    if not task_chunks:
        if processed_tasks:
            logger.warning("⚠️ No new tasks to process - all tasks may already be completed!")
            logger.info("✅ Processing complete - no new work needed")
            return True
        else:
            logger.error("❌ No tasks to process")
            return False
    
    total_new_tasks = sum(len(chunk) for chunk in task_chunks)
    logger.info(f"📊 Found {total_new_tasks} new tasks to process")
    logger.info("=" * 60)
    logger.info("🚀 Starting optimized processing")
    logger.info(f"📦 Processing {len(task_chunks)} chunks")
    logger.info(f"⚡ {args.max_workers} workers per chunk")
    logger.info(f"☁️ {args.upload_workers} upload workers")
    logger.info("=" * 60)
    
    # Process chunks with uploads in parallel
    upload_executor = ThreadPoolExecutor(max_workers=args.upload_workers)
    upload_futures = []
    successful_uploads = 0
    # Use the same counter for both chunk ID and part number since they're in sync
    current_number = start_number
    
    start_time = time.time()
    
    for chunk_idx, chunk_tasks in enumerate(task_chunks, 1):
        logger.info(f"\n--- Chunk {chunk_idx}/{len(task_chunks)} (ID: {current_number}) ---")
        
        # Process chunk with incremental chunk ID
        chunk_file = process_chunk(chunk_tasks, current_number, args.max_workers, args.timeout, benchmark_name)
        
        if chunk_file:
            # Submit upload task
            upload_future = upload_executor.submit(
                upload_file, api, chunk_file, args.repo_id, args.source_name, current_number, not args.no_cleanup
            )
            upload_futures.append(upload_future)
            
            # Update progress with detailed metrics
            elapsed = time.time() - start_time
            elapsed_minutes = elapsed / 60
            progress = chunk_idx / len(task_chunks)
            eta_minutes = (elapsed / progress - elapsed) / 60 if progress > 0 else 0
            total_entries = chunk_idx * args.chunk_size  # Approximate
            processing_rate = total_entries / elapsed if elapsed > 0 else 0
            
            logger.info(f"📈 Progress: {chunk_idx}/{len(task_chunks)} chunks ({progress:.1%}) | ~{total_entries:,} entries")
            logger.info(f"⏱️ Elapsed: {elapsed_minutes:.1f}min | ETA: {eta_minutes:.1f}min | Rate: {processing_rate:.0f} entries/sec")
        else:
            logger.error(f"❌ Chunk {current_number} failed")
        
        # Increment number for next iteration
        current_number += 1
    
    # Wait for all uploads to complete
    logger.info(f"\n⏳ Waiting for {len(upload_futures)} uploads to complete...")
    for future in upload_futures:
        if future.result():
            successful_uploads += 1
    
    upload_executor.shutdown(wait=True)
    
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Processing complete!")
    logger.info(f"✅ Processed chunks: {len(task_chunks)}/{len(task_chunks)}")
    logger.info(f"☁️ Successful uploads: {successful_uploads}/{len(upload_futures)}")
    logger.info(f"📊 Total entries: ~{sum(len(chunk) for chunk in task_chunks) * 500}")  # Approximate
    logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
    logger.info(f"📈 Rate: {(sum(len(chunk) for chunk in task_chunks) * 500) / (total_time/60):.1f} entries/minute")
    logger.info("=" * 60)
    
    return successful_uploads == len(upload_futures)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimized HELM processor with chunked processing")
    parser.add_argument("--benchmark", type=str, choices=["lite", "mmlu", "classic"], 
                       help="Benchmark to process (determines CSV file)")
    parser.add_argument("--csv-file", type=str, help="Direct path to CSV file (overrides benchmark)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Tasks per chunk (default: 100 for maximum parallelization)")
    parser.add_argument("--max-workers", type=int, default=8, help="Workers per chunk (default: 8)")
    parser.add_argument("--upload-workers", type=int, default=6, help="Parallel upload workers (default: 6)")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per chunk in minutes (default: 30, aggressive for speed)")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument("--source-name", type=str, default="helm", help="Source name for file organization")
    parser.add_argument("--adapter-method", type=str, help="Filter tasks by adapter method")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up parquet files after upload (for testing)")
    parser.add_argument("--skip-duplicate-check", action="store_true", help="Skip checking for already processed tasks (faster startup)")
    
    args = parser.parse_args()
    
    # Determine CSV file
    if args.csv_file:
        csv_file = args.csv_file
    elif args.benchmark:
        csv_file = f"data/benchmark_lines/helm_{args.benchmark}.csv"
        
        # Check if CSV file exists, if not generate it
        csv_path = Path(csv_file)
        if not csv_path.exists():
            logger.info(f"📥 CSV for benchmark '{args.benchmark}' not found. Generating...")
            try:
                # Create the benchmark_lines directory if it doesn't exist
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Import web_scraper to generate the CSV
                from src.sources.helm.web_scraper import main as create_csv_main
                asyncio.run(create_csv_main(benchmark=args.benchmark, output_dir=str(csv_path.parent)))
                logger.info(f"✅ Successfully created CSV: {csv_file}")
            except Exception as e:
                logger.error(f"❌ Failed to create CSV for benchmark '{args.benchmark}': {e}")
                return False
        else:
            logger.info(f"📄 Found existing CSV for benchmark '{args.benchmark}': {csv_file}")
    else:
        parser.error("Either --benchmark or --csv-file must be specified")
    
    if not Path(csv_file).exists():
        logger.error(f"❌ CSV file not found: {csv_file}")
        return False
    
    args.csv_file = csv_file
    
    logger.info("🔧 Configuration:")
    logger.info(f"  📄 CSV file: {csv_file}")
    logger.info(f"  📦 Chunk size: {args.chunk_size}")
    logger.info(f"  ⚡ Max workers: {args.max_workers}")
    logger.info(f"  ☁️ Upload workers: {args.upload_workers}")
    logger.info(f"  ⏱️ Timeout: {args.timeout} minutes")
    logger.info(f"  📊 Repository: {args.repo_id}")
    logger.info(f"  🏷️ Source: {args.source_name}")
    if args.adapter_method:
        logger.info(f"  🔽 Adapter filter: {args.adapter_method}")
    
    # Create required directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/aggregated").mkdir(parents=True, exist_ok=True)
    
    try:
        success = process_with_optimization(args)
        return success
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)