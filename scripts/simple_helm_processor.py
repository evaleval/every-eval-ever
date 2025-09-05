#!/usr/bin/env python3
"""
Simplified HELM processor with global deduplication.

This script implements a much cleaner approach:
1. Read all existing task IDs from repository parquet files
2. Compare with all scraped task IDs across all benchmarks
3. Split remaining tasks into chunks and process them in parallel
4. Use source+benchmark+task_id as unique identifier

Usage:
    python scripts/simple_helm_processor.py --repo-id evaleval/every_eval_ever --chunk-size 100 --max-workers 4
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set, Dict
import logging
import pandas as pd

from huggingface_hub import HfApi

# Try to import datasets library for reading existing data
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging

# Set up logging
logger = setup_logging("simple_helm.log", "SimpleHELM")


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


def get_existing_task_ids(repo_id: str) -> Set[str]:
    """
    Read all existing task IDs from the repository.
    Returns set of unique identifiers in format: source+benchmark+task_id
    """
    existing_tasks = set()
    
    if not DATASETS_AVAILABLE:
        logger.warning("⚠️ datasets library not available - skipping duplicate detection")
        return existing_tasks
    
    try:
        logger.info("🔍 Reading existing task IDs from repository...")
        dataset = load_dataset(repo_id, streaming=True, split='train')
        
        count = 0
        for example in dataset:
            # Create unique identifier: source+benchmark+task_id
            source = example.get('source', 'unknown')
            benchmark = example.get('benchmark', 'unknown') 
            task_id = example.get('task_id', '')
            
            if task_id:
                unique_id = f"{source}+{benchmark}+{task_id}"
                existing_tasks.add(unique_id)
                count += 1
                
                if count % 10000 == 0:
                    logger.info(f"🔄 Processed {count:,} existing records, found {len(existing_tasks):,} unique tasks")
        
        logger.info(f"✅ Found {len(existing_tasks):,} existing unique task IDs in repository")
        return existing_tasks
        
    except Exception as e:
        logger.info(f"📝 Repository appears to be empty or inaccessible: {e}")
        return existing_tasks


def collect_all_tasks_from_benchmarks() -> Dict[str, List[str]]:
    """
    Collect all task IDs from all benchmark CSV files.
    Returns dict: {benchmark: [task_ids]}
    """
    logger.info("📊 Collecting tasks from all benchmark CSV files...")
    
    all_tasks = {}
    benchmarks = ['lite', 'mmlu', 'classic']
    
    for benchmark in benchmarks:
        csv_file = f"data/benchmark_lines/helm_{benchmark}.csv"
        csv_path = Path(csv_file)
        
        if not csv_path.exists():
            logger.info(f"📥 CSV for benchmark '{benchmark}' not found. Generating...")
            try:
                # Create the benchmark_lines directory if it doesn't exist
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Import web_scraper to generate the CSV
                from src.sources.helm.web_scraper import main as create_csv_main
                asyncio.run(create_csv_main(benchmark=benchmark, output_dir=str(csv_path.parent)))
                logger.info(f"✅ Successfully created CSV: {csv_file}")
            except Exception as e:
                logger.error(f"❌ Failed to create CSV for benchmark '{benchmark}': {e}")
                continue
        
        # Read tasks from CSV
        try:
            df = pd.read_csv(csv_path, dtype=str, low_memory=False)
            
            # Check which column contains the task identifiers
            if 'task' in df.columns:
                tasks = df['task'].tolist()
            elif 'Run' in df.columns:
                tasks = df['Run'].tolist()
            else:
                tasks = df.iloc[:, 0].tolist()
                logger.warning(f"⚠️ No 'task' or 'Run' column found in {benchmark}, using first column")
            
            # Filter out NaN values
            tasks = [task for task in tasks if pd.notna(task)]
            all_tasks[benchmark] = tasks
            logger.info(f"📋 {benchmark}: {len(tasks)} tasks")
            
        except Exception as e:
            logger.error(f"❌ Error reading {csv_file}: {e}")
            all_tasks[benchmark] = []
    
    total_tasks = sum(len(tasks) for tasks in all_tasks.values())
    logger.info(f"📊 Total tasks across all benchmarks: {total_tasks:,}")
    return all_tasks


def filter_new_tasks(all_tasks: Dict[str, List[str]], existing_tasks: Set[str], source: str = "helm") -> Dict[str, List[str]]:
    """
    Filter out tasks that already exist in the repository.
    Returns dict: {benchmark: [new_task_ids]}
    """
    logger.info("🔍 Filtering out already processed tasks...")
    
    new_tasks = {}
    total_existing = 0
    total_new = 0
    
    for benchmark, tasks in all_tasks.items():
        new_benchmark_tasks = []
        existing_count = 0
        
        for task in tasks:
            unique_id = f"{source}+{benchmark}+{task}"
            if unique_id not in existing_tasks:
                new_benchmark_tasks.append(task)
            else:
                existing_count += 1
        
        new_tasks[benchmark] = new_benchmark_tasks
        total_existing += existing_count
        total_new += len(new_benchmark_tasks)
        
        logger.info(f"📋 {benchmark}: {len(new_benchmark_tasks)} new tasks ({existing_count} already processed)")
    
    logger.info(f"✅ Total: {total_new:,} new tasks, {total_existing:,} already processed")
    return new_tasks


def create_global_chunks(new_tasks: Dict[str, List[str]], chunk_size: int) -> List[Dict]:
    """
    Create chunks from all new tasks across all benchmarks.
    Returns list of chunk dicts: [{'benchmark': str, 'tasks': [str], 'chunk_id': int}]
    """
    logger.info(f"📦 Creating global chunks (size: {chunk_size})...")
    
    # Flatten all tasks with their benchmark info
    all_task_items = []
    for benchmark, tasks in new_tasks.items():
        for task in tasks:
            all_task_items.append({'benchmark': benchmark, 'task': task})
    
    # Create chunks
    chunks = []
    for i in range(0, len(all_task_items), chunk_size):
        chunk_items = all_task_items[i:i + chunk_size]
        
        # Group by benchmark within chunk for easier processing
        chunk_data = {}
        for item in chunk_items:
            benchmark = item['benchmark']
            if benchmark not in chunk_data:
                chunk_data[benchmark] = []
            chunk_data[benchmark].append(item['task'])
        
        chunks.append({
            'chunk_id': i // chunk_size + 1,
            'benchmarks': chunk_data,
            'total_tasks': len(chunk_items)
        })
    
    logger.info(f"📦 Created {len(chunks)} chunks from {len(all_task_items)} total tasks")
    return chunks


def process_chunk(chunk: Dict, workers: int, source: str = "helm") -> str:
    """Process a chunk of tasks and return the output file path."""
    chunk_id = chunk['chunk_id']
    benchmarks = chunk['benchmarks']
    total_tasks = chunk['total_tasks']
    
    logger.info(f"🔄 Processing chunk {chunk_id} with {total_tasks} tasks across {len(benchmarks)} benchmarks")
    
    # Import required functions
    from src.sources.helm.processor import process_line
    from src.sources.helm.downloader import download_tasks
    
    # Create chunk output directory  
    chunk_dir = Path(f"data/processed/chunk_{chunk_id:04d}")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    downloads_dir = Path("data/downloads")
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up any previous downloads to save space
    import shutil
    if downloads_dir.exists():
        shutil.rmtree(downloads_dir, ignore_errors=True)
        downloads_dir.mkdir(parents=True, exist_ok=True)
    
    all_processed_files = []
    task_file_mapping = {}
    
    # Process each benchmark in the chunk
    for benchmark, tasks in benchmarks.items():
        logger.info(f"📥 Processing {len(tasks)} {benchmark} tasks in chunk {chunk_id}")
        
        # Download tasks for this benchmark
        try:
            downloaded_files = download_tasks(
                tasks=tasks,
                output_dir=str(downloads_dir),
                benchmark=benchmark,
                overwrite=False,
                show_progress=False
            )
            logger.info(f"✅ Downloaded {len(downloaded_files)} {benchmark} tasks")
        except Exception as e:
            logger.error(f"❌ Failed to download {benchmark} tasks: {e}")
            continue
        
        # Process tasks 
        def process_single_task(task_info):
            i, task = task_info
            try:
                result = process_line(
                    line=task,
                    output_dir=str(chunk_dir),
                    benchmark=benchmark,
                    downloads_dir=str(downloads_dir),
                    keep_temp_files=False,
                    overwrite=False,
                    show_progress=False
                )
                
                if result and result.get("status") == "success":
                    output_file = result.get("converted_file")
                    if output_file and Path(output_file).exists():
                        return output_file, task, benchmark
                    
                return None, task, benchmark
                
            except Exception as e:
                logger.error(f"❌ Task processing error: {e}")
                return None, task, benchmark
        
        # Process with limited parallelism
        max_parallel = min(workers, 4)
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            task_infos = [(i+1, task) for i, task in enumerate(tasks)]
            results = list(executor.map(process_single_task, task_infos))
        
        # Collect successful results
        for output_file, task, task_benchmark in results:
            if output_file:
                all_processed_files.append(output_file)
                task_file_mapping[output_file] = {
                    'task': task, 
                    'benchmark': task_benchmark
                }
        
        # Clean up downloads for this benchmark to save space
        benchmark_download_pattern = downloads_dir.glob(f"*{benchmark}*")
        for download_file in benchmark_download_pattern:
            try:
                if download_file.is_file():
                    download_file.unlink()
                elif download_file.is_dir():
                    shutil.rmtree(download_file, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors
    
    logger.info(f"📊 Chunk {chunk_id}: {len(all_processed_files)} successful tasks")
    
    if not all_processed_files:
        logger.error(f"❌ No files processed in chunk {chunk_id}")
        return None
    
    # Combine all processed files into one parquet
    try:
        combined_data = []
        
        for file_path in all_processed_files:
            df = pd.read_csv(file_path)
            
            # Add metadata
            file_info = task_file_mapping[file_path]
            task_id = file_info['task']
            benchmark = file_info['benchmark']
            
            # Create unique identifier: source+benchmark+task_id
            df['unique_id'] = f"{source}+{benchmark}+{task_id}"
            df['task_id'] = task_id
            df['source'] = source
            df['benchmark'] = benchmark
            df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            df['processing_date'] = datetime.now().strftime('%Y-%m-%d')
            
            logger.debug(f"📝 Created unique_id: {source}+{benchmark}+{task_id[:50]}{'...' if len(task_id) > 50 else ''}")
            
            combined_data.append(df)
        
        # Combine all dataframes
        combined_df = pd.concat(combined_data, ignore_index=True, copy=False, sort=False)
        
        # Free up memory from individual dataframes
        del combined_data
        
        # Reorder columns to put metadata first
        metadata_cols = ['unique_id', 'task_id', 'source', 'benchmark', 'timestamp', 'processing_date']
        other_cols = [col for col in combined_df.columns if col not in metadata_cols]
        combined_df = combined_df[metadata_cols + other_cols]
        
        # Create output file
        agg_dir = Path("data/aggregated")
        agg_dir.mkdir(parents=True, exist_ok=True)
        output_file = agg_dir / f"chunk_{chunk_id:04d}.parquet"
        
        combined_df.to_parquet(output_file, compression='snappy', index=False, engine='pyarrow')
        
        # Clean up all processed CSV files immediately to save space
        for file_path in all_processed_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors
        
        # Clean up chunk directory
        try:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors
        
        logger.info(f"✅ Created chunk parquet: {output_file} ({len(combined_df)} entries)")
        return str(output_file)
        
    except Exception as e:
        logger.error(f"❌ Error creating chunk parquet: {e}")
        return None


def upload_file(api: HfApi, local_path: str, repo_id: str, chunk_id: int, cleanup: bool = True) -> bool:
    """Upload a file to HuggingFace."""
    try:
        file_path = Path(local_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        remote_name = f"data-{chunk_id:05d}.parquet"
        
        logger.info(f"☁️ Uploading {file_path.name} as {remote_name} ({file_size_mb:.1f}MB)")
        
        start_time = time.time()
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="dataset",
            run_as_future=False
        )
        
        upload_time = time.time() - start_time
        logger.info(f"✅ Uploaded {remote_name} in {upload_time:.1f}s")
        
        if cleanup:
            file_path.unlink()
            logger.info(f"🧹 Cleaned up local file")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simplified HELM processor with global deduplication")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument("--chunk-size", type=int, default=100, help="Tasks per chunk (default: 100)")
    parser.add_argument("--max-workers", type=int, default=4, help="Workers per chunk (default: 4)")
    parser.add_argument("--source-name", type=str, default="helm", help="Source name")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up files after upload")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting simplified HELM processing")
    logger.info(f"🔧 Config: chunk_size={args.chunk_size}, workers={args.max_workers}")
    
    # Setup HuggingFace API
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("❌ HF_TOKEN environment variable not set")
        return False
    
    api = setup_hf_api(hf_token, args.repo_id)
    
    # Step 1: Get existing task IDs from repository
    existing_tasks = get_existing_task_ids(args.repo_id)
    
    # Step 2: Collect all tasks from benchmark CSVs
    all_tasks = collect_all_tasks_from_benchmarks()
    
    # Step 3: Filter out already processed tasks
    new_tasks = filter_new_tasks(all_tasks, existing_tasks, args.source_name)
    
    # Check if there's anything to do
    total_new = sum(len(tasks) for tasks in new_tasks.values())
    if total_new == 0:
        logger.info("✅ No new tasks to process - all benchmarks are up to date!")
        return True
    
    # Step 4: Create global chunks
    chunks = create_global_chunks(new_tasks, args.chunk_size)
    
    # Step 5: Process chunks in parallel
    logger.info(f"🚀 Processing {len(chunks)} chunks with up to {args.max_workers} workers")
    
    successful_uploads = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all chunk processing jobs
        future_to_chunk = {
            executor.submit(process_chunk, chunk, args.max_workers, args.source_name): chunk 
            for chunk in chunks
        }
        
        # Process completed chunks
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            chunk_id = chunk['chunk_id']
            
            try:
                chunk_file = future.result()
                if chunk_file:
                    logger.info(f"✅ Chunk {chunk_id} completed: {chunk_file}")
                    
                    # Upload immediately
                    if upload_file(api, chunk_file, args.repo_id, chunk_id, not args.no_cleanup):
                        successful_uploads += 1
                        logger.info(f"📤 Chunk {chunk_id} uploaded successfully")
                    else:
                        logger.error(f"❌ Upload failed for chunk {chunk_id}")
                else:
                    logger.error(f"❌ Chunk {chunk_id} processing failed")
                    
            except Exception as e:
                logger.error(f"❌ Chunk {chunk_id} failed with error: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Processing complete!")
    logger.info(f"✅ Successful uploads: {successful_uploads}/{len(chunks)}")
    logger.info(f"📊 Total new tasks processed: {total_new:,}")
    logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
    logger.info("=" * 60)
    
    return successful_uploads == len(chunks)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
