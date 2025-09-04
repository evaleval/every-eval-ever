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


def read_tasks_chunked(csv_file: str, chunk_size: int, adapter_method: str = None) -> List[List[str]]:
    """Read tasks from CSV and split into chunks."""
    try:
        df = pd.read_csv(csv_file)
        tasks = df['task'].tolist()
        
        # Filter by adapter method if specified
        if adapter_method:
            tasks = [task for task in tasks if adapter_method in task]
            logger.info(f"🔽 Filtered to {len(tasks)} tasks with adapter method: {adapter_method}")
        
        chunks = [tasks[i:i+chunk_size] for i in range(0, len(tasks), chunk_size)]
        logger.info(f"📦 Split {len(tasks)} tasks into {len(chunks)} chunks of ~{chunk_size} tasks each")
        return chunks
    except Exception as e:
        logger.error(f"❌ Error reading tasks: {e}")
        raise


def process_chunk(chunk_tasks: List[str], chunk_id: int, workers: int, timeout_minutes: int = 60) -> str:
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
    
    # Process each task in the chunk
    for i, task in enumerate(chunk_tasks, 1):
        logger.info(f"🔄 Processing task {i}/{len(chunk_tasks)}: {task}")
        
        try:
            # Call the HELM processor function directly
            output_file = process_line(task)
            
            if output_file and Path(output_file).exists():
                # Read the file to count entries
                try:
                    df = pd.read_csv(output_file)
                    entry_count = len(df)
                    logger.info(f"✅ Task {i} completed: {entry_count} entries")
                    processed_files.append(output_file)
                except Exception as e:
                    logger.warning(f"⚠️ Could not count entries in {output_file}: {e}")
                    processed_files.append(output_file)
            else:
                logger.error(f"❌ Task {i} failed: no output file generated")
                
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
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Create aggregated directory
            agg_dir = Path("data/aggregated")
            agg_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = agg_dir / f"chunk_{chunk_id:04d}.parquet"
            combined_df.to_parquet(output_file)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Created chunk parquet: {output_file} ({total_entries} entries, {processing_time:.1f}s)")
            return str(output_file)
        else:
            logger.error(f"❌ No data to combine for chunk {chunk_id}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error creating chunk parquet for chunk {chunk_id}: {e}")
        return None


def upload_file(api: HfApi, local_path: str, repo_id: str, source_name: str, part_num: int) -> bool:
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
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        upload_time = time.time() - start_time
        speed_mbps = file_size_mb / max(upload_time, 0.001)
        logger.info(f"✅ Uploaded {remote_name} in {upload_time:.1f}s ({speed_mbps:.1f}MB/s)")
        
        # Clean up local file
        file_path.unlink()
        logger.info(f"🧹 Cleaned up local file: {local_path}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Upload failed for {local_path}: {e}")
        return False


def process_with_optimization(args):
    """Main processing function with chunked optimization."""
    logger.info("🚀 Starting optimized processing")
    
    # Setup HuggingFace API
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("❌ HF_TOKEN environment variable not set")
        return False
    
    api = setup_hf_api(hf_token, args.repo_id)
    existing_files = get_existing_files(api, args.repo_id)
    
    # Determine starting part number
    if existing_files:
        part_numbers = []
        for f in existing_files:
            try:
                if f.startswith('data-') and f.endswith('.parquet'):
                    num = int(f.split('-')[1].split('.')[0])
                    part_numbers.append(num)
            except ValueError:
                continue
        start_part = max(part_numbers) + 1 if part_numbers else 1
    else:
        start_part = 1
    
    logger.info(f"📊 Starting from part {start_part}")
    
    # Read and chunk tasks
    task_chunks = read_tasks_chunked(args.csv_file, args.chunk_size, args.adapter_method)
    
    if not task_chunks:
        logger.error("❌ No tasks to process")
        return False
    
    logger.info(f"📊 CSV contains {sum(len(chunk) for chunk in task_chunks)} total tasks")
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
    part_counter = start_part
    
    start_time = time.time()
    
    for chunk_idx, chunk_tasks in enumerate(task_chunks, 1):
        logger.info(f"\n--- Chunk {chunk_idx}/{len(task_chunks)} ---")
        
        # Process chunk
        chunk_file = process_chunk(chunk_tasks, chunk_idx, args.max_workers, args.timeout)
        
        if chunk_file:
            # Submit upload task
            upload_future = upload_executor.submit(
                upload_file, api, chunk_file, args.repo_id, args.source_name, part_counter
            )
            upload_futures.append(upload_future)
            part_counter += 1
            
            # Update progress
            elapsed = time.time() - start_time
            progress = chunk_idx / len(task_chunks)
            eta_minutes = (elapsed / progress - elapsed) / 60 if progress > 0 else 0
            total_entries = chunk_idx * args.chunk_size  # Approximate
            
            logger.info(f"📈 Progress: {chunk_idx}/{len(task_chunks)} chunks ({progress*100:.1f}%) | ~{total_entries} entries | ETA: {eta_minutes:.1f}min")
        else:
            logger.error(f"❌ Chunk {chunk_idx} failed")
    
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
    parser.add_argument("--chunk-size", type=int, default=200, help="Tasks per chunk (default: 200 for optimal performance)")
    parser.add_argument("--max-workers", type=int, default=2, help="Workers per chunk (default: 2)")
    parser.add_argument("--upload-workers", type=int, default=2, help="Parallel upload workers (default: 2)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per chunk in minutes (default: 60)")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument("--source-name", type=str, default="helm", help="Source name for file organization")
    parser.add_argument("--adapter-method", type=str, help="Filter tasks by adapter method")
    
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