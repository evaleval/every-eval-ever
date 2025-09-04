#!/usr/bin/env python3
"""
Optimized HELM processor with chunked processing and parallel uploads.

This script modifies the existing HELM processing flow to:
1. Process tasks in configurable chunks (default: 100 tasks)
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
    python scripts/optimized_helm_processor.py --benchmark lite --chunk-size 100 --max-workers 4 --repo-id evaleval/every_eval_ever
"""

import argparse
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

# Set up logging
logger = setup_logging("optimized_helm.log", "OptimizedHELM")


def setup_hf_api(token: str, repo_id: str) -> HfApi:
    """Initialize HuggingFace API."""
    if not token:
        raise ValueError('HF_TOKEN environment variable required')

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type='dataset', token=token)
        logger.info(f'✅ Dataset repo ready: {repo_id}')
    except Exception as e:
        logger.info(f'📁 Repo setup: {e}')

    return api


def get_next_part_number(api: HfApi, repo_id: str) -> int:
    """Find the next available part number."""
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')
        part_numbers = []
        
        for file in repo_files:
            if file.startswith('data-') and file.endswith('.parquet'):
                try:
                    part_num = int(file.split('-')[1].split('.')[0])
                    part_numbers.append(part_num)
                except (ValueError, IndexError):
                    continue
        
        next_part = max(part_numbers, default=0) + 1
        logger.info(f'📈 Next part number: {next_part}')
        return next_part
        
    except Exception as e:
        logger.warning(f'⚠️ Error checking files, starting from part 1: {e}')
        return 1


def read_tasks_chunked(csv_file: str, chunk_size: int, adapter_method: str = None) -> List[List[str]]:
    """Read tasks from CSV and split into chunks."""
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"📊 CSV contains {len(df)} total tasks")
        
        if adapter_method and 'Adapter method' in df.columns:
            df = df[df['Adapter method'] == adapter_method]
            logger.info(f"🔍 Filtered to {len(df)} tasks with adapter method: {adapter_method}")
        
        tasks = df['Run'].dropna().tolist()
        
        # Split into chunks
        chunks = [tasks[i:i+chunk_size] for i in range(0, len(tasks), chunk_size)]
        logger.info(f"📦 Split {len(tasks)} tasks into {len(chunks)} chunks of ~{chunk_size} tasks each")
        
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Error reading CSV: {e}")
        return []


def process_chunk(chunk_tasks: List[str], chunk_id: int, benchmark: str, source_name: str, 
                 max_workers: int = 2) -> Path:
    """Process a single chunk of tasks."""
    chunk_start = time.time()
    logger.info(f"\n🔄 Processing chunk {chunk_id} with {len(chunk_tasks)} tasks")
    
    # Create a temporary CSV file for this chunk
    chunk_dir = Path(f"data/chunks/chunk_{chunk_id}")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_csv = chunk_dir / f"tasks.csv"
    chunk_df = pd.DataFrame({
        'Run': chunk_tasks,
        'Model': [''] * len(chunk_tasks),  # Add required columns
        'Groups': [''] * len(chunk_tasks),
        'Adapter method': [''] * len(chunk_tasks),
        'Subject / Task': [''] * len(chunk_tasks)
    })
    chunk_df.to_csv(chunk_csv, index=False)
    
    # Process with HELM processor
    helm_cmd = [
        sys.executable, "main.py", source_name,
        "--benchmark", benchmark,
        "--max-workers", str(max_workers),
        "--overwrite"
    ]
    
    # Set CSV file for this chunk
    os.environ['HELM_CSV_FILE'] = str(chunk_csv)
    
    try:
        logger.info(f"🚀 Running HELM processor for chunk {chunk_id}")
        result = subprocess.run(
            helm_cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes per chunk
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode != 0:
            logger.error(f"❌ HELM processing failed for chunk {chunk_id}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return None
            
        logger.info(f"✅ HELM processing completed for chunk {chunk_id}")
        
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Chunk {chunk_id} processing timed out")
        return None
    
    # Create aggregated parquet file for this chunk
    processed_dir = Path(f"data/processed/{benchmark}")
    csv_files = list(processed_dir.glob("**/*_converted.csv"))
    
    if not csv_files:
        logger.warning(f"⚠️ No CSV files found for chunk {chunk_id}")
        return None
    
    # Filter CSV files created after chunk processing started
    recent_csv_files = [f for f in csv_files if f.stat().st_mtime > chunk_start]
    
    if not recent_csv_files:
        logger.warning(f"⚠️ No recent CSV files found for chunk {chunk_id}")
        recent_csv_files = csv_files[-len(chunk_tasks):]  # Take last N files
    
    logger.info(f"📊 Aggregating {len(recent_csv_files)} CSV files for chunk {chunk_id}")
    
    # Aggregate into single parquet file
    output_path = Path("data/aggregated") / f"chunk_{chunk_id:04d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        aggregated_data = []
        
        for csv_file in recent_csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Add metadata
                df['source'] = source_name.upper()
                df['processed_at'] = datetime.now(timezone.utc).isoformat()
                df['benchmark'] = benchmark
                df['chunk_id'] = chunk_id
                df['aggregation_timestamp'] = datetime.now(timezone.utc).isoformat()
                df['pipeline_stage'] = 'chunk_processing'
                
                aggregated_data.append(df)
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing {csv_file}: {e}")
        
        if aggregated_data:
            final_df = pd.concat(aggregated_data, ignore_index=True)
            
            # Write to parquet with compression
            final_df.to_parquet(output_path, compression='snappy', index=False)
            
            chunk_duration = time.time() - chunk_start
            logger.info(f"✅ Created chunk parquet: {output_path} ({len(final_df)} entries, {chunk_duration:.1f}s)")
            
            # Clean up temporary files
            import shutil
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir)
            
            # Optionally clean up processed CSV files to save space
            for csv_file in recent_csv_files:
                try:
                    csv_file.unlink()
                except Exception as e:
                    logger.warning(f"⚠️ Could not remove {csv_file}: {e}")
            
            return output_path
        
    except Exception as e:
        logger.error(f"❌ Error creating parquet for chunk {chunk_id}: {e}")
        return None


def upload_chunk(api: HfApi, repo_id: str, parquet_path: Path, part_number: int) -> bool:
    """Upload a chunk parquet file to HuggingFace."""
    upload_start = time.time()
    target_filename = f"data-{part_number:05d}.parquet"
    
    try:
        file_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"☁️ Uploading {parquet_path.name} as {target_filename} ({file_size:.1f}MB)")
        
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=target_filename,
            repo_id=repo_id,
            repo_type='dataset'
        )
        
        upload_duration = time.time() - upload_start
        upload_speed = file_size / upload_duration if upload_duration > 0 else 0
        
        logger.info(f"✅ Uploaded {target_filename} in {upload_duration:.1f}s ({upload_speed:.1f}MB/s)")
        
        # Clean up local file after successful upload
        parquet_path.unlink()
        logger.info(f"🧹 Cleaned up local file: {parquet_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Upload failed for {target_filename}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Optimized HELM processor with chunked uploads")
    parser.add_argument("--benchmark", default="lite", help="Benchmark to process")
    parser.add_argument("--csv-file", help="CSV file with tasks")
    parser.add_argument("--adapter-method", help="Filter by adapter method")
    parser.add_argument("--chunk-size", type=int, default=100, help="Tasks per chunk")
    parser.add_argument("--max-workers", type=int, default=2, help="Workers per chunk")
    parser.add_argument("--upload-workers", type=int, default=2, help="Parallel upload workers")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    parser.add_argument("--source-name", default="helm", help="Source name")
    
    args = parser.parse_args()
    
    # Auto-detect CSV file if not provided
    if not args.csv_file:
        # Use absolute path relative to project root
        benchmark_lines_dir = project_root / "data" / "benchmark_lines"
        
        # Try HELM naming convention first: helm_{benchmark}.csv
        csv_path = benchmark_lines_dir / f"helm_{args.benchmark}.csv"
        if not csv_path.exists():
            # Fallback to simple naming: {benchmark}.csv
            csv_path = benchmark_lines_dir / f"{args.benchmark}.csv"
        
        # If CSV file doesn't exist, generate it using web scraper
        if not csv_path.exists():
            logger.info(f"📥 CSV file not found, generating it by scraping HELM data for benchmark: {args.benchmark}")
            try:
                # Import and run the web scraper to create the CSV
                import asyncio
                import sys
                sys.path.insert(0, str(project_root))
                from src.sources.helm.web_scraper import main as create_csv_main
                
                # Create the benchmark_lines directory if it doesn't exist
                benchmark_lines_dir.mkdir(parents=True, exist_ok=True)
                
                # Run the web scraper
                asyncio.run(create_csv_main(benchmark=args.benchmark, output_dir=str(benchmark_lines_dir)))
                
                # Check if the file was created successfully
                csv_path = benchmark_lines_dir / f"helm_{args.benchmark}.csv"
                if csv_path.exists():
                    logger.info(f"✅ Successfully generated CSV: {csv_path}")
                else:
                    logger.error(f"❌ CSV generation failed - file not found after scraping")
                    return 1
                    
            except Exception as e:
                logger.error(f"❌ Failed to generate CSV for benchmark '{args.benchmark}': {e}")
                return 1
        
        if csv_path.exists():
            args.csv_file = str(csv_path)
            logger.info(f"📄 Using CSV file: {csv_path}")
        else:
            logger.error(f"❌ CSV file not found. Tried:")
            logger.error(f"   - {benchmark_lines_dir}/helm_{args.benchmark}.csv")
            logger.error(f"   - {benchmark_lines_dir}/{args.benchmark}.csv")
            logger.error(f"📁 Available files in {benchmark_lines_dir}:")
            if benchmark_lines_dir.exists():
                for file in benchmark_lines_dir.iterdir():
                    logger.error(f"   - {file.name}")
            else:
                logger.error(f"   Directory does not exist: {benchmark_lines_dir}")
            return 1
    
    # Set up HuggingFace API
    hf_token = os.getenv('HF_TOKEN')
    api = setup_hf_api(hf_token, args.repo_id)
    
    # Get starting part number
    next_part = get_next_part_number(api, args.repo_id)
    
    # Read and chunk tasks
    task_chunks = read_tasks_chunked(args.csv_file, args.chunk_size, args.adapter_method)
    
    if not task_chunks:
        logger.error("❌ No tasks to process")
        return 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Starting optimized processing")
    logger.info(f"📦 Processing {len(task_chunks)} chunks")
    logger.info(f"⚡ {args.max_workers} workers per chunk")
    logger.info(f"☁️ {args.upload_workers} upload workers")
    logger.info(f"{'='*60}")
    
    start_time = datetime.now()
    completed_chunks = 0
    total_entries = 0
    upload_executor = ThreadPoolExecutor(max_workers=args.upload_workers)
    upload_futures = []
    
    # Process chunks sequentially, upload in parallel
    for chunk_idx, chunk_tasks in enumerate(task_chunks, 1):
        logger.info(f"\n--- Chunk {chunk_idx}/{len(task_chunks)} ---")
        
        # Process chunk
        parquet_path = process_chunk(
            chunk_tasks, chunk_idx, args.benchmark, args.source_name, args.max_workers
        )
        
        if parquet_path and parquet_path.exists():
            # Submit upload task
            part_number = next_part + chunk_idx - 1
            upload_future = upload_executor.submit(upload_chunk, api, args.repo_id, parquet_path, part_number)
            upload_futures.append((chunk_idx, upload_future))
            
            completed_chunks += 1
            
            # Estimate entries (rough)
            try:
                df_sample = pd.read_parquet(parquet_path)
                total_entries += len(df_sample)
            except:
                total_entries += len(chunk_tasks)  # Rough estimate
            
            # Progress logging
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = completed_chunks / elapsed * 60 if elapsed > 0 else 0
            eta_minutes = (len(task_chunks) - completed_chunks) / rate if rate > 0 else 0
            
            logger.info(f"📈 Progress: {completed_chunks}/{len(task_chunks)} chunks "
                       f"({completed_chunks/len(task_chunks)*100:.1f}%) "
                       f"| ~{total_entries} entries | ETA: {eta_minutes:.1f}min")
        else:
            logger.warning(f"⚠️ Chunk {chunk_idx} failed")
    
    # Wait for uploads to complete
    logger.info(f"\n⏳ Waiting for {len(upload_futures)} uploads to complete...")
    
    successful_uploads = 0
    for chunk_idx, future in upload_futures:
        try:
            if future.result(timeout=300):
                successful_uploads += 1
        except Exception as e:
            logger.error(f"❌ Upload failed for chunk {chunk_idx}: {e}")
    
    # Final summary
    total_duration = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Processing complete!")
    logger.info(f"✅ Processed chunks: {completed_chunks}/{len(task_chunks)}")
    logger.info(f"☁️ Successful uploads: {successful_uploads}/{len(upload_futures)}")
    logger.info(f"📊 Total entries: ~{total_entries}")
    logger.info(f"⏱️ Total time: {total_duration/60:.1f} minutes")
    logger.info(f"📈 Rate: {total_entries/(total_duration/60):.1f} entries/minute")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
