#!/usr/bin/env python3
"""
Chunked parallel HELM processor with immediate HuggingFace uploads.

This optimized script:
1. Processes tasks in smaller chunks (e.g., 50-100 tasks at a time)
2. Uses parallel processing for downloads within chunks
3. Immediately uploads completed chunks to HuggingFace
4. Cleans up local files after upload to save disk space
5. Continues processing while uploading in background

Benefits:
- Faster feedback and intermediate results
- Lower memory usage (smaller chunks)
- Parallel download + upload pipeline
- Fault tolerance (partial completion)
- Disk space efficiency

Usage:
    python scripts/chunked_parallel_upload.py --repo-id evaleval/every_eval_ever --benchmark lite --chunk-size 50 --max-workers 4
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import logging
import pandas as pd

from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.core.aggregator import process_csv_files

# Set up logging
logger = setup_logging("chunked_upload.log", "ChunkedUploader")


class ChunkedHELMProcessor:
    """Processes HELM data in chunks with parallel upload."""
    
    def __init__(self, repo_id: str, source_name: str = "helm", chunk_size: int = 50, 
                 max_workers: int = 4, upload_workers: int = 2):
        self.repo_id = repo_id
        self.source_name = source_name
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.upload_workers = upload_workers
        
        # Initialize HF API
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable required")
        
        self.api = self._setup_hf_api(hf_token)
        self.upload_executor = ThreadPoolExecutor(max_workers=upload_workers)
        
        # Track processing state
        self.completed_chunks = 0
        self.total_entries = 0
        self.start_time = datetime.now(timezone.utc)
        
    def _setup_hf_api(self, token: str) -> HfApi:
        """Initialize HuggingFace API."""
        logger.info("🔧 Setting up HuggingFace integration...")
        api = HfApi()
        
        try:
            api.create_repo(repo_id=self.repo_id, repo_type='dataset', token=token)
            logger.info(f'✅ Dataset repo ready: {self.repo_id}')
        except Exception as e:
            logger.info(f'📁 Repo already exists: {e}')
            
        return api
    
    def _get_next_part_number(self) -> int:
        """Find the next available part number."""
        try:
            repo_files = self.api.list_repo_files(repo_id=self.repo_id, repo_type='dataset')
            part_numbers = []
            
            for file in repo_files:
                if file.startswith('data-') and file.endswith('.parquet'):
                    try:
                        part_num = int(file.split('-')[1].split('.')[0])
                        part_numbers.append(part_num)
                    except (ValueError, IndexError):
                        continue
            
            return max(part_numbers, default=0) + 1
            
        except Exception as e:
            logger.warning(f'⚠️ Error checking existing files: {e}')
            return 1
    
    def _read_tasks_from_csv(self, csv_file: str, adapter_method: str = None) -> List[str]:
        """Read and filter tasks from CSV file."""
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"📊 CSV contains {len(df)} total tasks")
            
            if adapter_method and 'Adapter method' in df.columns:
                df = df[df['Adapter method'] == adapter_method]
                logger.info(f"🔍 Filtered to {len(df)} tasks with adapter method: {adapter_method}")
            
            tasks = df['Run'].dropna().tolist()
            logger.info(f"📄 Found {len(tasks)} valid tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"❌ Error reading CSV: {e}")
            return []
    
    def _process_chunk(self, chunk_tasks: List[str], chunk_id: int, benchmark: str) -> Path:
        """Process a single chunk of tasks."""
        chunk_start = time.time()
        logger.info(f"\n🔄 Processing chunk {chunk_id} ({len(chunk_tasks)} tasks)")
        
        # Create temporary directories for this chunk
        chunk_dir = Path(f"data/processed/{benchmark}/chunk_{chunk_id}")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Process tasks with parallel downloads
        helm_cmd = [
            sys.executable, "main.py", self.source_name,
            "--benchmark", benchmark,
            "--max-workers", str(self.max_workers),
            "--overwrite"
        ]
        
        # Create a temporary CSV file with just this chunk's tasks
        chunk_csv = chunk_dir / f"chunk_{chunk_id}_tasks.csv"
        chunk_df = pd.DataFrame({'Run': chunk_tasks})
        chunk_df.to_csv(chunk_csv, index=False)
        
        logger.info(f"📝 Created chunk CSV: {chunk_csv}")
        
        # Process the chunk
        try:
            logger.info(f"🚀 Running HELM processor for chunk {chunk_id}")
            result = subprocess.run(
                helm_cmd + ["--csv-file", str(chunk_csv)],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes per chunk
            )
            
            if result.returncode != 0:
                logger.error(f"❌ HELM processing failed for chunk {chunk_id}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Chunk {chunk_id} timed out")
            return None
        
        # Aggregate chunk into parquet
        output_path = Path("data/aggregated") / f"chunk_{chunk_id}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Find all CSV files created by this chunk
            csv_files = list(chunk_dir.glob("**/*_converted.csv"))
            
            if not csv_files:
                logger.warning(f"⚠️ No CSV files found for chunk {chunk_id}")
                return None
            
            logger.info(f"📊 Aggregating {len(csv_files)} CSV files for chunk {chunk_id}")
            
            # Use the aggregator to create parquet
            from src.core.aggregator import process_csv_files
            
            aggregated_data = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Add metadata columns
                    df['source'] = self.source_name.upper()
                    df['processed_at'] = datetime.now(timezone.utc).isoformat()
                    df['benchmark'] = benchmark
                    df['chunk_id'] = chunk_id
                    
                    aggregated_data.append(df)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing {csv_file}: {e}")
            
            if aggregated_data:
                final_df = pd.concat(aggregated_data, ignore_index=True)
                
                # Write to parquet
                table = pa.Table.from_pandas(final_df)
                pq.write_table(table, output_path)
                
                logger.info(f"✅ Created parquet file: {output_path} ({len(final_df)} entries)")
                
                # Update counters
                self.total_entries += len(final_df)
                
                # Clean up processed files to save space
                import shutil
                shutil.rmtree(chunk_dir)
                logger.info(f"🧹 Cleaned up chunk directory: {chunk_dir}")
                
                chunk_duration = time.time() - chunk_start
                logger.info(f"⏱️ Chunk {chunk_id} completed in {chunk_duration:.1f}s")
                
                return output_path
            
        except Exception as e:
            logger.error(f"❌ Error aggregating chunk {chunk_id}: {e}")
            return None
    
    def _upload_chunk(self, parquet_path: Path, part_number: int) -> bool:
        """Upload a chunk to HuggingFace."""
        upload_start = time.time()
        target_filename = f"data-{part_number:05d}.parquet"
        
        logger.info(f"☁️ Uploading {parquet_path.name} as {target_filename}")
        
        try:
            self.api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo=target_filename,
                repo_id=self.repo_id,
                repo_type='dataset'
            )
            
            upload_duration = time.time() - upload_start
            file_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
            
            logger.info(f"✅ Uploaded {target_filename} ({file_size:.1f}MB in {upload_duration:.1f}s)")
            
            # Clean up local file after successful upload
            parquet_path.unlink()
            logger.info(f"🧹 Cleaned up local file: {parquet_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Upload failed for {target_filename}: {e}")
            return False
    
    def process_benchmark(self, benchmark: str, csv_file: str, adapter_method: str = None):
        """Process entire benchmark in chunks with parallel uploads."""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting chunked processing for benchmark: {benchmark}")
        logger.info(f"📁 Chunk size: {self.chunk_size}")
        logger.info(f"⚡ Max workers: {self.max_workers}")
        logger.info(f"☁️ Upload workers: {self.upload_workers}")
        logger.info(f"{'='*60}")
        
        # Read all tasks
        all_tasks = self._read_tasks_from_csv(csv_file, adapter_method)
        if not all_tasks:
            logger.error("❌ No tasks to process")
            return
        
        # Split into chunks
        chunks = [all_tasks[i:i+self.chunk_size] for i in range(0, len(all_tasks), self.chunk_size)]
        logger.info(f"📦 Split {len(all_tasks)} tasks into {len(chunks)} chunks")
        
        # Get starting part number
        next_part = self._get_next_part_number()
        logger.info(f"📊 Starting from part number: {next_part}")
        
        # Process chunks with concurrent uploads
        upload_futures = []
        
        for chunk_idx, chunk_tasks in enumerate(chunks, 1):
            logger.info(f"\n--- Processing chunk {chunk_idx}/{len(chunks)} ---")
            
            # Process chunk
            parquet_path = self._process_chunk(chunk_tasks, chunk_idx, benchmark)
            
            if parquet_path and parquet_path.exists():
                # Submit upload task to background thread
                part_number = next_part + chunk_idx - 1
                upload_future = self.upload_executor.submit(self._upload_chunk, parquet_path, part_number)
                upload_futures.append((chunk_idx, upload_future))
                
                self.completed_chunks += 1
                
                # Log progress
                elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                rate = self.completed_chunks / elapsed * 60  # chunks per minute
                eta_minutes = (len(chunks) - self.completed_chunks) / rate if rate > 0 else 0
                
                logger.info(f"📈 Progress: {self.completed_chunks}/{len(chunks)} chunks "
                           f"({self.completed_chunks/len(chunks)*100:.1f}%) "
                           f"| {self.total_entries} total entries "
                           f"| ETA: {eta_minutes:.1f}min")
            else:
                logger.warning(f"⚠️ Chunk {chunk_idx} failed, skipping upload")
        
        # Wait for all uploads to complete
        logger.info(f"\n⏳ Waiting for {len(upload_futures)} uploads to complete...")
        
        successful_uploads = 0
        for chunk_idx, future in upload_futures:
            try:
                if future.result(timeout=300):  # 5 minutes per upload
                    successful_uploads += 1
            except Exception as e:
                logger.error(f"❌ Upload failed for chunk {chunk_idx}: {e}")
        
        # Final summary
        total_duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Processing complete for benchmark: {benchmark}")
        logger.info(f"✅ Successful chunks: {self.completed_chunks}/{len(chunks)}")
        logger.info(f"☁️ Successful uploads: {successful_uploads}/{len(upload_futures)}")
        logger.info(f"📊 Total entries processed: {self.total_entries}")
        logger.info(f"⏱️ Total duration: {total_duration/60:.1f} minutes")
        logger.info(f"📈 Processing rate: {self.total_entries/(total_duration/60):.1f} entries/minute")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Chunked parallel HELM processor")
    parser.add_argument("--repo-id", required=True, help="HuggingFace dataset repo ID")
    parser.add_argument("--benchmark", default="lite", help="Benchmark to process")
    parser.add_argument("--csv-file", help="CSV file with tasks (auto-detected if not provided)")
    parser.add_argument("--adapter-method", help="Filter by adapter method")
    parser.add_argument("--chunk-size", type=int, default=50, help="Tasks per chunk")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers for processing")
    parser.add_argument("--upload-workers", type=int, default=2, help="Parallel workers for uploads")
    parser.add_argument("--source-name", default="helm", help="Source name")
    
    args = parser.parse_args()
    
    # Auto-detect CSV file if not provided
    if not args.csv_file:
        csv_path = Path("data/benchmark_lines") / f"{args.benchmark}.csv"
        if csv_path.exists():
            args.csv_file = str(csv_path)
        else:
            logger.error(f"❌ CSV file not found: {csv_path}")
            return 1
    
    # Initialize processor
    processor = ChunkedHELMProcessor(
        repo_id=args.repo_id,
        source_name=args.source_name,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        upload_workers=args.upload_workers
    )
    
    # Process benchmark
    processor.process_benchmark(
        benchmark=args.benchmark,
        csv_file=args.csv_file,
        adapter_method=args.adapter_method
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
