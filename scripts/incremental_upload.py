#!/usr/bin/env python3
"""
Incremental HELM scraper and HuggingFace uploader.

This script handles the weekly automation for:
1. Finding the next part number from existing dataset
2. Processing benchmarks incrementally 
3. Uploading shards to HuggingFace immediately after processing
4. Cleaning up local files to save space

Usage:
    python scripts/incremental_upload.py --repo-id evaleval/every_eval_ever --benchmarks lite mmlu classic --source-name helm
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('incremental_upload.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def setup_hf_api(token: str, repo_id: str) -> HfApi:
    """Initialize HuggingFace API and ensure dataset repo exists."""
    logger.info("🔧 Setting up HuggingFace integration...")
    
    if not token:
        raise SystemExit('❌ HF_TOKEN not provided')

    api = HfApi()

    # Ensure dataset repo exists
    try:
        api.create_repo(repo_id=repo_id, repo_type='dataset', token=token)
        logger.info(f'✅ Dataset repo ready: {repo_id}')
    except Exception as e:
        logger.info(f'📁 Repo setup: {e}')

    return api


def find_next_part_number(api: HfApi, repo_id: str) -> int:
    """Find the highest existing part number in the dataset."""
    logger.info("📊 Checking for existing parts in dataset...")
    
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')
        part_numbers = []
        
        for file in repo_files:
            # Look for files matching pattern: data-XXXXX.parquet
            match = re.match(r'data-(\d+)\.parquet$', file)
            if match:
                part_numbers.append(int(match.group(1)))
        
        if part_numbers:
            next_part = max(part_numbers) + 1
            logger.info(f'📈 Found existing parts up to {max(part_numbers)}, starting from part {next_part}')
        else:
            next_part = 1
            logger.info('🆕 No existing parts found, starting from part 1')
            
        return next_part
        
    except Exception as e:
        logger.warning(f'⚠️ Error checking existing files, starting from part 1: {e}')
        return 1


def process_benchmark(benchmark: str, part_num: int, source_name: str, 
                     max_workers: int = 2, timeout: int = 3600) -> Path:
    """Process a single benchmark and create parquet file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"🔄 Processing benchmark: {benchmark}")
    logger.info(f"   Part number: {part_num}")
    logger.info(f"   Source: {source_name}")
    logger.info(f"{'='*50}")
    
    # Create shard filename with incremental numbering
    shard_name = f"data-{part_num:05d}.parquet"
    output_path = Path("data/aggregated") / shard_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Output file: {shard_name}")
    
    # Step 1: Run HELM processor to generate CSV files
    helm_cmd = [
        sys.executable, "main.py", source_name,
        "--benchmark", benchmark,
        "--overwrite",
        "--max-workers", str(max_workers)
    ]
    
    logger.info(f"🚀 Step 1 - Processing HELM data: {' '.join(helm_cmd)}")
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            helm_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(f"   [HELM] {line.rstrip()}")
        
        process.wait(timeout=timeout)
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, helm_cmd)
        
        logger.info(f"✅ HELM processing completed for {benchmark}")
        
        # Step 2: Run aggregator to create parquet file
        agg_cmd = [
            sys.executable, "-m", "src.core.aggregator",
            "--benchmark", benchmark,
            "--output-dir", str(output_path.parent)
        ]
        
        logger.info(f"🔄 Step 2 - Aggregating to parquet: {' '.join(agg_cmd)}")
        
        agg_process = subprocess.Popen(
            agg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream aggregator output
        for line in agg_process.stdout:
            logger.info(f"   [AGG] {line.rstrip()}")
        
        agg_process.wait(timeout=timeout)
        
        if agg_process.returncode != 0:
            raise subprocess.CalledProcessError(agg_process.returncode, agg_cmd)
        
        # The aggregator creates helm_{benchmark}_aggregated.parquet
        # We need to rename it to our desired format
        aggregated_file = output_path.parent / f"helm_{benchmark}_aggregated.parquet"
        if aggregated_file.exists():
            aggregated_file.rename(output_path)
            logger.info(f"✅ Renamed {aggregated_file.name} to {output_path.name}")
        else:
            raise FileNotFoundError(f"Expected aggregated file not found: {aggregated_file}")
        
        logger.info(f"✅ Successfully processed {benchmark}")
        return output_path
        
    except subprocess.TimeoutExpired:
        process.kill()
        logger.error(f"⏰ TIMEOUT: {benchmark} took longer than {timeout//60} minutes")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ ERROR processing {benchmark}: exit code {e.returncode}")
        raise
    except Exception as e:
        logger.error(f"❌ ERROR processing {benchmark}: {e}")
        raise


def upload_shard(api: HfApi, file_path: Path, repo_id: str, token: str):
    """Upload a parquet shard to HuggingFace and clean up local file."""
    if not file_path.exists():
        logger.warning(f"⚠️ Output file not found: {file_path}")
        return
    
    # Get file size for logging
    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
    
    logger.info(f"☁️ Uploading shard: {file_path.name} ({file_size:.1f} MB)")
    
    try:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=repo_id,
            repo_type='dataset',
            token=token,
        )
        logger.info(f"✅ Uploaded {file_path.name} to dataset")
        
        # Clean up local file to save space
        file_path.unlink()
        logger.info(f"🧹 Cleaned up local file {file_path.name}")
        
    except Exception as e:
        logger.error(f"❌ Failed to upload {file_path.name}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Incremental HELM scraper and uploader")
    parser.add_argument("--repo-id", required=True, help="HuggingFace dataset repo ID")
    parser.add_argument("--benchmarks", nargs="+", default=["lite", "mmlu", "classic"],
                       help="Benchmarks to process")
    parser.add_argument("--source-name", default="helm", help="Source name for file naming")
    parser.add_argument("--max-workers", type=int, default=2, help="Max workers for processing")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per benchmark (seconds)")
    
    args = parser.parse_args()
    
    # Get HF token from environment
    token = os.environ.get('HF_TOKEN')
    if not token:
        logger.error("❌ HF_TOKEN environment variable not set")
        return 1
    
    logger.info("🚀 Starting incremental HELM scrape and upload process...")
    
    try:
        # Setup HuggingFace API
        api = setup_hf_api(token, args.repo_id)
        
        # Find next part number
        next_part_num = find_next_part_number(api, args.repo_id)
        
        logger.info(f"🎯 Processing {len(args.benchmarks)} benchmarks: {args.benchmarks}")
        
        # Process each benchmark incrementally
        for i, benchmark in enumerate(args.benchmarks):
            current_part_num = next_part_num + i
            
            try:
                # Process benchmark
                output_path = process_benchmark(
                    benchmark, 
                    current_part_num, 
                    args.source_name,
                    args.max_workers,
                    args.timeout
                )
                
                # Upload immediately
                upload_shard(api, output_path, args.repo_id, token)
                
            except Exception as e:
                logger.error(f"❌ Failed to process benchmark {benchmark}: {e}")
                continue
        
        final_part = next_part_num + len(args.benchmarks) - 1
        logger.info(f"\n🎉 Incremental upload complete!")
        logger.info(f"📊 Processed parts {next_part_num} to {final_part}")
        logger.info(f"📈 Next run will start from part {final_part + 1}")
        
        return 0
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
