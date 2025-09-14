#!/usr/bin/env python3
"""
One-time script to rename all .json files to .jsonl in HuggingFace repository.

This script downloads all JSON files from the HF repository, renames them locally,
and uploads them back as JSONL files, then deletes the old JSON files.

Usage:
    python scripts/rename_json_to_jsonl.py --repo-id evaleval/every_eval_ever
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, list_repo_files

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rename JSON files to JSONL in HuggingFace repository")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repository ID")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without actually doing it")
    
    args = parser.parse_args()
    
    # Check for HF token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("❌ HF_TOKEN environment variable required")
        return 1
    
    try:
        # Initialize HuggingFace API
        api = HfApi(token=hf_token)
        
        logger.info(f"🔍 Scanning repository {args.repo_id} for JSON files...")
        
        # Get list of all files in the repository
        files = list_repo_files(args.repo_id, repo_type="dataset")
        
        # Find all JSON files
        json_files = [f for f in files if f.endswith('.json')]
        
        logger.info(f"📊 Found {len(json_files)} JSON files to rename")
        
        if len(json_files) == 0:
            logger.info("✅ No JSON files found to rename")
            return 0
        
        # Show first few files as examples
        logger.info("📝 Examples of files to rename:")
        for i, file in enumerate(json_files[:10]):
            jsonl_file = file[:-5] + '.jsonl'  # Replace .json with .jsonl
            logger.info(f"  {file} -> {jsonl_file}")
        
        if len(json_files) > 10:
            logger.info(f"  ... and {len(json_files) - 10} more files")
        
        if args.dry_run:
            logger.info("🔍 DRY RUN - No files will be modified")
            return 0
        
        logger.info("🚀 Starting rename operation...")
        logger.info("⚠️  This operation will rename all JSON files to JSONL format")
        
        # Process files in batches to avoid overwhelming the API
        batch_size = 50
        successful_renames = 0
        failed_renames = 0
        
        for i in range(0, len(json_files), batch_size):
            batch = json_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(json_files) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} files)...")
            
            for json_file in batch:
                try:
                    jsonl_file = json_file[:-5] + '.jsonl'  # Replace .json with .jsonl
                    
                    # Download the JSON file content
                    logger.debug(f"📥 Downloading {json_file}...")
                    file_content = api.hf_hub_download(
                        repo_id=args.repo_id,
                        filename=json_file,
                        repo_type="dataset"
                    )
                    
                    # Read the content
                    with open(file_content, 'rb') as f:
                        content = f.read()
                    
                    # Upload as JSONL file
                    logger.debug(f"📤 Uploading as {jsonl_file}...")
                    api.upload_file(
                        path_or_fileobj=content,
                        path_in_repo=jsonl_file,
                        repo_id=args.repo_id,
                        repo_type="dataset",
                        commit_message=f"Rename {json_file} to {jsonl_file}"
                    )
                    
                    # Delete the old JSON file
                    logger.debug(f"🗑️  Deleting old {json_file}...")
                    api.delete_file(
                        path_in_repo=json_file,
                        repo_id=args.repo_id,
                        repo_type="dataset",
                        commit_message=f"Remove old JSON file: {json_file}"
                    )
                    
                    successful_renames += 1
                    logger.info(f"✅ Successfully renamed: {json_file} -> {jsonl_file}")
                    
                except Exception as e:
                    failed_renames += 1
                    logger.error(f"❌ Failed to rename {json_file}: {e}")
                    continue
            
            logger.info(f"📊 Batch {batch_num} completed. Success: {successful_renames}, Failed: {failed_renames}")
        
        logger.info(f"🎉 Rename operation completed!")
        logger.info(f"✅ Successfully renamed: {successful_renames} files")
        if failed_renames > 0:
            logger.warning(f"❌ Failed to rename: {failed_renames} files")
        
        logger.info(f"📋 Next steps:")
        logger.info(f"1. Verify the renamed files in the repository")
        logger.info(f"2. Test statistics generation with the new JSONL files")
        logger.info(f"3. Clear HuggingFace datasets cache if needed")
        
        return 0 if failed_renames == 0 else 1
        
    except Exception as e:
        logger.error(f"❌ Rename operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
