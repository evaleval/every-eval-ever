#!/usr/bin/env python3
"""
Quick test to verify that the processor will start from the correct chunk number 
when continuing from an existing HF dataset.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / 'scripts'))

def test_chunk_continuation():
    """Test chunk numbering based on HF dataset files."""
    
    # Simulate existing files in HF dataset
    existing_files = [
        'data-00001.parquet',  # Would be chunk 1
        'data-00002.parquet',  # Would be chunk 2  
        'README.md',
        'config.json'
    ]
    
    # Extract part numbers (same logic as in optimized_helm_processor.py)
    part_numbers = []
    for f in existing_files:
        try:
            if f.startswith('data-') and f.endswith('.parquet'):
                num = int(f.split('-')[1].split('.')[0])
                part_numbers.append(num)
        except ValueError:
            continue
    
    start_number = max(part_numbers) + 1 if part_numbers else 1
    
    print(f"📊 Existing HF files: {existing_files}")
    print(f"📊 Extracted part numbers: {part_numbers}")
    print(f"📊 Next chunk/part number will be: {start_number}")
    print(f"📁 This means chunk directory will be: chunk_{start_number:04d}")
    
    # Verify this matches expectation
    expected = 3  # Since we have data-00001.parquet and data-00002.parquet
    if start_number == expected:
        print(f"✅ Correct! Will start from chunk {start_number} as expected")
    else:
        print(f"❌ Wrong! Expected {expected}, got {start_number}")

if __name__ == "__main__":
    test_chunk_continuation()
