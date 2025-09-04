#!/usr/bin/env python3
"""
Test to verify chunk numbering is unique across benchmarks.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def simulate_chunk_numbering():
    """Test the chunk numbering logic without actually processing data."""
    
    # Simulate existing files in repository
    test_cases = [
        {
            "name": "Empty repository",
            "existing_files": [],
            "expected_start": 1
        },
        {
            "name": "Repository with some chunks",
            "existing_files": ["data-00001.parquet", "data-00002.parquet", "data-00005.parquet"],
            "expected_start": 6
        },
        {
            "name": "Repository with many chunks",
            "existing_files": [f"data-{i:05d}.parquet" for i in range(1, 15)],
            "expected_start": 15
        }
    ]
    
    print("🧪 Testing chunk numbering logic")
    
    for test_case in test_cases:
        print(f"\n📋 Test: {test_case['name']}")
        print(f"📁 Existing files: {test_case['existing_files']}")
        
        # Simulate the logic from optimized_helm_processor.py
        existing_files = set(test_case['existing_files'])
        
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
        
        print(f"🔢 Calculated start number: {start_number}")
        print(f"✅ Expected: {test_case['expected_start']}, Got: {start_number}")
        
        if start_number == test_case['expected_start']:
            print("✅ PASS")
        else:
            print("❌ FAIL")

def check_manifest_chunk_conflicts():
    """Check if there are chunk number conflicts in existing manifests."""
    print("\n🔍 Checking for chunk number conflicts in manifests...")
    
    manifest_dir = Path("data/aggregated")
    if not manifest_dir.exists():
        print("📁 No manifest directory found")
        return
    
    all_chunk_mappings = {}  # chunk_number -> {file, benchmark}
    
    for manifest_file in manifest_dir.glob("manifest_helm_*.json"):
        benchmark = manifest_file.stem.split("_")[-1]  # Extract benchmark name
        print(f"\n📋 Checking manifest: {manifest_file} (benchmark: {benchmark})")
        
        try:
            import json
            with open(manifest_file, 'r') as f:
                manifest_data = json.load(f)
            
            files = manifest_data.get('files', [])
            print(f"📁 Found {len(files)} files in manifest")
            
            for file_info in files:
                chunk_num = file_info.get('chunk_number')
                remote_name = file_info.get('remote_name')
                
                if chunk_num in all_chunk_mappings:
                    existing = all_chunk_mappings[chunk_num]
                    print(f"⚠️ CONFLICT: Chunk {chunk_num} appears in multiple benchmarks:")
                    print(f"   {existing['benchmark']}: {existing['file']}")
                    print(f"   {benchmark}: {remote_name}")
                else:
                    all_chunk_mappings[chunk_num] = {
                        'file': remote_name,
                        'benchmark': benchmark
                    }
                    print(f"✅ Chunk {chunk_num}: {remote_name} ({benchmark})")
                    
        except Exception as e:
            print(f"❌ Error reading manifest {manifest_file}: {e}")
    
    print(f"\n📊 Summary: {len(all_chunk_mappings)} unique chunk numbers found")
    
    # Check for gaps
    if all_chunk_mappings:
        chunk_numbers = sorted(all_chunk_mappings.keys())
        expected_range = range(1, max(chunk_numbers) + 1)
        missing = [i for i in expected_range if i not in chunk_numbers]
        
        if missing:
            print(f"⚠️ Missing chunk numbers: {missing}")
        else:
            print("✅ No gaps in chunk numbering")

if __name__ == "__main__":
    simulate_chunk_numbering()
    check_manifest_chunk_conflicts()
