#!/usr/bin/env python3
"""
Simple test to check if HELM data download works.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_download():
    """Test basic download functionality."""
    try:
        from src.sources.helm.downloader import download_tasks
        
        # Test with one Classic task (bAbI QA)
        task = "babi_qa:task=15,model=together_opt-175b"
        output_dir = "data/downloads"
        benchmark = "classic"
        
        print(f"🧪 Testing download of: {task}")
        print(f"📂 Output directory: {output_dir}")
        print(f"🏷️  Benchmark: {benchmark}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Call download_tasks with correct arguments
        result = download_tasks([task], output_dir, benchmark)
        
        print(f"✅ Download result: {result}")
        
        if result:
            for file_path in result:
                if Path(file_path).exists():
                    print(f"✅ File exists: {file_path}")
                else:
                    print(f"❌ File missing: {file_path}")
        else:
            print("❌ No files downloaded")
            
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_download()
