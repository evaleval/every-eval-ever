#!/usr/bin/env python3
"""
Simple test for the benchmark parameter fix in optimized processor.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_benchmark_fix():
    """Test that process_line uses the correct benchmark parameter."""
    
    # Test task that we know exists
    task = "mmlu:subject=clinical_knowledge,method=multiple_choice_joint,model=01-ai_yi-6b,eval_split=test,groups=mmlu_clinical_knowledge"
    benchmark = "mmlu"
    
    print(f"🧪 Testing benchmark parameter fix")
    print(f"📝 Task: {task}")
    print(f"🏷️ Benchmark: {benchmark}")
    
    # Ensure the data exists (download if needed)
    try:
        from src.sources.helm.downloader import download_tasks
        
        downloads_dir = "data/downloads"
        os.makedirs(downloads_dir, exist_ok=True)
        
        print(f"📥 Ensuring task data is downloaded...")
        download_tasks([task], downloads_dir, benchmark)
        print(f"✅ Download complete")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False
    
    # Test the process_line function with correct benchmark
    try:
        from src.sources.helm.processor import process_line
        
        output_dir = "data/test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔄 Testing process_line with benchmark='{benchmark}'...")
        
        result = process_line(
            line=task,
            output_dir=output_dir,
            benchmark=benchmark,  # This should now be correct (not hardcoded "lite")
            downloads_dir=downloads_dir,
            keep_temp_files=False,
            overwrite=True,
            show_progress=True
        )
        
        print(f"📊 Result: {result}")
        
        if result and result.get("status") == "success":
            output_file = result.get("converted_file")
            if output_file and Path(output_file).exists():
                print(f"✅ Success! Output file created: {output_file}")
                return True
            else:
                print(f"❌ Output file not created: {output_file}")
                return False
        else:
            print(f"❌ Processing failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_benchmark_fix()
    sys.exit(0 if success else 1)
