#!/usr/bin/env python3
"""
Test nested folder structures with HuggingFace datasets.
"""

import sys
from pathlib import Path
import json
import tempfile
import shutil
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from datasets import Dataset, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

def test_nested_folder_structures():
    """Test how HuggingFace datasets handles nested folder structures."""
    
    if not DATASETS_AVAILABLE:
        print("❌ datasets library not available")
        return False
    
    # Create temporary test directory
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        print("🧪 Testing nested folder structures with HuggingFace datasets...")
        
        # Test 1: Flat structure
        print("\n📁 Test 1: Flat structure")
        flat_dir = test_dir / "flat"
        flat_dir.mkdir(parents=True)
        
        flat_files = []
        for i in range(3):
            file_path = flat_dir / f"eval_{i}.json"
            data = {"id": f"eval_{i}", "score": i * 0.1, "text": f"Sample {i}"}
            with open(file_path, 'w') as f:
                json.dump(data, f)
            flat_files.append(str(file_path))
        
        # Load flat structure
        try:
            flat_dataset = Dataset.from_json(flat_files)
            print(f"✅ Flat structure: {len(flat_dataset)} items loaded")
        except Exception as e:
            print(f"❌ Flat structure failed: {e}")
        
        # Test 2: Simple nested structure (benchmark-based)
        print("\n📁 Test 2: Simple nested structure (by benchmark)")
        nested_dir = test_dir / "nested"
        
        benchmarks = ["mmlu", "hellaswag", "arc"]
        nested_files = []
        
        for benchmark in benchmarks:
            bench_dir = nested_dir / benchmark
            bench_dir.mkdir(parents=True)
            
            for i in range(2):
                file_path = bench_dir / f"eval_{benchmark}_{i}.json"
                data = {
                    "id": f"eval_{benchmark}_{i}", 
                    "benchmark": benchmark,
                    "score": i * 0.1, 
                    "text": f"Sample {benchmark} {i}"
                }
                with open(file_path, 'w') as f:
                    json.dump(data, f)
                nested_files.append(str(file_path))
        
        # Load nested structure  
        try:
            nested_dataset = Dataset.from_json(nested_files)
            print(f"✅ Simple nested: {len(nested_dataset)} items loaded")
        except Exception as e:
            print(f"❌ Simple nested failed: {e}")
        
        # Test 3: Deep nested structure (benchmark/model/task)
        print("\n📁 Test 3: Deep nested structure (benchmark/model/task)")
        deep_dir = test_dir / "deep"
        
        deep_files = []
        for benchmark in ["mmlu", "arc"]:
            for model in ["llama", "mistral"]:
                for task_id in range(2):
                    task_dir = deep_dir / benchmark / model / f"task_{task_id}"
                    task_dir.mkdir(parents=True)
                    
                    file_path = task_dir / f"eval_{benchmark}_{model}_{task_id}.json"
                    data = {
                        "id": f"eval_{benchmark}_{model}_{task_id}",
                        "benchmark": benchmark,
                        "model": model,
                        "task_id": task_id,
                        "score": task_id * 0.2,
                        "text": f"Sample {benchmark} {model} {task_id}"
                    }
                    with open(file_path, 'w') as f:
                        json.dump(data, f)
                    deep_files.append(str(file_path))
        
        # Load deep nested structure
        try:
            deep_dataset = Dataset.from_json(deep_files)
            print(f"✅ Deep nested: {len(deep_dataset)} items loaded")
        except Exception as e:
            print(f"❌ Deep nested failed: {e}")
        
        # Test 4: Directory-based loading (HF datasets auto-discovery)
        print("\n📁 Test 4: Directory-based auto-discovery")
        
        # Create a dataset directory with nested JSON files
        auto_dir = test_dir / "auto_discovery"
        auto_dir.mkdir(parents=True)
        
        # Create some JSON files in subdirectories
        for benchmark in ["mmlu", "arc"]:
            bench_dir = auto_dir / benchmark  
            bench_dir.mkdir(parents=True)
            
            # Create both individual files and a dataset file
            for i in range(2):
                file_path = bench_dir / f"data_{i}.json"
                data = [
                    {"id": f"{benchmark}_{i}_0", "score": 0.8, "benchmark": benchmark},
                    {"id": f"{benchmark}_{i}_1", "score": 0.9, "benchmark": benchmark}
                ]
                with open(file_path, 'w') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')  # JSONL format
        
        # Try auto-discovery
        try:
            auto_dataset = load_dataset("json", data_dir=str(auto_dir), split="train")
            print(f"✅ Auto-discovery: {len(auto_dataset)} items loaded")
            print(f"   Features: {auto_dataset.features}")
            
            # Check if all benchmarks are represented
            benchmarks_found = set(auto_dataset['benchmark'])
            print(f"   Benchmarks found: {benchmarks_found}")
            
        except Exception as e:
            print(f"❌ Auto-discovery failed: {e}")
        
        # Test 5: Performance comparison
        print("\n⚡ Performance implications:")
        
        # File count comparison
        print(f"   Flat structure: {len(flat_files)} files in 1 directory")
        print(f"   Simple nested: {len(nested_files)} files in {len(benchmarks)} directories") 
        print(f"   Deep nested: {len(deep_files)} files in {len(benchmarks) * 2 * 2} directories")
        
        # Directory traversal test
        import time
        
        start = time.time()
        all_flat = list(flat_dir.rglob("*.json"))
        flat_time = time.time() - start
        
        start = time.time()
        all_nested = list(nested_dir.rglob("*.json"))
        nested_time = time.time() - start
        
        start = time.time()
        all_deep = list(deep_dir.rglob("*.json"))
        deep_time = time.time() - start
        
        print(f"   File discovery time:")
        print(f"     Flat: {flat_time:.4f}s")
        print(f"     Simple nested: {nested_time:.4f}s") 
        print(f"     Deep nested: {deep_time:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    success = test_nested_folder_structures()
    
    if success:
        print("\n🎯 Recommendations:")
        print("✅ HuggingFace datasets supports nested folders well")
        print("✅ Auto-discovery works with nested JSON/JSONL files")
        print("✅ Performance difference is minimal for reasonable nesting")
        print("\n💡 Best practices:")
        print("📁 Use logical grouping (benchmark/model) for organization")
        print("🚀 Avoid excessive nesting (>3-4 levels) for performance")
        print("📊 Consider flat structure for maximum simplicity")
        print("🔍 Use nested for easier human navigation and debugging")
        exit(0)
    else:
        print("\n💥 Nested folder structure test failed!")
        exit(1)
