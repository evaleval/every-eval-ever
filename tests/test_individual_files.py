#!/usr/bin/env python3
"""
Test individual JSON file approach for EvalHub evaluations.
"""

import sys
from pathlib import Path
import pandas as pd
import tempfile
import os
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.simple_helm_processor_evalhub import (
    _create_evaluation_result, save_individual_evaluations, 
    aggregate_individual_files_to_hf_dataset, EVALHUB_SCHEMA_AVAILABLE
)

def test_individual_file_approach():
    """Test the individual JSON file approach for better deduplication and parallelization."""
    
    if not EVALHUB_SCHEMA_AVAILABLE:
        print("❌ EvalHub schema not available")
        return False
    
    # Create temporary directory for test
    test_dir = Path("data/evaluations_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create sample evaluations
        sample_data = [
            {
                'model': 'meta-llama/Llama-2-7b-hf',
                'input': 'What is the capital of France?',
                'output': 'Paris',
                'score': 1.0,
                'metric': 'exact_match',
                'instance_id': 123,
                'references': 'Paris'
            },
            {
                'model': 'mistralai/Mistral-7B-v0.1',
                'input': 'What is 2+2?',
                'output': '4',
                'score': 1.0,
                'metric': 'exact_match',
                'instance_id': 124,
                'references': '4'
            },
            {
                'model': 'unknown-vendor/test-model',
                'input': 'Test question',
                'output': 'Test answer',
                'score': 0.8,
                'metric': 'exact_match',
                'instance_id': 125,
                'references': 'Test answer'
            }
        ]
        
        # Create EvaluationResult objects
        evaluation_results = []
        for i, data in enumerate(sample_data):
            row = pd.Series(data)
            result = _create_evaluation_result(
                row=row,
                task_id=f'test_task_{i}',
                benchmark='test_benchmark',
                source='helm'
            )
            if result:
                evaluation_results.append(result)
        
        print(f"✅ Created {len(evaluation_results)} EvaluationResult objects")
        
        # Test individual file saving
        print("\n🧪 Testing individual file saving...")
        
        # Temporarily change the save directory
        original_save_dir = Path("data/evaluations")
        temp_save_dir = test_dir / "individual_files"
        
        # Monkey patch the save function to use test directory
        import scripts.simple_helm_processor_evalhub as processor_module
        original_path = processor_module.Path
        
        def patched_path(path_str):
            if path_str == "data/evaluations":
                return temp_save_dir
            return original_path(path_str)
        
        processor_module.Path = patched_path
        
        try:
            saved_files = save_individual_evaluations(evaluation_results, "helm")
            print(f"✅ Saved {len(saved_files)} individual files")
            
            # Verify files exist and have correct content
            for file_path in saved_files:
                if Path(file_path).exists():
                    print(f"   📄 {Path(file_path).name}")
                else:
                    print(f"   ❌ Missing: {file_path}")
                    return False
            
            # Test deduplication - save again
            print("\n🧪 Testing deduplication...")
            saved_files_2 = save_individual_evaluations(evaluation_results, "helm")
            print(f"✅ Second save attempt: {len(saved_files_2)} new files (should be 0)")
            
            if len(saved_files_2) == 0:
                print("✅ Deduplication working correctly")
            else:
                print("❌ Deduplication failed")
                return False
            
            # Test aggregation
            print("\n🧪 Testing aggregation to HuggingFace dataset...")
            aggregated_file = aggregate_individual_files_to_hf_dataset(saved_files, 1)
            
            if aggregated_file and Path(aggregated_file).exists():
                print(f"✅ Aggregated file created: {Path(aggregated_file).name}")
                file_size = Path(aggregated_file).stat().st_size
                print(f"   📏 File size: {file_size} bytes")
            else:
                print("❌ Aggregation failed")
                return False
            
        finally:
            # Restore original Path
            processor_module.Path = original_path
        
        # Show sample filenames for different evaluation IDs
        print("\n📋 Sample filename patterns:")
        for result in evaluation_results[:2]:
            eval_id = result['evaluation_id']
            safe_name = eval_id.replace('/', '_').replace(':', '_').replace(',', '_').replace('=', '_')
            print(f"   {eval_id} → {safe_name}.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    success = test_individual_file_approach()
    if success:
        print("\n🎉 Individual file approach test passed!")
        print("\n💡 Benefits:")
        print("   🚀 Parallelization: Each evaluation can be processed independently") 
        print("   🔍 Deduplication: File existence check prevents duplicates")
        print("   📁 Organization: Each evaluation has unique filename")
        print("   🔧 Efficiency: Only new evaluations are processed")
        exit(0)
    else:
        print("\n💥 Individual file approach test failed!")
        exit(1)
