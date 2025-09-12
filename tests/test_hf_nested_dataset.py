#!/usr/bin/env python3
"""
Test script to validate HuggingFace datasets with nested EvaluationResult objects.
"""

import sys
from pathlib import Path
import pandas as pd
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import required libraries
try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Import the processor functions
from scripts.simple_helm_processor_evalhub import (
    _create_evaluation_result, EVALHUB_SCHEMA_AVAILABLE
)

def test_huggingface_dataset_with_nested_objects():
    """Test creating HuggingFace datasets from nested EvaluationResult objects."""
    
    if not EVALHUB_SCHEMA_AVAILABLE:
        print("❌ EvalHub schema not available")
        return False
    
    if not DATASETS_AVAILABLE:
        print("❌ HuggingFace datasets library not available")
        return False
    
    # Create multiple sample HELM data entries
    sample_data_list = [
        {
            'model': 'meta-llama/Llama-2-7b-hf',
            'input': 'What is the capital of France?',
            'output': 'Paris',
            'score': 1.0,
            'metric': 'exact_match',
            'instance_id': 123,
            'references': 'Paris',
            'temperature': 0.7,
            'max_tokens': 100
        },
        {
            'model': 'mistralai/Mistral-7B-v0.1',
            'input': 'What is 2+2?',
            'output': '4',
            'score': 1.0,
            'metric': 'exact_match',
            'instance_id': 124,
            'references': '4',
            'temperature': 0.0,
            'max_tokens': 50
        },
        {
            'model': 'google/gemma-2b',
            'input': 'What color is the sky?',
            'output': 'blue',
            'score': 1.0,
            'metric': 'exact_match',
            'instance_id': 125,
            'references': 'blue',
            'temperature': 0.5,
            'max_tokens': 20
        }
    ]
    
    # Create EvaluationResult objects
    evaluation_results = []
    for i, sample_data in enumerate(sample_data_list):
        row = pd.Series(sample_data)
        result = _create_evaluation_result(
            row=row,
            task_id=f'test_task_{i}',
            benchmark='test_benchmark',
            source='helm'
        )
        
        if result is None:
            print(f"❌ Failed to create EvaluationResult for sample {i}")
            return False
        
        evaluation_results.append(result)
    
    print(f"✅ Created {len(evaluation_results)} EvaluationResult objects")
    
    # Create HuggingFace dataset from nested objects
    try:
        dataset = Dataset.from_list(evaluation_results)
        print("✅ Successfully created HuggingFace Dataset from nested objects")
        print(f"📊 Dataset length: {len(dataset)}")
        print(f"📋 Dataset features: {list(dataset.features.keys())}")
        
        # Test data access patterns
        print("\n🔍 Testing data access patterns:")
        
        # Access first item
        first_item = dataset[0]
        print(f"📄 First evaluation ID: {first_item['evaluation_id']}")
        print(f"🤖 Model name: {first_item['model']['model_info']['name']}")
        print(f"📊 Score: {first_item['evaluation']['score']}")
        
        # Test filtering
        llama_items = dataset.filter(lambda x: 'llama' in x['model']['model_info']['name'].lower())
        print(f"🦙 Llama items: {len(llama_items)}")
        
        # Test conversion to pandas
        df = dataset.to_pandas()
        print(f"🐼 Converted to pandas: {df.shape}")
        print(f"🔍 Columns: {list(df.columns)}")
        
        # Test JSON serialization/deserialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            dataset.to_json(json_path)
            print(f"💾 Saved to JSON: {json_path}")
            
            # Load back from JSON
            loaded_dataset = Dataset.from_json(json_path)
            print(f"📂 Loaded from JSON: {len(loaded_dataset)} items")
            
            # Verify data integrity
            if len(loaded_dataset) == len(dataset):
                print("✅ Data integrity preserved after JSON round-trip")
            else:
                print("❌ Data integrity lost during JSON round-trip")
                return False
                
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error working with HuggingFace Dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_huggingface_dataset_with_nested_objects()
    if success:
        print("\n🎉 All HuggingFace dataset tests passed!")
        exit(0)
    else:
        print("\n💥 HuggingFace dataset tests failed!")
        exit(1)
