#!/usr/bin/env python3
"""
Test end-to-end EvalHub schema compliance with real HELM data.
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

from scripts.simple_helm_processor_evalhub import (
    _create_evaluation_result, EVALHUB_SCHEMA_AVAILABLE
)

def test_with_real_helm_data():
    """Test with actual HELM data structure to ensure schema compliance."""
    
    if not EVALHUB_SCHEMA_AVAILABLE:
        print("❌ EvalHub schema not available")
        return False
    
    if not DATASETS_AVAILABLE:
        print("❌ HuggingFace datasets library not available") 
        return False
    
    # Check if we have any real HELM data
    test_files = [
        "data/test_output/mmlu:subject=clinical_knowledge,method=multiple_choice_joint,model=01-ai_yi-6b,eval_split=test,groups=mmlu_clinical_knowledge_converted.csv",
        "data/test_output/sample_evalhub_format.csv"
    ]
    
    real_data_file = None
    for file_path in test_files:
        if Path(file_path).exists():
            real_data_file = file_path
            break
    
    if not real_data_file:
        print("⚠️ No real HELM data files found, using synthetic data")
        return test_with_synthetic_helm_data()
    
    print(f"📄 Testing with real HELM data: {real_data_file}")
    
    try:
        # Load real HELM data
        df = pd.read_csv(real_data_file)
        print(f"📊 Loaded {len(df)} rows from real HELM data")
        print(f"🔍 Columns: {list(df.columns)}")
        
        # Take first few rows for testing
        test_rows = df.head(5)
        
        # Create EvaluationResult objects
        evaluation_results = []
        for idx, (_, row) in enumerate(test_rows.iterrows()):
            result = _create_evaluation_result(
                row=row,
                task_id=f'real_task_{idx}',
                benchmark='mmlu_clinical_knowledge',
                source='helm'
            )
            
            if result is None:
                print(f"❌ Failed to create EvaluationResult for row {idx}")
                continue
            
            evaluation_results.append(result)
        
        if not evaluation_results:
            print("❌ No valid EvaluationResult objects created")
            return False
        
        print(f"✅ Created {len(evaluation_results)} EvaluationResult objects from real data")
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(evaluation_results)
        print(f"✅ Successfully created HuggingFace Dataset: {len(dataset)} items")
        
        # Validate schema compliance
        first_item = dataset[0]
        required_top_level = ['schema_version', 'evaluation_id', 'model', 'prompt_config', 'instance', 'output', 'evaluation']
        
        for field in required_top_level:
            if field not in first_item:
                print(f"❌ Missing required top-level field: {field}")
                return False
        
        # Validate nested structure
        if 'model_info' not in first_item['model']:
            print("❌ Missing model.model_info")
            return False
        
        if 'sample_identifier' not in first_item['instance']:
            print("❌ Missing instance.sample_identifier")
            return False
        
        if 'evaluation_method' not in first_item['evaluation']:
            print("❌ Missing evaluation.evaluation_method")
            return False
        
        print("✅ Schema structure validation passed")
        
        # Test serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            dataset.to_json(json_path)
            file_size = os.path.getsize(json_path) / 1024  # KB
            print(f"💾 Serialized to JSON: {file_size:.1f} KB")
            
            # Load back and verify
            loaded_dataset = Dataset.from_json(json_path)
            if len(loaded_dataset) == len(dataset):
                print("✅ JSON round-trip successful")
            else:
                print("❌ JSON round-trip failed")
                return False
                
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)
        
        # Display sample for verification
        print("\n📋 Sample EvaluationResult:")
        sample = first_item
        print(f"  📄 ID: {sample['evaluation_id']}")
        print(f"  🤖 Model: {sample['model']['model_info']['name']}")
        print(f"  🏗️  Architecture: {sample['model']['configuration']['architecture']}")
        print(f"  💬 Prompt Class: {sample['prompt_config']['prompt_class']}")
        print(f"  🎯 Task Type: {sample['instance']['task_type']}")
        print(f"  📊 Score: {sample['evaluation']['score']}")
        print(f"  📈 Method: {sample['evaluation']['evaluation_method']['method_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing real HELM data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_synthetic_helm_data():
    """Test with synthetic HELM-like data."""
    
    # Create synthetic HELM data that matches common patterns
    synthetic_data = [
        {
            'model': 'meta-llama/Llama-2-7b-hf',
            'input': 'A 65-year-old patient presents with chest pain. What is the most appropriate initial diagnostic test?',
            'output': 'ECG',
            'score': 1.0,
            'metric': 'multiple_choice_grade',
            'instance_id': 42,
            'references': 'ECG',
            'subject': 'clinical_knowledge',
            'choices': 'A) ECG B) Chest X-ray C) Blood test D) MRI',
            'temperature': 0.0,
            'max_tokens': 100
        },
        {
            'model': 'mistralai/Mistral-7B-Instruct-v0.1',
            'input': 'What is the mechanism of action of aspirin?',
            'output': 'Irreversible inhibition of cyclooxygenase enzymes',
            'score': 0.8,
            'metric': 'exact_match',
            'instance_id': 43,
            'references': 'Irreversible inhibition of cyclooxygenase',
            'subject': 'clinical_knowledge',
            'temperature': 0.7,
            'max_tokens': 150
        }
    ]
    
    evaluation_results = []
    for idx, data in enumerate(synthetic_data):
        row = pd.Series(data)
        result = _create_evaluation_result(
            row=row,
            task_id=f'synthetic_task_{idx}',
            benchmark='mmlu_clinical_knowledge',
            source='helm'
        )
        
        if result:
            evaluation_results.append(result)
    
    if not evaluation_results:
        print("❌ No synthetic evaluation results created")
        return False
    
    # Create dataset and validate
    dataset = Dataset.from_list(evaluation_results)
    print(f"✅ Created dataset from {len(evaluation_results)} synthetic samples")
    
    return True


if __name__ == "__main__":
    success = test_with_real_helm_data()
    if success:
        print("\n🎉 End-to-end EvalHub schema test passed!")
        exit(0)
    else:
        print("\n💥 End-to-end EvalHub schema test failed!")
        exit(1)
