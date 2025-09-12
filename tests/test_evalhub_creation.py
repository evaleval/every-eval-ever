#!/usr/bin/env python3
"""
Test script to validate EvalHub schema EvaluationResult creation.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the processor functions
from scripts.simple_helm_processor_evalhub import (
    _create_evaluation_result, EVALHUB_SCHEMA_AVAILABLE
)

def test_evaluation_result_creation():
    """Test creating EvaluationResult objects from sample HELM data."""
    
    if not EVALHUB_SCHEMA_AVAILABLE:
        print("❌ EvalHub schema not available")
        return False
    
    # Create sample HELM data
    sample_data = {
        'model': 'meta-llama/Llama-2-7b-hf',
        'input': 'What is the capital of France?',
        'output': 'Paris',
        'score': 1.0,
        'metric': 'exact_match',
        'instance_id': 123,
        'references': 'Paris',
        'temperature': 0.7,
        'max_tokens': 100
    }
    
    # Convert to pandas Series (simulating CSV row)
    row = pd.Series(sample_data)
    
    # Test creating EvaluationResult
    try:
        result = _create_evaluation_result(
            row=row,
            task_id='test_task',
            benchmark='test_benchmark',
            source='helm'
        )
        
        if result is None:
            print("❌ Failed to create EvaluationResult - returned None")
            return False
        
        # Validate the structure
        required_fields = [
            'schema_version', 'evaluation_id', 'model', 'prompt_config',
            'instance', 'output', 'evaluation'
        ]
        
        for field in required_fields:
            if field not in result:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check nested structure
        if 'model_info' not in result['model']:
            print("❌ Missing model.model_info")
            return False
            
        if 'name' not in result['model']['model_info']:
            print("❌ Missing model.model_info.name")
            return False
        
        print("✅ EvaluationResult created successfully!")
        print(f"📋 Evaluation ID: {result['evaluation_id']}")
        print(f"🤖 Model: {result['model']['model_info']['name']}")
        print(f"📊 Score: {result['evaluation']['score']}")
        print(f"🎯 Task Type: {result['instance']['task_type']}")
        print(f"💬 Prompt Class: {result['prompt_config']['prompt_class']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating EvaluationResult: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_evaluation_result_creation()
    if success:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n💥 Tests failed!")
        exit(1)
