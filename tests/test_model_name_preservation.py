#!/usr/bin/env python3
"""
Test to verify model name preservation even when family enum mapping fails.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.simple_helm_processor_evalhub import (
    _create_evaluation_result, _map_family_to_enum, EVALHUB_SCHEMA_AVAILABLE
)

def test_model_name_preservation():
    """Test that model names are preserved even when family mapping fails."""
    
    if not EVALHUB_SCHEMA_AVAILABLE:
        print("❌ EvalHub schema not available")
        return False
    
    # Test cases with models that won't map to existing Family enums
    test_cases = [
        {
            'model': 'unknown-vendor/mystery-model-7b',
            'expected_family_mapped': False,
            'description': 'Completely unknown model'
        },
        {
            'model': 'anthropic/claude-3-opus',
            'expected_family_mapped': True,  # 'claude' should map
            'description': 'Claude model (should map)'
        },
        {
            'model': 'bigscience/bloom-7b',
            'expected_family_mapped': False,  # 'bloom' not in mapping
            'description': 'BLOOM model (no mapping)'
        },
        {
            'model': '01-ai/Yi-34B-Chat',
            'expected_family_mapped': False,  # 'Yi' not in EvalHub Family enum
            'description': 'Yi model (not in EvalHub enum)'
        },
        {
            'model': 'meta-llama/Llama-2-70b-chat-hf',
            'expected_family_mapped': True,  # 'llama' should map
            'description': 'Llama model (should map)'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        model_name = test_case['model']
        print(f"\n🧪 Testing: {test_case['description']}")
        print(f"   Model: {model_name}")
        
        # Test family mapping
        family_enum = _map_family_to_enum(model_name)
        family_mapped = family_enum is not None
        
        print(f"   Family enum: {family_enum}")
        print(f"   Family mapped: {family_mapped}")
        
        # Create sample row data
        row_data = {
            'model': model_name,
            'input': 'Test question',
            'output': 'Test answer', 
            'score': 0.8,
            'metric': 'exact_match',
            'instance_id': 42,
            'references': 'Test answer'
        }
        
        row = pd.Series(row_data)
        
        # Create EvaluationResult
        result = _create_evaluation_result(
            row=row,
            task_id='test_task',
            benchmark='test_benchmark', 
            source='test'
        )
        
        if result is None:
            print(f"   ❌ Failed to create EvaluationResult")
            results.append(False)
            continue
        
        # Check if model name is preserved
        stored_name = result['model']['model_info']['name']
        stored_family = result['model']['model_info']['family']
        
        print(f"   📝 Stored name: {stored_name}")
        print(f"   👥 Stored family: {stored_family}")
        
        # Verify name preservation
        name_preserved = (stored_name == model_name)
        family_expectation_met = (family_mapped == test_case['expected_family_mapped'])
        
        if name_preserved:
            print(f"   ✅ Model name preserved correctly")
        else:
            print(f"   ❌ Model name NOT preserved: expected '{model_name}', got '{stored_name}'")
        
        if family_expectation_met:
            print(f"   ✅ Family mapping as expected")
        else:
            print(f"   ❌ Family mapping unexpected: expected mapped={test_case['expected_family_mapped']}, got mapped={family_mapped}")
        
        test_passed = name_preserved and family_expectation_met
        results.append(test_passed)
        
        if test_passed:
            print(f"   🎉 Test PASSED")
        else:
            print(f"   💥 Test FAILED")
    
    # Summary
    passed_count = sum(results)
    total_count = len(results)
    
    print(f"\n📊 Summary: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All model name preservation tests passed!")
        return True
    else:
        print("💥 Some model name preservation tests failed!")
        return False


if __name__ == "__main__":
    success = test_model_name_preservation()
    exit(0 if success else 1)
