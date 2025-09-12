#!/usr/bin/env python3
"""
Demonstrate nested folder structure for EvalHub evaluations.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.simple_helm_processor_evalhub import (
    _create_evaluation_result, save_individual_evaluations, EVALHUB_SCHEMA_AVAILABLE
)

def demonstrate_nested_structure():
    """Demonstrate the nested folder structure with realistic evaluation IDs."""
    
    if not EVALHUB_SCHEMA_AVAILABLE:
        print("❌ EvalHub schema not available")
        return False
    
    # Create realistic sample data that would result in nested structure
    sample_data = [
        # MMLU Clinical Knowledge
        {
            'model': 'meta-llama/Llama-2-7b-hf',
            'input': 'A 65-year-old patient presents with chest pain. What is the most appropriate initial diagnostic test?',
            'output': 'ECG',
            'score': 1.0,
            'metric': 'multiple_choice_grade',
            'instance_id': 42,
            'references': 'ECG',
            'subject': 'clinical_knowledge'
        },
        # MMLU Abstract Algebra  
        {
            'model': 'mistralai/Mistral-7B-v0.1',
            'input': 'What is the order of the cyclic group Z_12?',
            'output': '12',
            'score': 1.0,
            'metric': 'exact_match',
            'instance_id': 43,
            'references': '12',
            'subject': 'abstract_algebra'
        },
        # HellaSwag
        {
            'model': 'anthropic/claude-3-sonnet',
            'input': 'A person is washing dishes. What happens next?',
            'output': 'They rinse the dishes with clean water',
            'score': 0.8,
            'metric': 'multiple_choice_grade',
            'instance_id': 44,
            'references': 'They rinse the dishes with clean water'
        },
        # ARC Challenge
        {
            'model': 'google/gemma-2b',
            'input': 'Which of the following best describes the role of chlorophyll in photosynthesis?',
            'output': 'It captures light energy',
            'score': 1.0,
            'metric': 'multiple_choice_grade', 
            'instance_id': 45,
            'references': 'It captures light energy'
        }
    ]
    
    # Create EvaluationResult objects with different benchmarks
    evaluation_results = []
    benchmarks = ['mmlu', 'mmlu', 'hellaswag', 'arc']
    
    for i, (data, benchmark) in enumerate(zip(sample_data, benchmarks)):
        row = pd.Series(data)
        result = _create_evaluation_result(
            row=row,
            task_id=f'task_{i+1000}',
            benchmark=benchmark,
            source='helm'
        )
        if result:
            # Manually set dataset_name to create realistic nested structure
            if benchmark == 'mmlu':
                subject = data.get('subject', 'default')
                result['instance']['sample_identifier']['dataset_name'] = f"mmlu.{subject}"
            elif benchmark == 'arc':
                result['instance']['sample_identifier']['dataset_name'] = f"arc.challenge"
            else:
                result['instance']['sample_identifier']['dataset_name'] = f"{benchmark}.default"
            
            evaluation_results.append(result)
    
    print(f"✅ Created {len(evaluation_results)} EvaluationResult objects")
    
    # Show expected directory structure
    print("\n📁 Expected nested structure:")
    for result in evaluation_results:
        eval_id = result['evaluation_id']
        dataset_name = result['instance']['sample_identifier']['dataset_name']
        
        parts = eval_id.split('_')
        source_part = parts[0] if parts else 'helm'
        
        if '.' in dataset_name:
            benchmark_name, subject = dataset_name.split('.', 1)
            path = f"data/evaluations/{source_part}/{benchmark_name}/{subject}/{eval_id}.json"
        else:
            path = f"data/evaluations/{source_part}/{dataset_name}/default/{eval_id}.json"
        
        print(f"   📄 {path}")
    
    # Save with nested structure
    print(f"\n💾 Saving evaluations with nested structure...")
    saved_files = save_individual_evaluations(evaluation_results, "helm")
    
    print(f"\n📊 Actual files created:")
    base_path = Path("data/evaluations")
    for file_path in saved_files[:10]:  # Show first 10
        rel_path = Path(file_path).relative_to(base_path)
        print(f"   📄 {rel_path}")
    
    # Show directory tree
    print(f"\n🌳 Directory tree:")
    if base_path.exists():
        for root, dirs, files in base_path.walk():
            level = len(root.relative_to(base_path).parts)
            indent = "  " * level
            print(f"{indent}📁 {root.name}/")
            sub_indent = "  " * (level + 1)
            for file in files[:3]:  # Show first 3 files per directory
                print(f"{sub_indent}📄 {file}")
            if len(files) > 3:
                print(f"{sub_indent}📄 ... and {len(files) - 3} more files")
    
    return True


if __name__ == "__main__":
    success = demonstrate_nested_structure()
    
    if success:
        print("\n🎯 Nested Structure Benefits:")
        print("📁 Logical organization by benchmark and subject")
        print("🔍 Easy navigation for humans and scripts")
        print("⚡ Filesystem performance with distributed files")
        print("🚀 Parallel processing by directory")
        print("🔧 Easier maintenance and bulk operations")
        print("\n✅ HuggingFace datasets will handle this structure perfectly!")
        exit(0)
    else:
        print("\n💥 Nested structure demonstration failed!")
        exit(1)
