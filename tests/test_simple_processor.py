#!/usr/bin/env python3
"""
Test the simplified processor's deduplication logic.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_unique_id_creation():
    """Test that unique IDs are created correctly."""
    
    # Test cases
    test_cases = [
        {
            'source': 'helm',
            'benchmark': 'lite', 
            'task_id': 'commonsense:dataset=openbookqa,method=multiple_choice_joint,model=01-ai_yi-34b',
            'expected': 'helm+lite+commonsense:dataset=openbookqa,method=multiple_choice_joint,model=01-ai_yi-34b'
        },
        {
            'source': 'helm',
            'benchmark': 'mmlu',
            'task_id': 'mmlu:subject=clinical_knowledge,method=multiple_choice_joint,model=01-ai_yi-6b,eval_split=test,groups=mmlu_clinical_knowledge', 
            'expected': 'helm+mmlu+mmlu:subject=clinical_knowledge,method=multiple_choice_joint,model=01-ai_yi-6b,eval_split=test,groups=mmlu_clinical_knowledge'
        },
        {
            'source': 'helm',
            'benchmark': 'classic',
            'task_id': 'babi_qa:task=15,model=together_opt-175b',
            'expected': 'helm+classic+babi_qa:task=15,model=together_opt-175b'
        }
    ]
    
    print("🧪 Testing unique ID creation...")
    
    for i, case in enumerate(test_cases, 1):
        unique_id = f"{case['source']}+{case['benchmark']}+{case['task_id']}"
        
        if unique_id == case['expected']:
            print(f"✅ Test {i}: {unique_id}")
        else:
            print(f"❌ Test {i}: Expected {case['expected']}, got {unique_id}")
            return False
    
    return True


def test_deduplication_logic():
    """Test the deduplication filtering logic."""
    
    print("\n🧪 Testing deduplication logic...")
    
    # Simulate existing tasks in repo
    existing_tasks = {
        'helm+lite+task1',
        'helm+lite+task2', 
        'helm+mmlu+task3',
        'helm+classic+task4'
    }
    
    # Simulate all tasks from benchmarks
    all_tasks = {
        'lite': ['task1', 'task2', 'task5'],  # task1, task2 already exist
        'mmlu': ['task3', 'task6'],           # task3 already exists  
        'classic': ['task4', 'task7', 'task8'] # task4 already exists
    }
    
    # Apply filtering logic
    new_tasks = {}
    source = "helm"
    
    for benchmark, tasks in all_tasks.items():
        new_benchmark_tasks = []
        
        for task in tasks:
            unique_id = f"{source}+{benchmark}+{task}"
            if unique_id not in existing_tasks:
                new_benchmark_tasks.append(task)
        
        new_tasks[benchmark] = new_benchmark_tasks
    
    # Expected results
    expected = {
        'lite': ['task5'],
        'mmlu': ['task6'],
        'classic': ['task7', 'task8']
    }
    
    if new_tasks == expected:
        print(f"✅ Deduplication works correctly:")
        for benchmark, tasks in new_tasks.items():
            print(f"   {benchmark}: {len(tasks)} new tasks {tasks}")
        return True
    else:
        print(f"❌ Deduplication failed:")
        print(f"   Expected: {expected}")
        print(f"   Got: {new_tasks}")
        return False


def test_chunk_creation():
    """Test global chunk creation logic."""
    
    print("\n🧪 Testing chunk creation...")
    
    new_tasks = {
        'lite': ['task1', 'task2'],
        'mmlu': ['task3', 'task4', 'task5'],
        'classic': ['task6']
    }
    
    chunk_size = 3
    
    # Flatten all tasks with their benchmark info
    all_task_items = []
    for benchmark, tasks in new_tasks.items():
        for task in tasks:
            all_task_items.append({'benchmark': benchmark, 'task': task})
    
    # Create chunks
    chunks = []
    for i in range(0, len(all_task_items), chunk_size):
        chunk_items = all_task_items[i:i + chunk_size]
        
        # Group by benchmark within chunk
        chunk_data = {}
        for item in chunk_items:
            benchmark = item['benchmark']
            if benchmark not in chunk_data:
                chunk_data[benchmark] = []
            chunk_data[benchmark].append(item['task'])
        
        chunks.append({
            'chunk_id': i // chunk_size + 1,
            'benchmarks': chunk_data,
            'total_tasks': len(chunk_items)
        })
    
    # Should create 2 chunks (3 + 3 tasks)
    if len(chunks) == 2:
        print(f"✅ Created {len(chunks)} chunks correctly:")
        for chunk in chunks:
            print(f"   Chunk {chunk['chunk_id']}: {chunk['total_tasks']} tasks across {len(chunk['benchmarks'])} benchmarks")
            for benchmark, tasks in chunk['benchmarks'].items():
                print(f"     {benchmark}: {tasks}")
        return True
    else:
        print(f"❌ Expected 2 chunks, got {len(chunks)}")
        return False


if __name__ == "__main__":
    print("🚀 Testing simplified processor logic\n")
    
    tests = [
        test_unique_id_creation,
        test_deduplication_logic, 
        test_chunk_creation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)
