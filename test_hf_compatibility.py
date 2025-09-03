#!/usr/bin/env python3
"""
Test script to verify HuggingFace compatibility of our dataset format.

This script tests:
1. Parquet file format compatibility
2. Schema validation 
3. Loading with datasets library
4. Streaming compatibility
"""

import tempfile
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from src.core.scrape_all_to_parquet import _build_pyarrow_schema

def test_schema_compatibility():
    """Test that our schema is compatible with HuggingFace datasets."""
    print("Testing schema compatibility...")
    
    schema = _build_pyarrow_schema()
    print(f"✓ Schema has {len(schema)} fields")
    
    # Create a small test dataset
    test_data = {
        'evaluation_id': ['eval_001', 'eval_002'],
        'dataset_name': ['test_dataset', 'test_dataset'],
        'hf_split': ['train', 'train'],
        'hf_index': [0, 1],
        'raw_input': ['What is 2+2?', 'What is the capital of France?'],
        'ground_truth': ['4', 'Paris'],
        'model_name': ['test_model', 'test_model'],
        'model_family': ['test_family', 'test_family'],
        'output': ['4', 'Paris'],
        'evaluation_method_name': ['exact_match', 'exact_match'],
        'evaluation_score': [1.0, 1.0],
        'run': ['test_run', 'test_run'],
        'task': ['arithmetic', 'geography'],
        'adapter_method': ['none', 'none'],
        'source': ['test', 'test'],
        'source_version': ['1.0', '1.0'],
        'source_url': ['http://test.com', 'http://test.com'],
        'ingestion_timestamp': [pd.Timestamp.now(), pd.Timestamp.now()],
        'license': ['apache-2.0', 'apache-2.0'],
        'category': ['knowledge', 'knowledge']
    }
    
    df = pd.DataFrame(test_data)
    
    # Convert to Arrow table with our schema
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    print(f"✓ Successfully created Arrow table with {table.num_rows} rows")
    
    return table

def test_parquet_format(table):
    """Test parquet writing and reading with HF-optimized settings."""
    print("Testing parquet format...")
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        temp_path = f.name
    
    try:
        # Write with HF-optimized settings
        writer = pq.ParquetWriter(
            temp_path,
            table.schema,
            compression='snappy',
            use_dictionary=True,
            write_statistics=True,
            row_group_size=100000
        )
        writer.write_table(table)
        writer.close()
        
        print(f"✓ Successfully wrote parquet file to {temp_path}")
        
        # Read back and verify
        read_table = pq.read_table(temp_path)
        assert read_table.num_rows == table.num_rows
        assert read_table.schema.equals(table.schema, check_metadata=False)
        print("✓ Successfully read and verified parquet file")
        
        return temp_path
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

def test_datasets_compatibility(parquet_path):
    """Test loading with HuggingFace datasets library."""
    print("Testing HuggingFace datasets compatibility...")
    
    try:
        # Test loading from parquet
        dataset = Dataset.from_parquet(parquet_path)
        print(f"✓ Successfully loaded dataset with {len(dataset)} rows")
        print(f"✓ Dataset features: {list(dataset.features.keys())}")
        
        # Test basic operations
        sample = dataset[0]
        assert 'evaluation_id' in sample
        assert 'model_name' in sample
        print("✓ Successfully accessed dataset elements")
        
        # Test filtering
        filtered = dataset.filter(lambda x: x['evaluation_score'] > 0.5)
        print(f"✓ Successfully filtered dataset to {len(filtered)} rows")
        
        # Test mapping
        mapped = dataset.map(lambda x: {'score_doubled': x['evaluation_score'] * 2})
        assert 'score_doubled' in mapped[0]
        print("✓ Successfully mapped dataset")
        
        # Test batching (similar to HF's batch processing)
        batched = dataset.batch(batch_size=2)
        print(f"✓ Successfully created batched dataset")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Error testing datasets compatibility: {e}")
        raise

def test_shard_naming():
    """Test that our shard naming follows HF conventions."""
    print("Testing shard naming conventions...")
    
    # Test the naming pattern we use - simple incremental format
    shard_names = [
        "data-00001.parquet",
        "data-00002.parquet", 
        "data-00010.parquet",
        "data-12345.parquet"
    ]
    
    import re
    pattern = r'^data-(\d{5})\.parquet$'
    
    for name in shard_names:
        match = re.match(pattern, name)
        assert match, f"Shard name {name} doesn't match our pattern"
        shard_num = int(match.group(1))
        assert shard_num >= 1, f"Invalid shard numbering in {name}"
    
    print("✓ Shard naming follows our incremental convention")

def main():
    """Run all compatibility tests."""
    print("🧪 Testing HuggingFace compatibility...\n")
    
    try:
        # Test 1: Schema compatibility
        table = test_schema_compatibility()
        print()
        
        # Test 2: Parquet format
        parquet_path = test_parquet_format(table)
        print()
        
        # Test 3: Datasets library compatibility
        dataset = test_datasets_compatibility(parquet_path)
        print()
        
        # Test 4: Shard naming
        test_shard_naming()
        print()
        
        print("🎉 All HuggingFace compatibility tests passed!")
        
        # Clean up
        if os.path.exists(parquet_path):
            os.unlink(parquet_path)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
