#!/usr/bin/env python3
"""
Examine the structure of generated parquet files
"""

import pandas as pd
from pathlib import Path

def examine_parquet_files():
    """Examine the structure and content of parquet files."""
    
    agg_dir = Path("data/aggregated")
    parquet_files = list(agg_dir.glob("*.parquet"))
    
    print(f"📊 Found {len(parquet_files)} parquet files")
    
    for file in parquet_files:
        print(f"\n" + "="*60)
        print(f"📄 Examining: {file.name}")
        print(f"📊 File size: {file.stat().st_size / 1024:.1f} KB")
        
        # Read parquet file
        df = pd.read_parquet(file)
        
        print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show column names and types
        print(f"\n📋 Columns:")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            print(f"  • {col}: {dtype} ({non_null}/{len(df)} non-null)")
        
        # Show metadata columns specifically
        print(f"\n🏷️  Metadata columns:")
        metadata_cols = ['source', 'benchmark', 'timestamp', 'processing_date']
        for col in metadata_cols:
            if col in df.columns:
                unique_vals = df[col].nunique()
                sample_val = df[col].iloc[0] if len(df) > 0 else "N/A"
                print(f"  • {col}: {unique_vals} unique values, sample: '{sample_val}'")
        
        # Show first few rows
        print(f"\n📋 Sample data (first 3 rows, key columns):")
        key_cols = ['source', 'benchmark', 'dataset_name', 'model_name', 'evaluation_score']
        available_key_cols = [col for col in key_cols if col in df.columns]
        if available_key_cols:
            print(df[available_key_cols].head(3).to_string(index=False))
        
        # Show unique models and datasets
        if 'model_name' in df.columns:
            models = df['model_name'].unique()
            print(f"\n🤖 Models in this file ({len(models)}):")
            for model in sorted(models):
                count = (df['model_name'] == model).sum()
                if 'evaluation_score' in df.columns:
                    avg_score = df[df['model_name'] == model]['evaluation_score'].mean()
                    print(f"  • {model}: {count} samples, avg score: {avg_score:.3f}")
                else:
                    print(f"  • {model}: {count} samples")
        
        if 'dataset_name' in df.columns:
            datasets = df['dataset_name'].unique()
            print(f"\n📊 Datasets in this file ({len(datasets)}):")
            for dataset in sorted(datasets):
                count = (df['dataset_name'] == dataset).sum()
                print(f"  • {dataset}: {count} samples")

if __name__ == "__main__":
    examine_parquet_files()
