#!/usr/bin/env python3
"""
Simple Statistics Generator for HELM Evaluations

This script generates comprehensive statistics from evaluation data in HuggingFace repositories
and uploads them to a dedicated statistics repository.

Usage:
    python scripts/simple_stats_generator.py --repo-id evaleval/every_eval_ever --stats-repo-id evaleval/every_eval_score_ever --source-name helm
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_comprehensive_statistics(repo_id: str, source_name: str) -> dict:
    """
    Generate comprehensive statistics from evaluation data.
    
    Args:
        repo_id: Source repository ID containing evaluation data
        source_name: Source name (e.g., 'helm')
        
    Returns:
        Dictionary containing comprehensive statistics
    """
    logger.info(f"📊 Loading evaluation data from {repo_id}")
    
    try:
        # Load the dataset with explicit JSONL format
        # Use streaming to avoid rate limits when loading all files at once
        try:
            dataset = load_dataset(repo_id, data_files="**/*.jsonl", split="train", streaming=False)
            logger.info(f"✅ Loaded {len(dataset)} evaluation records from all JSONL files")
        except Exception as e:
            # Fallback: try JSON files if JSONL not available yet
            logger.warning(f"JSONL files not found, trying JSON files: {e}")
            try:
                dataset = load_dataset(repo_id, data_files="**/*.json", split="train", streaming=False)
                logger.info(f"✅ Loaded {len(dataset)} evaluation records from JSON files")
            except Exception as e2:
                # Final fallback: use cached version
                logger.warning(f"Rate limit hit, trying cached version: {e2}")
                dataset = load_dataset(repo_id, split="train")
                logger.info(f"✅ Loaded {len(dataset)} evaluation records from cached version")
        
        # Convert to pandas for easier analysis
        df = dataset.to_pandas()
        
        # Extract nested values for analysis
        logger.info("🔍 Extracting nested values from EvalHub schema...")
        
        # Extract model names from nested structure
        df['model_name'] = df['model'].apply(
            lambda x: x['model_info']['name'] if isinstance(x, dict) and 'model_info' in x and 'name' in x['model_info'] else 'unknown'
        )
        
        # Extract evaluation scores, filtering out sentinel values
        def extract_score(evaluation_data):
            if isinstance(evaluation_data, dict) and 'score' in evaluation_data:
                score = evaluation_data['score']
                # Filter out sentinel values (-1.0, None, negative scores)
                if score is not None and score >= 0 and score != -1.0:
                    return score
            return None
        
        df['score'] = df['evaluation'].apply(extract_score)
        
        # Extract dataset names
        df['dataset_name'] = df['instance'].apply(
            lambda x: x['sample_identifier']['dataset_name'] if isinstance(x, dict) and 'sample_identifier' in x and 'dataset_name' in x['sample_identifier'] else 'unknown'
        )
        
        # Smart dataset extraction and tagging using schema + evaluation_id
        def extract_dataset_and_tags(row):
            eval_id = row.get('evaluation_id', '')
            
            # Extract from instance schema
            instance = row.get('instance', {})
            sample_id = instance.get('sample_identifier', {})
            dataset_name = sample_id.get('dataset_name', '')
            hf_repo = sample_id.get('hf_repo', '')
            task_type = instance.get('task_type', '')
            language = instance.get('language', 'en')
            
            # Extract specific dataset from schema when available
            specific_dataset = 'unknown'
            tags = []
            
            # First try to extract from schema fields
            if dataset_name and 'unknown' not in dataset_name.lower():
                # Clean dataset name from schema
                if 'openbookqa' in dataset_name.lower():
                    specific_dataset = 'openbookqa'
                    tags.append('commonsense')
                elif 'mmlu' in dataset_name.lower():
                    specific_dataset = 'mmlu'
                    tags.append('knowledge')
                elif 'hellaswag' in dataset_name.lower():
                    specific_dataset = 'hellaswag'
                    tags.append('commonsense')
                elif 'gsm8k' in dataset_name.lower() or 'gsm' in dataset_name.lower():
                    specific_dataset = 'gsm8k'
                    tags.append('math')
                elif 'boolq' in dataset_name.lower():
                    specific_dataset = 'boolq'
                    tags.append('reading_comprehension')
                elif 'civil_comments' in dataset_name.lower():
                    specific_dataset = 'civil_comments'
                    tags.append('toxicity')
                else:
                    # Use the cleaned dataset name from schema
                    specific_dataset = dataset_name.split('.')[-1] if '.' in dataset_name else dataset_name
            
            # If schema didn't give us useful info, parse evaluation_id
            if specific_dataset == 'unknown' and eval_id:
                # Extract dataset name from evaluation_id (format varies)
                if 'civil_comments' in eval_id.lower():
                    specific_dataset = 'civil_comments'
                    tags.append('toxicity')
                elif 'mmlu' in eval_id.lower():
                    specific_dataset = 'mmlu'
                    tags.append('knowledge')
                elif 'hellaswag' in eval_id.lower():
                    specific_dataset = 'hellaswag'
                    tags.append('commonsense')
                elif 'gsm' in eval_id.lower():
                    specific_dataset = 'gsm8k'
                    tags.append('math')
                elif 'boolq' in eval_id.lower():
                    specific_dataset = 'boolq'
                    tags.append('reading_comprehension')
                elif 'openbookqa' in eval_id.lower():
                    specific_dataset = 'openbookqa'
                    tags.append('commonsense')
                else:
                    # Try to extract from the task portion of evaluation_id
                    parts = eval_id.split('_')
                    if len(parts) >= 3:
                        specific_dataset = parts[2].split(':')[0]  # Get task name before parameters
            
            # Add tags based on schema task_type
            if task_type:
                tags.append(task_type.lower())
            
            # Add tags based on evaluation_id patterns
            if eval_id:
                if 'multiple_choice' in eval_id.lower():
                    tags.append('multiple_choice')
                if 'generation' in eval_id.lower():
                    tags.append('generation')
                if 'demographic' in eval_id.lower():
                    tags.append('bias_evaluation')
                if 'ablation' in eval_id.lower():
                    tags.append('ablation_study')
            
            # Add language tag if not English
            if language and language.lower() != 'en':
                tags.append(f'lang_{language.lower()}')
                
            return specific_dataset, ','.join(set(tags)) if tags else 'general'  # Remove duplicates with set()
        
        # Apply the smart extraction
        logger.info("🧠 Extracting dataset and tag information from schema and evaluation_id...")
        df[['specific_dataset', 'tags']] = df.apply(
            lambda row: pd.Series(extract_dataset_and_tags(row)), axis=1
        )
        
        logger.info(f"✅ Extracted: {df['model_name'].nunique()} models, {df['specific_dataset'].nunique()} datasets")
        
        # Filter to only valid scores for statistics
        valid_df = df[df['score'].notna() & (df['score'] >= 0)]
        logger.info(f"📊 Found {len(valid_df)} records with valid scores out of {len(df)} total records")
        
        # Generate comprehensive statistics
        stats = {
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'source_repo': repo_id,
                'source_name': source_name,
                'total_evaluations': len(df),
                'generator_version': '1.2.0'  # Updated version for simplified extraction
            },
            'overview': {
                'total_evaluations': len(df),
                'unique_models': df['model_name'].nunique(),
                'unique_datasets': df['specific_dataset'].nunique(),
                'unique_model_dataset_combinations': len(df.groupby(['model_name', 'specific_dataset'])),
                'date_range': {
                    'earliest': df.index.min() if len(df) > 0 else None,
                    'latest': df.index.max() if len(df) > 0 else None
                }
            },
            'model_statistics': {},
            'dataset_statistics': {},
            'model_dataset_combinations': {},
            'performance_statistics': {}
        }
        
        # Model-level statistics (only for valid scores)
        if len(valid_df) > 0:
            model_stats = valid_df.groupby('model_name').agg({
                'score': ['count', 'mean', 'std', 'min', 'max']
            }).round(4)
            
            for model_name in model_stats.index:
                stats['model_statistics'][model_name] = {
                    'evaluation_count': int(model_stats.loc[model_name, ('score', 'count')]),
                    'average_score': float(model_stats.loc[model_name, ('score', 'mean')]),
                    'score_std': float(model_stats.loc[model_name, ('score', 'std')]) if pd.notna(model_stats.loc[model_name, ('score', 'std')]) else 0.0,
                    'min_score': float(model_stats.loc[model_name, ('score', 'min')]),
                    'max_score': float(model_stats.loc[model_name, ('score', 'max')])
                }
        
        # Dataset-level statistics (only for valid scores)
        if len(valid_df) > 0:
            dataset_stats = valid_df.groupby('specific_dataset').agg({
                'score': ['count', 'mean', 'std', 'min', 'max'],
                'tags': 'first'  # Get tags for each dataset
            }).round(4)
            
            for dataset_name in dataset_stats.index:
                stats['dataset_statistics'][dataset_name] = {
                    'evaluation_count': int(dataset_stats.loc[dataset_name, ('score', 'count')]),
                    'average_score': float(dataset_stats.loc[dataset_name, ('score', 'mean')]),
                    'score_std': float(dataset_stats.loc[dataset_name, ('score', 'std')]) if pd.notna(dataset_stats.loc[dataset_name, ('score', 'std')]) else 0.0,
                    'min_score': float(dataset_stats.loc[dataset_name, ('score', 'min')]),
                    'max_score': float(dataset_stats.loc[dataset_name, ('score', 'max')]),
                    'tags': dataset_stats.loc[dataset_name, ('tags', 'first')]
                }
        
        # Model-Dataset combination statistics (only for valid scores)
        if len(valid_df) > 0:
            combo_stats = valid_df.groupby(['model_name', 'specific_dataset']).agg({
                'score': ['count', 'mean', 'std', 'min', 'max'],
                'tags': 'first'
            }).round(4)
            
            for (model_name, dataset_name) in combo_stats.index:
                combo_key = f"{model_name}||{dataset_name}"
                stats['model_dataset_combinations'][combo_key] = {
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'evaluation_count': int(combo_stats.loc[(model_name, dataset_name), ('score', 'count')]),
                    'average_score': float(combo_stats.loc[(model_name, dataset_name), ('score', 'mean')]),
                    'score_std': float(combo_stats.loc[(model_name, dataset_name), ('score', 'std')]) if pd.notna(combo_stats.loc[(model_name, dataset_name), ('score', 'std')]) else 0.0,
                    'min_score': float(combo_stats.loc[(model_name, dataset_name), ('score', 'min')]),
                    'max_score': float(combo_stats.loc[(model_name, dataset_name), ('score', 'max')]),
                    'tags': combo_stats.loc[(model_name, dataset_name), ('tags', 'first')]
                }
        
        # Overall performance statistics (only for valid scores)
        valid_scores = df['score'].dropna()
        valid_scores = valid_scores[valid_scores >= 0]  # Remove any remaining negative values
        
        if len(valid_scores) > 0:
            stats['performance_statistics'] = {
                'overall_average': float(valid_scores.mean()),
                'overall_std': float(valid_scores.std()) if len(valid_scores) > 1 else 0.0,
                'overall_min': float(valid_scores.min()),
                'overall_max': float(valid_scores.max()),
                'valid_score_count': len(valid_scores),
                'total_score_count': len(df),
                'score_distribution': {
                    'percentile_25': float(valid_scores.quantile(0.25)),
                    'percentile_50': float(valid_scores.quantile(0.50)),
                    'percentile_75': float(valid_scores.quantile(0.75)),
                    'percentile_90': float(valid_scores.quantile(0.90)),
                    'percentile_95': float(valid_scores.quantile(0.95))
                }
            }
        else:
            stats['performance_statistics'] = {
                'message': 'No valid scores found',
                'valid_score_count': 0,
                'total_score_count': len(df)
            }
        
        logger.info(f"📈 Generated statistics for:")
        logger.info(f"   🤖 {len(stats['model_statistics'])} models")
        logger.info(f"   📊 {len(stats['dataset_statistics'])} datasets")
        logger.info(f"   � {len(stats['model_dataset_combinations'])} model-dataset combinations")
        logger.info(f"   🎯 {stats['metadata']['total_evaluations']} total evaluations")
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to generate statistics: {e}")
        raise


def upload_statistics(stats: dict, stats_repo_id: str, api: HfApi):
    """
    Upload statistics to the dedicated statistics repository.
    
    Args:
        stats: Statistics dictionary
        stats_repo_id: Repository ID for statistics storage
        api: HuggingFace API instance
    """
    logger.info(f"📤 Uploading statistics to {stats_repo_id}")
    
    try:
        # Create repository if it doesn't exist
        try:
            api.repo_info(stats_repo_id, repo_type="dataset")
            logger.info(f"📁 Repository exists: {stats_repo_id}")
        except Exception:
            logger.info(f"📁 Creating repository: {stats_repo_id}")
            api.create_repo(stats_repo_id, repo_type="dataset", exist_ok=True)
        
        # Convert statistics to DataFrame for parquet storage
        detailed_stats = []
        
        # Add model statistics
        for model_name, model_stats in stats['model_statistics'].items():
            detailed_stats.append({
                'type': 'model',
                'name': model_name,
                'evaluation_count': model_stats['evaluation_count'],
                'average_score': model_stats['average_score'],
                'score_std': model_stats['score_std'],
                'min_score': model_stats['min_score'],
                'max_score': model_stats['max_score']
            })
        
        # Add dataset statistics
        for dataset_name, dataset_stats in stats['dataset_statistics'].items():
            detailed_stats.append({
                'type': 'dataset',
                'name': dataset_name,
                'evaluation_count': dataset_stats['evaluation_count'],
                'average_score': dataset_stats['average_score'],
                'score_std': dataset_stats['score_std'],
                'min_score': dataset_stats['min_score'],
                'max_score': dataset_stats['max_score'],
                'tags': dataset_stats.get('tags', 'general')
            })
        
        # Add model-dataset combination statistics
        for combo_key, combo_stats in stats['model_dataset_combinations'].items():
            detailed_stats.append({
                'type': 'model_dataset_combo',
                'name': combo_key,
                'model_name': combo_stats['model_name'],
                'dataset_name': combo_stats['dataset_name'],
                'evaluation_count': combo_stats['evaluation_count'],
                'average_score': combo_stats['average_score'],
                'score_std': combo_stats['score_std'],
                'min_score': combo_stats['min_score'],
                'max_score': combo_stats['max_score'],
                'tags': combo_stats.get('tags', 'general')
            })
        
        # Save detailed statistics as parquet
        if detailed_stats:
            df_stats = pd.DataFrame(detailed_stats)
            parquet_path = "detailed_statistics.parquet"
            df_stats.to_parquet(parquet_path, index=False)
            
            api.upload_file(
                path_or_fileobj=parquet_path,
                path_in_repo="detailed_statistics.parquet",
                repo_id=stats_repo_id,
                repo_type="dataset"
            )
            
            os.remove(parquet_path)  # Clean up local file
            logger.info(f"✅ Uploaded detailed statistics parquet")
        
        # Save metadata as JSON
        metadata_path = "detailed_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        api.upload_file(
            path_or_fileobj=metadata_path,
            path_in_repo="detailed_metadata.json",
            repo_id=stats_repo_id,
            repo_type="dataset"
        )
        
        os.remove(metadata_path)  # Clean up local file
        logger.info(f"✅ Uploaded metadata JSON")
        
        logger.info(f"🎉 Statistics successfully uploaded to {stats_repo_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to upload statistics: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate comprehensive statistics from evaluation data")
    parser.add_argument("--repo-id", type=str, required=True, help="Source repository ID")
    parser.add_argument("--stats-repo-id", type=str, required=True, help="Statistics repository ID")
    parser.add_argument("--source-name", type=str, default="helm", help="Source name")
    parser.add_argument("--dry-run", action="store_true", help="Generate statistics without uploading")
    
    args = parser.parse_args()
    
    # Check for HF token (only if not dry-run)
    if not args.dry_run:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            logger.error("❌ HF_TOKEN environment variable required")
            return 1
    
    try:
        # Generate comprehensive statistics
        logger.info(f"🚀 Starting statistics generation for {args.source_name}")
        stats = generate_comprehensive_statistics(args.repo_id, args.source_name)
        
        if args.dry_run:
            logger.info("🔍 DRY RUN - Statistics preview:")
            logger.info(f"📊 Overview: {stats['overview']}")
            logger.info(f"📊 Datasets found: {list(stats['dataset_statistics'].keys())}")
            logger.info(f"🤖 Models found: {list(stats['model_statistics'].keys())[:5]}...")  # Show first 5
            logger.info(f"🔄 Example model-dataset combos: {list(stats['model_dataset_combinations'].keys())[:3]}...")  # Show first 3
            return 0
        
        # Initialize HuggingFace API
        api = HfApi(token=hf_token)
        
        # Upload statistics
        upload_statistics(stats, args.stats_repo_id, api)
        
        logger.info(f"🎉 Statistics generation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Statistics generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
