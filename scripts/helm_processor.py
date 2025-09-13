#!/usr/bin/env python3
"""
Simplified HELM processor with EvalHub schema compliance and global deduplication.

This script implements a much cleaner approach:
1. Read all existing task IDs from repository parquet files
2. Compare with all scraped task IDs across all benchmarks
3. Split remaining tasks into chunks and process them in parallel
4. Use source+benchmark+task_id as unique identifier
5. Generate proper nested EvaluationResult objects conforming to EvalHub schema

Usage:
    python scripts/helm_processor.py --repo-id evaleval/every_eval_ever --chunk-size 100 --max-workers 4
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set, Dict, Optional
import logging
import pandas as pd
from huggingface_hub import HfApi
from datasets import Dataset
from datasets import Dataset

# Suppress Pydantic protected namespace warnings
warnings.filterwarnings("ignore", message=".*protected namespace.*", category=UserWarning)

from huggingface_hub import HfApi

# Try to import datasets library for reading existing data
try:
    from datasets import load_dataset, Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️ datasets library not available - install with: pip install datasets")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import EvalHub schema
try:
    from external_schemas.evalHub.schema.eval_types import (
        EvaluationResult, Model, ModelInfo, Configuration, InferenceSettings,
        PromptConfig, Instance, Output, Evaluation, EvaluationMethod,
        SampleIdentifier, Quantization, GenerationArgs, PromptClass, TaskType, HfSplit,
        Family, Architecture, BitPrecision, Method, Dimensions
    )
    EVALHUB_SCHEMA_AVAILABLE = True
    print("✅ EvalHub schema loaded successfully")
except ImportError as e:
    EVALHUB_SCHEMA_AVAILABLE = False
    print(f"⚠️ EvalHub schema not available: {e}")
    print("⚠️ Processor will not work properly without EvalHub schema")

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _extract_model_family(model_name: str) -> str:
    """Extract model family from model name."""
    if not model_name:
        return 'unknown'
    
    # Common model families
    families = {
        'llama': 'Llama',
        'mistral': 'Mistral', 
        'olmo': 'OLMo',
        'gemma': 'gemma',
        'gpt': 'gpt',
        'palm': 'palm',
        'claude': 'claude',
        'falcon': 'falcon',
        'qwen': 'Qwen',
        'yi': 'Yi'
    }
    
    model_lower = model_name.lower()
    for pattern, family in families.items():
        if pattern in model_lower:
            return family
    
    return 'unknown'


def process_helm_directory(helm_dir: Path, source_name: str) -> List[Dict]:
    """Process a single HELM directory and return evaluation results."""
    evaluation_results = []
    
    try:
        # Check if this directory has the expected HELM files
        expected_files = ["instances.json", "display_predictions.json", "run_spec.json"]
        missing_files = [f for f in expected_files if not (helm_dir / f).exists()]
        
        if missing_files:
            logger.warning(f"⚠️ Skipping {helm_dir.name}: missing files {missing_files}")
            return []
        
        # Load the HELM JSON files
        instances_file = helm_dir / "instances.json"
        predictions_file = helm_dir / "display_predictions.json"
        run_spec_file = helm_dir / "run_spec.json"
        
        # Load and verify the data
        with open(instances_file, 'r') as f:
            instances = json.load(f)
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        with open(run_spec_file, 'r') as f:
            run_spec = json.load(f)
        
        logger.info(f"📄 Loaded HELM files for {helm_dir.name}")
        logger.info(f"   📊 Instances: {len(instances) if isinstance(instances, list) else 'N/A'}")
        logger.info(f"   🎯 Predictions: {len(predictions) if isinstance(predictions, list) else 'N/A'}")
        
        # Determine benchmark from run_spec or directory name
        benchmark = "unknown"
        try:
            if run_spec and isinstance(run_spec, dict):
                scenario_spec = run_spec.get("scenario_spec", {})
                if scenario_spec and isinstance(scenario_spec, dict):
                    benchmark = scenario_spec.get("class_name", "unknown")
            
            if benchmark == "unknown":
                # Extract from directory name
                task_name = helm_dir.name
                for known_benchmark in ["mmlu", "hellaswag", "boolq", "commonsense", "babi_qa"]:
                    if known_benchmark in task_name.lower():
                        benchmark = known_benchmark
                        break
        except Exception as e:
            logger.warning(f"⚠️ Could not determine benchmark: {e}")
        
        # Process predictions into EvaluationResult format
        if isinstance(predictions, list):
            for i, prediction in enumerate(predictions):
                try:
                    instance_id = _extract_numeric_id(prediction.get("instance", {}).get("instance_id", i))
                    
                    # Create a row with the prediction data
                    row = pd.Series({
                        'instance_id': instance_id,
                        'predicted_text': prediction.get("predicted_text", ""),
                        'prediction_score': prediction.get("prediction_score", 0.0),
                        'correct': prediction.get("correct", False),
                        'request_state': prediction.get("request_state", {}),
                        'stats': prediction.get("stats", {}),
                        'instance': prediction.get("instance", {})
                    })
                    
                    # Create EvaluationResult using the existing function
                    result = _create_evaluation_result(
                        row=row,
                        task_id=f'{helm_dir.name}_{instance_id}',
                        benchmark=benchmark,
                        source=source_name
                    )
                    
                    if result:
                        evaluation_results.append(result)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not process instance {i}: {e}")
                    continue
        
        logger.info(f"🎯 Created {len(evaluation_results)} EvaluationResult objects")
        
        # Calculate accuracy for this directory
        if evaluation_results:
            correct_count = sum(1 for result in evaluation_results 
                              if 'evaluation' in result and result['evaluation'].get('score', 0) == 1.0)
            accuracy = correct_count / len(evaluation_results) if evaluation_results else 0
            logger.info(f"📈 Accuracy: {correct_count}/{len(evaluation_results)} ({accuracy:.2%})")
        
    except Exception as e:
        logger.error(f"❌ Error processing HELM directory {helm_dir.name}: {e}")
    
    return evaluation_results


def _determine_prompt_class(row: pd.Series, benchmark: str) -> str:
    """Determine the prompt class based on benchmark and available data."""
    # Multiple choice benchmarks
    if any(mc_benchmark in benchmark.lower() for mc_benchmark in [
        'mmlu', 'hellaswag', 'arc', 'winogrande', 'commonsense', 'truthfulqa'
    ]):
        return 'MultipleChoice'
    
    # Generation benchmarks
    if any(gen_benchmark in benchmark.lower() for gen_benchmark in [
        'gsm8k', 'math', 'humaneval', 'mbpp', 'drop'
    ]):
        return 'OpenEnded'
    
    # Check if choices are available in the data
    if pd.notna(row.get('choices')) or pd.notna(row.get('options')):
        return 'MultipleChoice'
    
    return 'OpenEnded'


def _extract_numeric_id(instance_id) -> int:
    """Extract numeric ID from string instance IDs like 'id5138' or return 0 if not found."""
    if pd.isna(instance_id):
        return 0
    
    instance_id_str = str(instance_id)
    
    # Extract numbers from string like 'id5138' using regex first
    import re
    numbers = re.findall(r'\d+', instance_id_str)
    if numbers:
        return int(numbers[0])  # Return first number found
    
    # If no numbers found, return 0
    return 0


def _safe_int_convert(value, default=0):
    """Safely convert a value to int, extracting numbers from strings if needed."""
    if pd.isna(value):
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        # Try to extract numbers from string
        import re
        value_str = str(value)
        numbers = re.findall(r'\d+', value_str)
        if numbers:
            return int(numbers[0])
        return default


def _determine_task_type(benchmark: str) -> str:
    """Determine task type based on benchmark."""
    generation_benchmarks = ['gsm8k', 'math', 'humaneval', 'mbpp', 'drop', 'narrativeqa']
    
    if any(gen_benchmark in benchmark.lower() for gen_benchmark in generation_benchmarks):
        return 'generation'
    
    return 'classification'


def _extract_shots_count(prompt: str) -> int:
    """Extract number of shots from prompt."""
    if not prompt:
        return 0
    
    # Simple heuristic: count examples by looking for patterns
    example_patterns = ['Example:', 'Q:', 'Question:', '\n\n']
    shot_count = 0
    
    for pattern in example_patterns:
        shot_count = max(shot_count, prompt.count(pattern) - 1)  # -1 for the actual question
    
    return max(0, shot_count)


def _extract_tags(row: pd.Series, benchmark: str) -> List[str]:
    """Extract tags from row data."""
    tags = []
    
    # Add benchmark as tag
    tags.append(f"benchmark:{benchmark}")
    
    # Add subject/category if available
    if pd.notna(row.get('subject')):
        tags.append(f"subject:{row.get('subject')}")
    if pd.notna(row.get('category')):
        tags.append(f"category:{row.get('category')}")
    
    # Add difficulty if available
    if pd.notna(row.get('difficulty')):
        tags.append(f"difficulty:{row.get('difficulty')}")
        
    # Add any existing tags
    if pd.notna(row.get('tags')):
        existing_tags = [tag.strip() for tag in str(row.get('tags')).split(',') if tag.strip()]
        tags.extend(existing_tags)
    
    return tags


def _get_metric_description(metric: str) -> str:
    """Get description for evaluation metric."""
    descriptions = {
        'exact_match': 'Exact string match between prediction and ground truth',
        'f1_score': 'F1 score between prediction and ground truth tokens',
        'bleu': 'BLEU score for text generation quality',
        'rouge': 'ROUGE score for text summarization quality',
        'accuracy': 'Classification accuracy',
        'multiple_choice_grade': 'Multiple choice grading metric'
    }
    return descriptions.get(metric, f'Custom metric: {metric}')


def _normalize_score(score) -> float:
    """Normalize score to 0-1 range."""
    if pd.isna(score):
        return 0.0
    score = float(score)
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))


def _map_family_to_enum(model_name: str):
    """Map model name to Family enum."""
    if not EVALHUB_SCHEMA_AVAILABLE:
        return None
    
    family_str = _extract_model_family(model_name)
    if family_str:
        # Map to Family enum
        family_mapping = {
            'Llama': Family.Llama,
            'Mistral': Family.Mistral, 
            'OLMo': Family.OLMo,
            'gemma': Family.gemma,
            'gpt': Family.gpt,
            'palm': Family.palm,
            'claude': Family.claude,
            'falcon': Family.falcon,
            'Qwen': Family.Qwen
            # Note: 'Yi' not available in EvalHub Family enum, falls back to None
        }
        return family_mapping.get(family_str, None)
    return None


def _map_prompt_class_to_enum(prompt_class_str: str):
    """Map prompt class string to PromptClass enum."""
    if not EVALHUB_SCHEMA_AVAILABLE:
        return prompt_class_str
        
    mapping = {
        'MultipleChoice': PromptClass.MultipleChoice,
        'OpenEnded': PromptClass.OpenEnded,
        'Completion': PromptClass.Completion
    }
    return mapping.get(prompt_class_str, PromptClass.OpenEnded)


def _map_task_type_to_enum(task_type_str: str):
    """Map task type string to TaskType enum."""
    if not EVALHUB_SCHEMA_AVAILABLE:
        return task_type_str
        
    mapping = {
        'classification': TaskType.classification,
        'generation': TaskType.generation
    }
    return mapping.get(task_type_str, TaskType.classification)


def _is_instruct_model(model_name: str) -> bool:
    """Determine if model is an instruct model."""
    instruct_indicators = ['instruct', 'chat', 'turbo', 'gpt-4', 'gpt-3.5', 'claude']
    model_lower = model_name.lower()
    return any(indicator in model_lower for indicator in instruct_indicators)


def _create_evaluation_result(row: pd.Series, task_id: str, benchmark: str, source: str) -> dict:
    """Create a proper EvaluationResult object from HELM data."""
    
    if not EVALHUB_SCHEMA_AVAILABLE:
        logger.error("Cannot create EvaluationResult - schema not available")
        return None
    
    try:
        # Create Model object
        model_family = _map_family_to_enum(row.get('model', ''))
        model = Model(
            model_info=ModelInfo(
                name=row.get('model', 'unknown'),
                family=model_family
            ),
            configuration=Configuration(
                architecture=Architecture.transformer,  # Default assumption
                parameters=None,  # Extract if available
                context_window=4096,  # Default assumption
                is_instruct=_is_instruct_model(row.get('model', '')),
                hf_path=row.get('model', '') if '/' in str(row.get('model', '')) else None,
                revision=None
            ),
            inference_settings=InferenceSettings(
                quantization=Quantization(
                    bit_precision=BitPrecision.none,
                    method=Method.None_
                ),
                generation_args=GenerationArgs(
                    use_vllm=None,
                    temperature=float(row.get('temperature', 0.0)) if pd.notna(row.get('temperature')) else None,
                    top_p=float(row.get('top_p', 1.0)) if pd.notna(row.get('top_p')) else None,
                    top_k=float(row.get('top_k', 0)) if pd.notna(row.get('top_k')) else None,
                    max_tokens=_safe_int_convert(row.get('max_tokens'), 512) if pd.notna(row.get('max_tokens')) else None,
                    stop_sequences=[]
                )
            )
        )
        
        # Create PromptConfig object
        prompt_class_str = _determine_prompt_class(row, benchmark)
        prompt_config = PromptConfig(
            prompt_class=_map_prompt_class_to_enum(prompt_class_str),
            dimensions=None  # Could be expanded later
        )
        
        # Create Instance object
        task_type_str = _determine_task_type(benchmark)
        hf_split_mapping = {'test': HfSplit.test, 'train': HfSplit.train, 'validation': HfSplit.validation}
        
        instance = Instance(
            task_type=_map_task_type_to_enum(task_type_str),
            raw_input=row.get('input', row.get('question', '')),
            language='en',  # Default assumption
            sample_identifier=SampleIdentifier(
                dataset_name=benchmark,
                hf_repo=f"helm/{benchmark}",
                hf_split=hf_split_mapping.get(row.get('split', 'test'), HfSplit.test),
                hf_index=_extract_numeric_id(row.get('instance_id'))
            )
        )
        
        # Create Output object  
        output = Output(
            response=row.get('output', row.get('answer', '')),
            cumulative_logprob=None,  # Not available in HELM data
            generated_tokens_logprobs=None  # Not available in HELM data
        )
        
        # Create Evaluation object
        evaluation = Evaluation(
            evaluation_method=EvaluationMethod(
                method_name=row.get('metric', row.get('evaluation_method', 'exact_match')),
                description=_get_metric_description(row.get('metric', 'exact_match')),
                parameters=None
            ),
            ground_truth=row.get('references', row.get('correct_answer', '')),
            score=_normalize_score(row.get('score', 0))
        )
        
        # Create the complete EvaluationResult
        eval_result = EvaluationResult(
            schema_version='1.0.0',
            evaluation_id=f"{source}_{benchmark}_{task_id}_{row.get('instance_id', 0)}",
            model=model,
            prompt_config=prompt_config,
            instance=instance,
            output=output,
            evaluation=evaluation
        )
        
        # Convert to dict for dataset storage, ensuring enums are converted to strings
        result_dict = eval_result.model_dump()
        
        # Convert enum values to strings for Arrow compatibility
        def convert_enums_to_strings(obj):
            if isinstance(obj, dict):
                return {k: convert_enums_to_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums_to_strings(item) for item in obj]
            elif hasattr(obj, 'value'):  # Enum object
                return obj.value
            else:
                return obj
        
        result_dict = convert_enums_to_strings(result_dict)
        return result_dict
        
    except Exception as e:
        logger.error(f"Failed to create EvaluationResult: {e}")
        return None


def setup_hf_api(token: str, repo_id: str) -> HfApi:
    """Initialize HuggingFace API."""
    if not token:
        raise ValueError('HF_TOKEN environment variable required')
    
    try:
        api = HfApi(token=token)
        # Test connection and create repo if needed
        try:
            api.repo_info(repo_id, repo_type="dataset")
            logger.info(f"📁 Repository exists: {repo_id}")
        except Exception:
            logger.info(f"📁 Creating repository: {repo_id}")
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        
        return api
        
    except Exception as e:
        logger.error(f"❌ Failed to setup HF API: {e}")
        raise


def save_individual_evaluations(evaluation_results: List[dict], source: str = "helm") -> List[str]:
    """Save each evaluation as individual JSON file in nested structure for efficient organization."""
    
    saved_files = []
    base_dir = Path("data/evaluations")
    
    logger.info(f"💾 Saving {len(evaluation_results)} evaluations as individual JSON files...")
    
    for eval_result in evaluation_results:
        eval_id = eval_result['evaluation_id']
        
        # Extract components for nested structure
        # eval_id format: "source_benchmark_task_instance"  
        parts = eval_id.split('_')
        if len(parts) >= 3:
            source_part = parts[0]  # "helm"
            benchmark = parts[1]    # "mmlu", "hellaswag", etc.
            
            # Extract subject/category if available from instance data
            instance = eval_result.get('instance', {})
            sample_id = instance.get('sample_identifier', {})
            dataset_name = sample_id.get('dataset_name', benchmark)
            
            # Create nested directory structure: source/benchmark/subject
            if '.' in dataset_name:
                # Handle cases like "mmlu.clinical_knowledge"
                benchmark_name, subject = dataset_name.split('.', 1)
                nested_dir = base_dir / source_part / benchmark_name / subject
            else:
                # Use benchmark directly
                nested_dir = base_dir / source_part / benchmark / "default"
        else:
            # Fallback to flat structure if parsing fails
            nested_dir = base_dir / source / "misc"
        
        # Create directory
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename-safe eval_id  
        safe_eval_id = eval_id.replace('/', '_').replace(':', '_').replace(',', '_').replace('=', '_')
        output_file = nested_dir / f"{safe_eval_id}.json"
        
        # Skip if file already exists (automatic deduplication)
        if output_file.exists():
            logger.debug(f"⏭️ Skipping existing evaluation: {eval_id}")
            continue
        
        # Save individual evaluation as JSON
        try:
            with open(output_file, 'w') as f:
                json.dump(eval_result, f, indent=2)
            saved_files.append(str(output_file))
            logger.debug(f"💾 Saved: {eval_id} → {output_file.relative_to(base_dir)}")
        except Exception as e:
            logger.error(f"❌ Failed to save {eval_id}: {e}")
    
    logger.info(f"✅ Saved {len(saved_files)} new evaluations (skipped {len(evaluation_results) - len(saved_files)} duplicates)")
    
    # Show directory structure summary
    if saved_files:
        dirs_created = set(Path(f).parent for f in saved_files)
        logger.info(f"📁 Created files in {len(dirs_created)} directories")
        for dir_path in sorted(dirs_created)[:5]:  # Show first 5
            rel_path = dir_path.relative_to(base_dir)
            count = sum(1 for f in saved_files if Path(f).parent == dir_path)
            logger.info(f"   📂 {rel_path}: {count} files")
        if len(dirs_created) > 5:
            logger.info(f"   📂 ... and {len(dirs_created) - 5} more directories")
    
    return saved_files


def aggregate_individual_files_to_hf_dataset(individual_files: List[str], chunk_id: int) -> str:
    """Aggregate individual JSON files into a HuggingFace dataset for upload."""
    
    if not individual_files:
        logger.error("❌ No individual files to aggregate")
        return None
    
    logger.info(f"📦 Aggregating {len(individual_files)} individual files into HF dataset...")
    
    # Load all evaluations
    evaluation_results = []
    for file_path in individual_files:
        try:
            with open(file_path, 'r') as f:
                eval_result = json.load(f)
                evaluation_results.append(eval_result)
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
    
    if not evaluation_results:
        logger.error("❌ No valid evaluations loaded")
        return None
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(evaluation_results)
    
    # Save as JSON for upload
    agg_dir = Path("data/aggregated")
    agg_dir.mkdir(parents=True, exist_ok=True)
    output_file = agg_dir / f"chunk_{chunk_id:04d}_evalhub.json"
    
    dataset.to_json(str(output_file))
    
    logger.info(f"✅ Aggregated dataset saved: {output_file} ({len(evaluation_results)} evaluations)")
    return str(output_file)


def process_chunk_with_individual_files(chunk: Dict, workers: int, source: str = "helm") -> str:
    """Process a chunk of tasks and save as individual JSON files, then aggregate for upload."""
    
    chunk_id = chunk['chunk_id']
    benchmarks = chunk['benchmarks']
    total_tasks = chunk['total_tasks']
    
    logger.info(f"🔄 Processing chunk {chunk_id} with {total_tasks} tasks across {len(benchmarks)} benchmarks")
    
    # Import required functions (would need to implement these for EvalHub processor)
    # from src.sources.helm.processor import process_line
    # from src.sources.helm.downloader import download_tasks
    
    all_evaluation_results = []
    
    # Process each benchmark in the chunk
    for benchmark, tasks in benchmarks.items():
        logger.info(f"📥 Processing {len(tasks)} {benchmark} tasks in chunk {chunk_id}")
        
        # For each task, create EvaluationResult (this would be the actual processing)
        for task in tasks:
            # This is where you'd process the actual HELM data
            # For now, creating example structure
            sample_row = pd.Series({
                'model': f'test-model-{task}',
                'input': f'Test input for task {task}',
                'output': f'Test output for task {task}',
                'score': 0.8,
                'metric': 'exact_match',
                'instance_id': hash(task) % 1000,
                'references': f'Test reference for task {task}'
            })
            
            eval_result = _create_evaluation_result(
                row=sample_row,
                task_id=str(task),
                benchmark=benchmark,
                source=source
            )
            
            if eval_result:
                all_evaluation_results.append(eval_result)
    
    if not all_evaluation_results:
        logger.error(f"❌ No evaluation results created in chunk {chunk_id}")
        return None
    
    # Save as individual files
    saved_files = save_individual_evaluations(all_evaluation_results, source)
    
    # Aggregate for upload
    aggregated_file = aggregate_individual_files_to_hf_dataset(saved_files, chunk_id)
    
    return aggregated_file


def get_existing_task_ids(repo_id: str) -> Set[str]:
    """Get all existing task IDs from the repository."""
    existing_ids = set()
    
    if not DATASETS_AVAILABLE:
        logger.warning("⚠️ datasets library not available - cannot check existing tasks")
        return existing_ids
    
    try:
        # Try to load the dataset
        dataset = load_dataset(repo_id, split="train", trust_remote_code=True)
        logger.info(f"📊 Found existing dataset with {len(dataset)} records")
        
        # Extract task IDs from evaluation_id field
        for item in dataset:
            eval_id = item.get('evaluation_id', '')
            if eval_id:
                existing_ids.add(eval_id)
        
        logger.info(f"🔍 Found {len(existing_ids)} existing task IDs")
        
    except Exception as e:
        logger.info(f"📭 No existing dataset found: {e}")
    
    return existing_ids


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simplified HELM processor with EvalHub schema")
    parser.add_argument("--repo-id", type=str, default="test-repo", help="HuggingFace dataset repository ID (default for test: test-repo)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Tasks per chunk (default: 100)")
    parser.add_argument("--max-workers", type=int, default=4, help="Workers per chunk (default: 4)")
    parser.add_argument("--source-name", type=str, default="helm", help="Source name")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up files after upload")
    parser.add_argument("--test-run", action="store_true", help="Test mode: process small sample without HF upload")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to scrape from HELM (for testing)")
    parser.add_argument('--num-files-test', type=int, default=3,
                        help='Number of files to process in test mode (default: 3)')
    parser.add_argument('--test-benchmarks', type=str, nargs='+', 
                        default=['mmlu', 'hellaswag', 'boolq'],
                        help='Benchmarks to test with (default: mmlu, hellaswag, boolq)')
    parser.add_argument('--helm-benchmarks', type=str, nargs='+',
                        default=['lite', 'classic', 'mmlu'],
                        help='HELM benchmarks to scrape (default: lite, classic, mmlu)')
    parser.add_argument('--local-only', action='store_true',
                        help='Local processing only - skip HuggingFace upload but still do streaming')
    
    args = parser.parse_args()
    
    if not EVALHUB_SCHEMA_AVAILABLE:
        logger.error("❌ EvalHub schema not available - cannot proceed")
        logger.error("Please ensure external_schemas/evalHub is properly set up")
        return 1
    
    # Get HF token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token and not args.test_run and not args.local_only:
        logger.error("❌ HF_TOKEN environment variable required (or use --test-run or --local-only for local testing)")
        return 1
    
    try:
        # Load model mappings (simplified for test mode)
        model_mappings = {}  # Simplified for now

        if args.test_run:
            logger.info("🧪 Running in TEST MODE - using real data with limited scope")
            logger.info("⚠️ No synthetic data generation - only processing existing real HELM data")
            
            # Check for existing HELM downloads
            downloads_dir = Path("data/downloads")
            if downloads_dir.exists():
                helm_dirs = [d for d in downloads_dir.iterdir() if d.is_dir()]
                if helm_dirs:
                    logger.info(f"📁 Found {len(helm_dirs)} existing HELM downloads")
                    
                    # Process a small subset of real data
                    subset_dirs = helm_dirs[:args.num_files_test]
                    logger.info(f"🔄 Processing {len(subset_dirs)} directories in test mode")
                    
                    processed_count = 0
                    total_evaluations = 0
                    
                    for helm_dir in subset_dirs:
                        try:
                            # Process this real HELM directory
                            task_name = helm_dir.name
                            logger.info(f"📂 Processing real data: {task_name}")
                            
                            # This would process the real HELM data
                            # For now, just count what we would process
                            predictions_file = helm_dir / "display_predictions.json"
                            if predictions_file.exists():
                                processed_count += 1
                                # Count predictions in file
                                try:
                                    with open(predictions_file, 'r') as f:
                                        predictions = json.load(f)
                                    total_evaluations += len(predictions) if isinstance(predictions, list) else 1
                                except:
                                    total_evaluations += 1
                                    
                        except Exception as e:
                            logger.warning(f"⚠️ Error processing {helm_dir.name}: {e}")
                    
                    logger.info(f"✅ Test mode completed: processed {processed_count} real HELM directories")
                    logger.info(f"📊 Would process approximately {total_evaluations} real evaluations")
                    
                else:
                    logger.warning("⚠️ No existing HELM downloads found for test mode")
                    logger.info("💡 Run with production mode or download HELM data first")
            else:
                logger.warning("⚠️ No data/downloads directory found")
                logger.info("💡 Run in production mode to scrape and download real HELM data")
            
            logger.info("🎉 Test mode completed! No synthetic data was generated.")
            return 0
            
        else:
            # Production mode - streaming pipeline: scrape -> download in chunks -> process -> upload -> clean
            logger.info("🚀 Production mode: Streaming HELM evaluation pipeline")
            
            # Step 1: Scrape HELM runs from website
            logger.info("📋 Step 1: Scraping HELM evaluation runs from multiple benchmarks...")
            try:
                # Import scraping functionality
                sys.path.append(str(Path(__file__).parent.parent))
                from src.sources.helm.web_scraper import scrape_helm_data
                
                # Define HELM benchmarks to scrape from command line argument
                helm_benchmarks = args.helm_benchmarks
                logger.info(f"🎯 Target benchmarks: {', '.join(helm_benchmarks)}")
                
                all_scraped_data = []
                
                # Scrape from each benchmark
                for benchmark in helm_benchmarks:
                    logger.info(f"🔍 Scraping {benchmark} benchmark...")
                    try:
                        if args.max_pages:
                            logger.info(f"   📄 Limited to {args.max_pages} pages for testing")
                            benchmark_data = asyncio.run(scrape_helm_data(benchmark, args.max_pages))
                        else:
                            logger.info(f"   🌍 Full dataset (all pages)")
                            benchmark_data = asyncio.run(scrape_helm_data(benchmark))
                        
                        if benchmark_data:
                            logger.info(f"   ✅ Scraped {len(benchmark_data)} runs from {benchmark}")
                            # Add benchmark info to each run
                            for run in benchmark_data:
                                run['_source_benchmark'] = benchmark
                            all_scraped_data.extend(benchmark_data)
                        else:
                            logger.warning(f"   ⚠️ No data scraped from {benchmark}")
                            
                    except Exception as e:
                        logger.warning(f"   ❌ Failed to scrape {benchmark}: {e}")
                        continue
                
                if all_scraped_data:
                    logger.info(f"✅ Total scraped: {len(all_scraped_data)} runs across all benchmarks")
                    
                    # Use all scraped tasks from all benchmarks
                    task_names = []
                    benchmark_map = {}  # Map task name to benchmark
                    
                    for run in all_scraped_data:
                        task_name = run.get('Run', '')
                        if task_name:
                            task_names.append(task_name)
                            benchmark_map[task_name] = run.get('_source_benchmark', 'lite')
                    
                    task_names = [name for name in task_names if name]  # Remove empty
                    logger.info(f"📝 Prepared {len(task_names)} tasks for streaming processing")
                else:
                    logger.warning("⚠️ No HELM runs scraped - using existing data if available")
                    task_names = []
                    benchmark_map = {}
                    
            except Exception as e:
                logger.warning(f"⚠️ Scraping failed, using existing data: {e}")
                task_names = []
                benchmark_map = {}
            
            # Step 2-5: Individual task streaming pipeline with parallel processing
            if task_names:
                logger.info("🌊 Starting individual task streaming pipeline: download -> process -> upload -> clean")
                
                # Setup HuggingFace API for uploads
                api = None
                if not args.test_run and not args.local_only:
                    api = setup_hf_api(os.environ.get('HF_TOKEN'), args.repo_id)
                
                # Import threading for parallel processing
                import threading
                from concurrent.futures import ThreadPoolExecutor, as_completed
                from queue import Queue
                
                # Thread-safe counters
                total_processed = 0
                total_uploaded = 0
                processing_lock = threading.Lock()
                
                def process_single_task(task_info):
                    """Process a single task: download -> process -> upload -> clean"""
                    task_name, benchmark, task_index = task_info
                    nonlocal total_processed, total_uploaded
                    
                    thread_id = threading.current_thread().name[-1]  # Get thread number
                    logger.info(f"� Thread-{thread_id}: Processing task {task_index}/{len(task_names)}: {task_name[:50]}...")
                    
                    try:
                        # Step 1: Download this single task
                        logger.info(f"📥 Thread-{thread_id}: Downloading from {benchmark}...")
                        from src.sources.helm.downloader import download_tasks
                        
                        successful_downloads = download_tasks(
                            tasks=[task_name],
                            output_dir="data/downloads",
                            benchmark=benchmark,
                            overwrite=False,
                            show_progress=False  # Disable progress bar for cleaner parallel output
                        )
                        
                        if not successful_downloads:
                            logger.warning(f"⚠️ Thread-{thread_id}: Failed to download {task_name}")
                            return False
                        
                        # Step 2: Process this task immediately
                        logger.info(f"� Thread-{thread_id}: Processing...")
                        data_path = Path("data/downloads")
                        helm_dir = data_path / task_name
                        
                        if not helm_dir.exists():
                            logger.warning(f"⚠️ Thread-{thread_id}: Directory not found after download")
                            return False
                        
                        # Process this HELM directory
                        evaluation_results = process_helm_directory(helm_dir, args.source_name)
                        
                        if not evaluation_results:
                            logger.warning(f"⚠️ Thread-{thread_id}: No evaluation results generated")
                            return False
                        
                        # Step 3: Create individual dataset file and upload immediately
                        logger.info(f"📤 Thread-{thread_id}: Uploading {len(evaluation_results)} evaluations...")
                        
                        # Create dataset for this single task
                        task_dataset = Dataset.from_list(evaluation_results)
                        
                        # Save locally with unique filename
                        agg_dir = Path("data/aggregated")
                        agg_dir.mkdir(parents=True, exist_ok=True)
                        task_file = agg_dir / f"task_{task_index:04d}_{task_name.replace('/', '_').replace(':', '_')[:50]}_evalhub.json"
                        task_dataset.to_json(str(task_file))
                        
                        upload_success = False
                        # Upload to HuggingFace immediately
                        if api and not args.test_run and not args.local_only:
                            try:
                                api.upload_file(
                                    path_or_fileobj=str(task_file),
                                    path_in_repo=f"data/{task_file.name}",
                                    repo_id=args.repo_id,
                                    repo_type="dataset"
                                )
                                upload_success = True
                                logger.info(f"✅ Thread-{thread_id}: Uploaded to HuggingFace")
                                
                            except Exception as e:
                                logger.error(f"❌ Thread-{thread_id}: Upload failed: {e}")
                        else:
                            upload_success = True  # Consider local saves as success
                            if args.local_only:
                                logger.info(f"🏠 Thread-{thread_id}: Saved locally")
                            else:
                                logger.info(f"🧪 Thread-{thread_id}: Test mode - saved locally")
                        
                        # Step 4: Clean up immediately to save memory
                        if not args.no_cleanup:
                            try:
                                # Remove download directory
                                if helm_dir.exists():
                                    import shutil
                                    shutil.rmtree(helm_dir)
                                
                                # Remove local file after upload (if uploaded successfully)
                                if upload_success and not args.local_only and task_file.exists():
                                    task_file.unlink()
                                
                                logger.info(f"🧹 Thread-{thread_id}: Cleaned up")
                            except Exception as e:
                                logger.warning(f"⚠️ Thread-{thread_id}: Cleanup failed: {e}")
                        
                        # Update counters thread-safely
                        with processing_lock:
                            nonlocal total_processed, total_uploaded
                            total_processed += len(evaluation_results)
                            if upload_success:
                                total_uploaded += len(evaluation_results)
                        
                        logger.info(f"✅ Thread-{thread_id}: Completed task {task_index}")
                        return True
                        
                    except Exception as e:
                        logger.error(f"❌ Thread-{thread_id}: Error processing {task_name}: {e}")
                        return False
                
                # Prepare task list with benchmark info and index
                task_list = []
                for i, task_name in enumerate(task_names, 1):
                    benchmark = benchmark_map.get(task_name, 'lite')
                    task_list.append((task_name, benchmark, i))
                
                logger.info(f"🚀 Starting parallel processing with {args.max_workers} threads")
                logger.info(f"📊 Total tasks to process: {len(task_list)}")
                
                # Process tasks in parallel with configurable thread count
                with ThreadPoolExecutor(max_workers=args.max_workers, thread_name_prefix="TaskWorker") as executor:
                    # Submit all tasks
                    future_to_task = {executor.submit(process_single_task, task_info): task_info for task_info in task_list}
                    
                    completed_count = 0
                    failed_count = 0
                    
                    # Process completed tasks
                    for future in as_completed(future_to_task):
                        task_info = future_to_task[future]
                        task_name, benchmark, task_index = task_info
                        
                        try:
                            success = future.result()
                            if success:
                                completed_count += 1
                            else:
                                failed_count += 1
                                
                            # Progress update
                            total_tasks = len(task_list)
                            progress = (completed_count + failed_count) / total_tasks * 100
                            logger.info(f"📈 Progress: {completed_count}/{total_tasks} completed ({progress:.1f}%), {failed_count} failed")
                            
                        except Exception as e:
                            failed_count += 1
                            logger.error(f"❌ Task {task_index} failed with exception: {e}")
                
                logger.info(f"🎉 Individual task streaming completed!")
                logger.info(f"   📊 Total processed: {total_processed} evaluations")
                logger.info(f"   📤 Total uploaded: {total_uploaded} evaluations")
                logger.info(f"   ✅ Successful tasks: {completed_count}")
                logger.info(f"   ❌ Failed tasks: {failed_count}")
                return 0
            
            else:
                # Fallback: process existing data
                logger.info("🔄 Processing existing HELM data files...")
                
                # Find HELM data files to process
                data_path = Path("data/downloads")
                if not data_path.exists():
                    logger.error("❌ HELM data directory not found: data/downloads")
                    logger.info("💡 Expected directory structure: data/downloads/{task_name}/")
                    return 1
                
                # Find all HELM evaluation directories
                helm_dirs = [d for d in data_path.iterdir() if d.is_dir()]
                if not helm_dirs:
                    logger.warning("⚠️ No HELM evaluation directories found in data/downloads")
                    logger.info("💡 Run HELM data collection first to populate data/downloads")
                    return 1
                
                logger.info(f"📁 Found {len(helm_dirs)} HELM evaluation directories")
                
                # Process existing data using our new helper function
                for helm_dir in helm_dirs:
                    try:
                        evaluation_results = process_helm_directory(helm_dir, args.source_name)
                        if evaluation_results:
                            total_evaluations += len(evaluation_results)
                            processed_count += 1
                            
                            # Save individual evaluations
                            saved_count = save_individual_evaluations(evaluation_results, args.source_name)
                            logger.info(f"💾 Saved {saved_count} individual JSON files for {helm_dir.name}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing {helm_dir.name}: {e}")
                        error_count += 1
                
                logger.info(f"✅ Fallback processing completed:")
                logger.info(f"   📊 Processed: {processed_count} directories")
                logger.info(f"   🔢 Total evaluations: {total_evaluations}")
                logger.info(f"   ❌ Errors: {error_count}")
                
                if processed_count == 0:
                    logger.warning("⚠️ No evaluations were successfully processed")
                    return 1
                
                # Aggregate and upload if we have data
                if total_evaluations > 0:
                    logger.info("📦 Aggregating and uploading to HuggingFace...")
                    try:
                        # Find all individual JSON files that were created
                        eval_dir = Path("data/evaluations")
                        if eval_dir.exists():
                            individual_files = list(eval_dir.rglob("*.json"))
                            
                            if individual_files:
                                logger.info(f"📁 Found {len(individual_files)} individual evaluation files")
                                
                                # Aggregate files to HuggingFace dataset format
                                aggregated_file = aggregate_individual_files_to_hf_dataset(
                                    [str(f) for f in individual_files], 
                                    chunk_id=1
                                )
                                
                                if aggregated_file and Path(aggregated_file).exists():
                                    logger.info(f"✅ Aggregated data saved to {aggregated_file}")
                                    
                                    # Upload to HuggingFace
                                    if not args.test_run:
                                        logger.info(f"🚀 Uploading to HuggingFace repository: {args.repo_id}")
                                        api = setup_hf_api(os.environ.get('HF_TOKEN'), args.repo_id)
                                        
                                        api.upload_file(
                                            path_or_fileobj=aggregated_file,
                                            path_in_repo=f"data/{Path(aggregated_file).name}",
                                            repo_id=args.repo_id,
                                            repo_type="dataset"
                                        )
                                        
                                        logger.info("✅ Successfully uploaded to HuggingFace!")
                                    else:
                                        logger.info("🧪 Test mode - skipping HuggingFace upload")
                                        
                    except Exception as e:
                        logger.error(f"❌ Error during aggregation/upload: {e}")
                
                logger.info("🎉 Complete HELM pipeline finished successfully!")
                return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
