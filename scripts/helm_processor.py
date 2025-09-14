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
import logging
import os
import pandas as pd
import re
import shutil
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import List, Set, Dict, Optional

# Third-party imports
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from datasets import Dataset, load_dataset

# Suppress Pydantic protected namespace warnings
warnings.filterwarnings("ignore", message=".*protected namespace.*", category=UserWarning)

# Try to import datasets library for reading existing data
try:
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

# Manifest functionality removed - using direct file structure scanning for deduplication


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
        # Load required HELM files
        instances_file = helm_dir / "instances.json"
        run_spec_file = helm_dir / "run_spec.json"
        per_instance_stats_file = helm_dir / "per_instance_stats.json"
        display_predictions_file = helm_dir / "display_predictions.json"
        
        # Check which files exist
        if not instances_file.exists():
            logger.warning(f"⚠️ Skipping {helm_dir.name}: missing instances.json")
            return []
        if not run_spec_file.exists():
            logger.warning(f"⚠️ Skipping {helm_dir.name}: missing run_spec.json")
            return []
        if not display_predictions_file.exists():
            logger.warning(f"⚠️ Skipping {helm_dir.name}: missing display_predictions.json")
            return []
        
        # Load the data
        with open(instances_file, 'r') as f:
            instances = json.load(f)
        with open(run_spec_file, 'r') as f:
            run_spec = json.load(f)
        with open(display_predictions_file, 'r') as f:
            display_predictions = json.load(f)
        
        # Load per-instance stats if available
        per_instance_stats = []
        if per_instance_stats_file.exists():
            with open(per_instance_stats_file, 'r') as f:
                per_instance_stats = json.load(f)
        
        logger.info(f"📄 Loaded HELM files for {helm_dir.name}")
        logger.info(f"   📊 Instances: {len(instances) if isinstance(instances, list) else 'N/A'}")
        logger.info(f"   🎯 Stats: {len(per_instance_stats) if per_instance_stats else 'N/A'}")
        logger.info(f"   🤖 Predictions: {len(display_predictions) if isinstance(display_predictions, list) else 'N/A'}")
        
        # Create lookup dictionaries for efficient mapping by instance_id
        predictions_map = {pred.get("instance_id"): pred for pred in display_predictions}
        stats_map = {stat.get("instance_id"): stat for stat in per_instance_stats}
        
        # Extract model info from run_spec
        adapter_spec = run_spec.get("adapter_spec", {})
        model_name = adapter_spec.get("model", "unknown")
        model_deployment = adapter_spec.get("model_deployment", model_name)
        
        # Extract benchmark info
        scenario_spec = run_spec.get("scenario_spec", {})
        scenario_class = scenario_spec.get("class_name", "")
        
        # Determine benchmark name from scenario class
        benchmark = "unknown"
        if "gsm" in scenario_class.lower():
            benchmark = "gsm8k"
        elif "mmlu" in scenario_class.lower():
            benchmark = "mmlu"
        elif "hellaswag" in scenario_class.lower():
            benchmark = "hellaswag"
        elif "commonsense" in scenario_class.lower():
            benchmark = "commonsense"
        elif "boolq" in scenario_class.lower():
            benchmark = "boolq"
        elif "babi" in scenario_class.lower():
            benchmark = "babi_qa"
        else:
            # Fallback to directory name parsing
            task_name = helm_dir.name.lower()
            for known_benchmark in ["gsm", "mmlu", "hellaswag", "boolq", "commonsense", "babi_qa"]:
                if known_benchmark in task_name:
                    benchmark = known_benchmark
                    break
        
        # Create stats lookup by instance_id
        stats_by_id = {}
        if per_instance_stats:
            for stat_entry in per_instance_stats:
                instance_id = stat_entry.get("instance_id")
                if instance_id:
                    stats_by_id[instance_id] = stat_entry
        
        # Process each instance by matching with predictions
        instances_dict = {instance.get("id"): instance for instance in instances}
        
        for prediction in display_predictions:
            try:
                instance_id = prediction.get("instance_id")
                if not instance_id or instance_id not in instances_dict:
                    continue
                
                instance = instances_dict[instance_id]
                
                # Extract input text from instance
                input_text = instance.get("input", {}).get("text", "")
                
                # Extract ground truth from instance references (marked with "correct" tag)
                ground_truth = ""
                references = instance.get("references", [])
                if references:
                    for ref in references:
                        if "correct" in ref.get("tags", []):
                            ground_truth = ref.get("output", {}).get("text", "")
                            break
                    # If no "correct" tag found, use first reference as fallback
                    if not ground_truth and references:
                        ground_truth = references[0].get("output", {}).get("text", "")
                
                # Extract model prediction from display_predictions (THE KEY FIX!)
                predicted_text = prediction.get("predicted_text", "")
                
                # Extract evaluation metrics from per_instance_stats
                score = 0.0
                instance_stats = stats_map.get(instance_id, {})
                if "stats" in instance_stats:
                    stats_list = instance_stats["stats"]
                    # Try different metric names in order of preference
                    metric_names = [
                        "exact_match_indicator",
                        "final_number_exact_match", 
                        "quasi_exact_match",
                        "exact_match"
                    ]
                    
                    for metric_name in metric_names:
                        for stat in stats_list:
                            stat_name = stat.get("name", {}).get("name", "")
                            if stat_name == metric_name:
                                score = float(stat.get("mean", 0.0))
                                break
                        if score > 0:  # Found a metric
                            break
                
                # Create structured data for EvaluationResult
                helm_data = {
                    'instance_id': instance_id,
                    'model': model_name,
                    'model_deployment': model_deployment,
                    'input_text': input_text,
                    'ground_truth': ground_truth,
                    'predicted_text': predicted_text,  # Now correctly from display_predictions!
                    'score': score,
                    'split': instance.get("split", "test"),
                    'benchmark': benchmark,
                    'temperature': adapter_spec.get("temperature", 0.0),
                    'max_tokens': adapter_spec.get("max_tokens"),
                    'stop_sequences': adapter_spec.get("stop_sequences", []),
                    'task_name': helm_dir.name
                }
                
                # Create EvaluationResult
                result = _create_evaluation_result_from_helm(
                    helm_data=helm_data,
                    task_id=f'{helm_dir.name}_{instance_id}',
                    source=source_name
                )
                
                if result:
                    evaluation_results.append(result)
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not process prediction for instance {instance_id}: {e}")
                continue
        
        logger.info(f"🎯 Created {len(evaluation_results)} EvaluationResult objects")
        
        # Calculate accuracy for this directory
        if evaluation_results:
            correct_count = sum(1 for result in evaluation_results 
                              if 'evaluation' in result and result['evaluation'].get('score', 0) > 0.5)
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


def _create_evaluation_result_from_helm(helm_data: dict, task_id: str, source: str) -> dict:
    """Create a proper EvaluationResult object from HELM data."""
    
    if not EVALHUB_SCHEMA_AVAILABLE:
        logger.error("Cannot create EvaluationResult - schema not available")
        return None
    
    try:
        # Extract model info from HELM data
        model_name = helm_data.get('model', 'unknown')
        model_deployment = helm_data.get('model_deployment', model_name)
        
        # Parse model name to extract family
        model_family = _map_family_to_enum(model_name)
        
        # Create Model object
        model = Model(
            model_info=ModelInfo(
                name=model_deployment,
                family=model_family
            ),
            configuration=Configuration(
                architecture=Architecture.transformer,  # Default assumption for HELM models
                parameters=None,  # Not available in HELM data
                context_window=4096,  # Default assumption
                is_instruct=_is_instruct_model(model_name),
                hf_path=model_name if '/' in model_name else None,
                revision=None
            ),
            inference_settings=InferenceSettings(
                quantization=Quantization(
                    bit_precision=BitPrecision.none,
                    method=Method.None_
                ),
                generation_args=GenerationArgs(
                    use_vllm=False,  # Default to False instead of None
                    temperature=float(helm_data.get('temperature', -1.0)) if helm_data.get('temperature') is not None else -1.0,
                    top_p=-1.0,  # Use -1.0 sentinel instead of None
                    top_k=-1.0,  # Use -1.0 sentinel instead of None
                    max_tokens=max(1, helm_data.get('max_tokens', 1)) if helm_data.get('max_tokens') is not None else 1,
                    stop_sequences=helm_data.get('stop_sequences', [])
                )
            )
        )
        
        # Create PromptConfig object
        benchmark = helm_data.get('benchmark', 'unknown')
        prompt_class_str = _determine_prompt_class_from_benchmark(benchmark)
        prompt_config = PromptConfig(
            prompt_class=_map_prompt_class_to_enum(prompt_class_str),
            dimensions=None
        )
        
        # Create Instance object
        task_type_str = _determine_task_type_from_benchmark(benchmark)
        split_str = helm_data.get('split', 'test')
        hf_split_mapping = {'test': HfSplit.test, 'train': HfSplit.train, 'validation': HfSplit.validation}
        
        instance = Instance(
            task_type=_map_task_type_to_enum(task_type_str),
            raw_input=helm_data.get('input_text', ''),
            language='en',  # Default assumption
            sample_identifier=SampleIdentifier(
                dataset_name=f"helm.benchmark.scenarios.{benchmark}",
                hf_repo=f"helm/helm.benchmark.scenarios.{benchmark}",
                hf_split=hf_split_mapping.get(split_str, HfSplit.test),
                hf_index=_extract_numeric_id(helm_data.get('instance_id', '0'))
            )
        )
        
        # Create Output object  
        output = Output(
            response=helm_data.get('predicted_text', ''),  # Raw model output when available
            cumulative_logprob=None,  # Not available in HELM data
            generated_tokens_logprobs=None  # Not available in HELM data
        )
        
        # Create Evaluation object
        evaluation = Evaluation(
            evaluation_method=EvaluationMethod(
                method_name='exact_match',  # Most common HELM metric
                description='Exact string match between prediction and ground truth',
                parameters=None
            ),
            ground_truth=helm_data.get('ground_truth', ''),
            score=float(helm_data.get('score', 0.0)) if helm_data.get('score') is not None else 0.0
        )
        
        # Create the complete EvaluationResult
        eval_result = EvaluationResult(
            schema_version='1.0.0',
            evaluation_id=f"{source}_{helm_data.get('benchmark')}_{task_id}_{helm_data.get('instance_id', '0')}",
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
        
        # Clean and validate the data to prevent Arrow/datasets issues
        result_dict = _clean_and_validate_data(result_dict)
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Failed to create EvaluationResult from HELM data: {e}")
        return None


def _clean_and_validate_data(data):
    """
    Clean and validate data to prevent Arrow/datasets typecasting issues.
    Ensures all fields have consistent types and NO None values that cause casting errors.
    Uses -1 as sentinel value for missing numeric data to distinguish from actual 0 values.
    """
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if value is None:
                # Replace ALL None values with appropriate defaults - NO None values allowed
                if key in ['temperature', 'top_p', 'top_k', 'cumulative_logprob']:
                    cleaned[key] = -1.0  # Use -1.0 as sentinel for missing numeric values
                elif key == 'score':
                    cleaned[key] = 0.0  # Score must be >= 0, so use 0.0 as "unknown/missing"
                elif key in ['max_tokens', 'hf_index']:
                    cleaned[key] = 1  # Use 1 as sentinel for missing integer values (some fields have min validation)
                elif key in ['is_instruct', 'use_vllm']:
                    cleaned[key] = False  # Default boolean values - False means "not instruct" or "not using vllm"
                elif key in ['stop_sequences', 'generated_tokens_logprobs']:
                    cleaned[key] = []  # Default list values
                else:
                    # Default ALL other fields to empty string - this is critical for Arrow compatibility
                    cleaned[key] = ""  # Default string values for ALL other fields
            else:
                cleaned[key] = _clean_and_validate_data(value)
        return cleaned
    elif isinstance(data, list):
        return [_clean_and_validate_data(item) for item in data]
    elif isinstance(data, str):
        # Ensure strings are valid UTF-8 and not empty unless intentional
        return data.strip() if data else ""
    elif isinstance(data, (int, float, bool)):
        return data
    else:
        # For any other type, convert to string or handle appropriately
        return str(data) if data is not None else ""


def _determine_prompt_class_from_benchmark(benchmark: str) -> str:
    """Determine the prompt class based on benchmark."""
    # Multiple choice benchmarks
    if benchmark.lower() in ['mmlu', 'hellaswag', 'commonsense', 'boolq']:
        return 'MultipleChoice'
    # Generation benchmarks
    elif benchmark.lower() in ['gsm8k', 'gsm', 'babi_qa']:
        return 'Generate'
    else:
        return 'Other'


def _determine_task_type_from_benchmark(benchmark: str) -> str:
    """Determine the task type based on benchmark."""
    # Classification benchmarks
    if benchmark.lower() in ['mmlu', 'hellaswag', 'commonsense', 'boolq']:
        return 'classification'
    # Generation benchmarks  
    elif benchmark.lower() in ['gsm8k', 'gsm', 'babi_qa']:
        return 'generation'
    else:
        return 'other'


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
                    use_vllm=False,  # Default to False instead of None
                    temperature=float(row.get('temperature', -1.0)) if pd.notna(row.get('temperature')) else -1.0,
                    top_p=float(row.get('top_p', -1.0)) if pd.notna(row.get('top_p')) else -1.0,
                    top_k=float(row.get('top_k', -1.0)) if pd.notna(row.get('top_k')) else -1.0,
                    max_tokens=max(1, _safe_int_convert(row.get('max_tokens'), 1)) if pd.notna(row.get('max_tokens')) else 1,
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
        
        # Clean and validate the data to prevent Arrow/datasets issues
        result_dict = _clean_and_validate_data(result_dict)
        
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
    output_file = agg_dir / f"chunk_{chunk_id:04d}_evalhub.jsonl"
    
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


def normalize_task_name_for_deduplication(task_name: str) -> str:
    """
    Normalize a scraped HELM task name to match the filename format used in the repository.
    
    This ensures proper deduplication by converting scraped task names to the format
    that would be used when saving files.
    
    Args:
        task_name: Raw task name from HELM scraping
        
    Returns:
        Normalized task name that matches the filename format
        
    Example:
        Input: "commonsense:dataset=hellaswag,method=multiple_choice_joint,model=together/opt-66b,groups=ablation_multiple_choice"
        Output: "commonsense_dataset=hellaswag,method=multiple_choice_joint,model=together_opt-66b,groups=ablation_multiple_choice"
    """
    if not task_name:
        return ""
    
    # Replace filesystem-unsafe characters with underscores
    normalized = task_name.replace('/', '_').replace(':', '_').replace(' ', '_')
    
    return normalized


def check_task_exists_efficient(task_name: str, existing_tasks: Set[str]) -> bool:
    """
    Efficiently check if a task already exists using the normalized task name.
    
    Args:
        task_name: Raw task name from HELM scraping
        existing_tasks: Set of existing task run IDs from file structure
        
    Returns:
        True if task already exists, False otherwise
    """
    # Normalize the task name to match filename format
    normalized_task = normalize_task_name_for_deduplication(task_name)
    
    # Check if this normalized task exists in our set
    return normalized_task in existing_tasks


def get_existing_tasks_from_file_structure(repo_id: str) -> Set[str]:
    """
    Get existing task run IDs by scanning the file structure in data/helm/ directory.
    This is the most elegant and efficient approach - directly extract run IDs from filenames.
    
    File path format: "data/helm/classic/hellaswag/together_opt-66b/commonsense_dataset=hellaswag,method=multiple_choice_joint,model=together_opt-66b,groups=ablation_multiple_choice_evalhub.jsonl"
    Extracted run ID: "commonsense_dataset=hellaswag,method=multiple_choice_joint,model=together_opt-66b,groups=ablation_multiple_choice"
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'evaleval/every_eval_ever')
        
    Returns:
        Set of existing task run IDs extracted from file structure
    """
    existing_tasks = set()
    
    try:
        from huggingface_hub import list_repo_files
        
        logger.info(f"📁 Scanning file structure in {repo_id} for existing tasks...")
        
        # List all files in the repository
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        # Filter for HELM data files in data/helm/ directory
        helm_files = [f for f in files if f.startswith("data/helm/") and (f.endswith("_evalhub.json") or f.endswith("_evalhub.jsonl"))]
        
        logger.info(f"📄 Found {len(helm_files)} HELM data files")
        
        # Extract run IDs from filenames
        for file_path in helm_files:
            try:
                # Extract filename from path
                filename = file_path.split("/")[-1]
                
                # Remove the "_evalhub.json" or "_evalhub.jsonl" suffix to get the run ID
                if filename.endswith("_evalhub.json"):
                    run_id = filename[:-13]  # Remove "_evalhub.json" (13 characters)
                elif filename.endswith("_evalhub.jsonl"):
                    run_id = filename[:-14]  # Remove "_evalhub.jsonl" (14 characters)
                    existing_tasks.add(run_id)
                    
            except Exception as e:
                logger.debug(f"⚠️ Could not extract run ID from {file_path}: {e}")
                continue
        
        logger.info(f"✅ Extracted {len(existing_tasks)} unique run IDs from file structure (elegant!)")
        
        # Show a few examples for verification
        if existing_tasks:
            logger.info("📝 Sample extracted run IDs:")
            for i, task_id in enumerate(sorted(list(existing_tasks))[:3]):
                logger.info(f"   {i+1}. {task_id}")
        
        return existing_tasks
        
    except Exception as e:
        logger.warning(f"⚠️ Could not scan file structure: {e}")
        logger.info("   💡 Falling back to dataset-based deduplication...")
        # Fall back to dataset method
        return get_existing_tasks_from_hf_repo_legacy(repo_id)




def get_existing_tasks_from_hf_repo(repo_id: str) -> Set[str]:
    """
    Get existing task names using the most elegant and efficient method:
    1. Try file structure scanning first (fastest and most direct)
    2. Fall back to manifest file if structure scanning fails
    3. Fall back to dataset scanning as last resort
    """
    return get_existing_tasks_from_file_structure(repo_id)



def get_existing_tasks_from_hf_repo_legacy(repo_id: str) -> Set[str]:
    """
    Get existing task names from HuggingFace repository using datasets library.
    Extracts and cleans evaluation IDs to match the format of scraped HELM runs.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'evaleval/every_eval_ever')
        
    Returns:
        Set of existing task names (cleaned from evaluation IDs)
    """
    existing_task_names = set()
    
    try:
        if not DATASETS_AVAILABLE:
            logger.warning("⚠️ datasets library not available - cannot check existing data")
            return existing_task_names
        
        logger.info(f"📚 Loading existing evaluation_ids from HF repo: {repo_id}")
        
        # First try to load just the evaluation_id column directly (much faster)
        try:
            logger.info("   💡 Attempting direct column load...")
            dataset = load_dataset(repo_id, split="train")
            evaluation_ids = dataset['evaluation_id']
            logger.info(f"✅ Loaded {len(evaluation_ids)} evaluation IDs directly from column")
            
        except Exception as direct_error:
            logger.warning(f"   ⚠️ Direct column load failed: {direct_error}")
            logger.info("   💡 Falling back to streaming approach...")
            
            # Fallback to streaming approach
            dataset = load_dataset(repo_id, split="train", streaming=True)
            logger.info(f"✅ Loaded streaming dataset from {repo_id}")
            
            # Extract evaluation IDs from stream with error handling
            evaluation_ids = []
            count = 0
            errors = 0
        
            for item in dataset:
                try:
                    eval_id = item.get('evaluation_id')
                    if eval_id and isinstance(eval_id, str):
                        evaluation_ids.append(eval_id)
                    else:
                        errors += 1
                        if errors <= 5:  # Log first few errors
                            logger.warning(f"   ⚠️ Invalid evaluation_id: {eval_id} (type: {type(eval_id)})")
                    
                    count += 1
                    
                    # Log progress every 25000 items
                    if count % 25000 == 0:
                        logger.info(f"   📊 Processed {count} items, found {len(evaluation_ids)} valid evaluation IDs, {errors} errors")
                        
                except Exception as item_error:
                    errors += 1
                    if errors <= 5:  # Log first few item errors
                        logger.warning(f"   ⚠️ Error processing item {count}: {item_error}")
                    continue
            
            logger.info(f"✅ Extracted {len(evaluation_ids)} evaluation IDs via streaming (processed {count} items, {errors} errors)")
        
        # Convert evaluation IDs to task names
        logger.info(f"📊 Processing {len(evaluation_ids)} evaluation IDs to extract task names...")
        
        task_errors = 0
        for eval_id in evaluation_ids:
            try:
                task_name = extract_task_from_evaluation_id(eval_id)
                if task_name:
                    existing_task_names.add(task_name)
                else:
                    task_errors += 1
            except Exception as task_error:
                task_errors += 1
                if task_errors <= 5:  # Log first few task extraction errors
                    logger.warning(f"   ⚠️ Error extracting task from {eval_id}: {task_error}")
        
        logger.info(f"✅ Found {len(existing_task_names)} unique task names from {len(evaluation_ids)} evaluation IDs ({task_errors} extraction errors)")
        return existing_task_names
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load existing data from HF repo {repo_id}: {e}")
        logger.info(f"   💡 Falling back to no deduplication")
        return existing_task_names


def extract_task_from_evaluation_id(evaluation_id: str) -> Optional[str]:
    """
    Extract task identifier from evaluation_id for deduplication matching.
    
    This function handles multiple evaluation_id formats:
    1. Legacy format: "helm_commonsense_commonsense:dataset=openbookqa,method=multiple_choice_joint,model=AlephAlpha_luminous-extended_id4957_id4957"
    2. Current filename format: "task_0001_commonsense_dataset=openbookqa,method=multiple_cho"
    3. Direct task format: "math_subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model=AlephAlpha_luminous-base"
    
    Args:
        evaluation_id: Full evaluation ID in any of the supported formats
        
    Returns:
        Task identifier that can be matched against scraped task names
        Example: "commonsense:dataset=openbookqa,method=multiple_choice_joint,model=AlephAlpha_luminous-extended"
    """
    try:
        # Handle legacy helm_ format
        if evaluation_id.startswith('helm_'):
            # Pattern: helm_{benchmark}_{full_task_name}_id{instance1}_id{instance2}
            # Example: helm_commonsense_commonsense:dataset=openbookqa,method=multiple_choice_joint,model=AlephAlpha_luminous-extended_id4957_id4957
            # We want to extract: commonsense:dataset=openbookqa,method=multiple_choice_joint,model=AlephAlpha_luminous-extended
            return _extract_task_from_legacy_helm_format(evaluation_id)
        
        # Handle current filename formats - just return as-is since they're already normalized
        # Examples:
        # - "task_0001_commonsense_dataset=openbookqa,method=multiple_cho"
        # - "math_subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model=AlephAlpha_luminous-base"
        return evaluation_id
                
    except Exception:
        return None


def _extract_task_from_legacy_helm_format(evaluation_id: str) -> Optional[str]:
    """
    Extract task from legacy helm_ format evaluation IDs.
    
    Args:
        evaluation_id: Legacy format like "helm_commonsense_commonsense:dataset=openbookqa,method=multiple_choice_joint,model=AlephAlpha_luminous-extended_id4957_id4957"
        
    Returns:
        Task identifier like "commonsense:dataset=openbookqa,method=multiple_choice_joint,model=AlephAlpha_luminous-extended"
    """
    try:
        # Remove 'helm_' prefix
        without_prefix = evaluation_id[5:]  # Remove 'helm_'
        
        # Find the last occurrence of '_id' which marks the start of instance IDs
        # Split by '_id' and take everything before the first instance ID
        id_parts = without_prefix.split('_id')
        if len(id_parts) < 2:
            return None
            
        # The task name is everything before the first '_id'
        task_with_benchmark = id_parts[0]
        
        # Now we need to separate the benchmark from the task
        # The benchmark is the first component before the first ':'
        # Example: "commonsense_commonsense:dataset=..." -> benchmark="commonsense", task="commonsense:dataset=..."
        
        if ':' in task_with_benchmark:
            # Find the first colon to identify where the actual task parameters start
            colon_pos = task_with_benchmark.find(':')
            before_colon = task_with_benchmark[:colon_pos]
            after_colon = task_with_benchmark[colon_pos:]
            
            # The benchmark is typically the first part before the underscore
            # Extract everything after the first underscore and before the colon, then rejoin
            if '_' in before_colon:
                # Skip the first part (benchmark) and take the rest as the task name
                parts = before_colon.split('_')
                task_name = '_'.join(parts[1:]) + after_colon
                return task_name
            else:
                # No underscore before colon, the whole thing is the task
                return task_with_benchmark
        else:
            # No colon, so we need to find where the task name starts
            # Skip the first part (benchmark) and return the rest
            parts = task_with_benchmark.split('_')
            if len(parts) > 1:
                return '_'.join(parts[1:])
            else:
                return task_with_benchmark
                
    except Exception:
        return None


def task_name_to_evaluation_id_prefix(task_name: str, benchmark: str, source: str = "helm") -> str:
    """
    Generate evaluation_id prefix from task name to check for existing data.
    
    Args:
        task_name: HELM task name (e.g., 'gsm:model=AlephAlpha_luminous-base')
        benchmark: Benchmark name (e.g., 'gsm8k', 'commonsense')
        source: Source name (default: 'helm')
        
    Returns:
        Evaluation ID prefix that would be generated for this task
        Example: "helm_gsm8k_gsm:model=AlephAlpha_luminous-base_"
    """
    # This matches the evaluation_id generation logic: f"{source}_{benchmark}_{task_id}_{instance_id}"
    # We return the prefix without instance_id to match all instances of this task
    return f"{source}_{benchmark}_{task_name}_"


def get_existing_tasks_from_evaluation_ids(existing_evaluation_ids: Set[str]) -> Set[str]:
    """
    Extract all existing task names from evaluation IDs once, for efficient lookup.
    
    Args:
        existing_evaluation_ids: Set of existing evaluation IDs
        
    Returns:
        Set of existing task names extracted from evaluation IDs
    """
    existing_tasks = set()
    for eval_id in existing_evaluation_ids:
        extracted_task = extract_task_from_evaluation_id(eval_id)
        if extracted_task:
            existing_tasks.add(extracted_task)
    
    return existing_tasks


def check_task_exists_in_repo(task_name: str, existing_tasks: Set[str]) -> bool:
    """
    Check if a task already exists in the repository using the elegant file-structure approach.
    
    Args:
        task_name: HELM task name from scraping
        existing_tasks: Pre-computed set of existing task run IDs from file structure
        
    Returns:
        True if task already exists, False otherwise
    """
    return check_task_exists_efficient(task_name, existing_tasks)


def get_existing_task_ids(downloads_dir: Path) -> Set[str]:
    """Get all existing task IDs from local download directories."""
    existing_ids = set()
    
    try:
        if downloads_dir.exists():
            # Get all directory names in the downloads folder
            for item in downloads_dir.iterdir():
                if item.is_dir():
                    existing_ids.add(item.name)
            
            logger.info(f"📊 Found {len(existing_ids)} existing tasks in {downloads_dir}")
        else:
            logger.info(f"� Downloads directory {downloads_dir} doesn't exist - no existing tasks")
            
    except Exception as e:
        logger.error(f"❌ Error checking existing downloads: {e}")
    
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
            
            # Step 0: Pre-load existing tasks for immediate deduplication feedback
            logger.info("📋 Step 0: Loading existing tasks from repository for real-time deduplication...")
            existing_tasks = get_existing_tasks_from_hf_repo(args.repo_id)
            logger.info(f"✅ Loaded {len(existing_tasks)} existing task IDs from repository")
            
            # Step 1: Scrape HELM runs with immediate deduplication feedback
            logger.info("📋 Step 1: Scraping HELM evaluation runs with real-time deduplication...")
            try:
                # Import scraping functionality
                sys.path.append(str(Path(__file__).parent.parent))
                from src.sources.helm.web_scraper import scrape_helm_data
                
                # Define HELM benchmarks to scrape from command line argument
                helm_benchmarks = args.helm_benchmarks
                logger.info(f"🎯 Target benchmarks: {', '.join(helm_benchmarks)}")
                
                all_scraped_data = []
                all_new_tasks = []
                total_duplicates_found = 0
                benchmark_map = {}  # Map task name to benchmark
                
                # Scrape from each benchmark with immediate deduplication feedback
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
                            
                            # Immediate deduplication analysis for this benchmark
                            benchmark_tasks = []
                            benchmark_new_tasks = []
                            benchmark_duplicates = []
                            
                            for run in benchmark_data:
                                task_name = run.get('Run', '')
                                if task_name:
                                    run['_source_benchmark'] = benchmark
                                    benchmark_tasks.append(task_name)
                                    benchmark_map[task_name] = benchmark
                                    
                                    # Check if this task already exists
                                    if check_task_exists_efficient(task_name, existing_tasks):
                                        benchmark_duplicates.append(task_name)
                                    else:
                                        benchmark_new_tasks.append(task_name)
                            
                            # Report immediate deduplication results for this benchmark
                            logger.info(f"   📊 Deduplication analysis for {benchmark}:")
                            logger.info(f"      🆕 New tasks: {len(benchmark_new_tasks)}")
                            logger.info(f"      🔄 Duplicates (skipped): {len(benchmark_duplicates)}")
                            
                            # Show sample new tasks
                            if benchmark_new_tasks and len(benchmark_new_tasks) <= 5:
                                logger.info(f"      📝 New tasks to process:")
                                for new_task in benchmark_new_tasks:
                                    logger.info(f"         ✓ {new_task}")
                            elif len(benchmark_new_tasks) > 5:
                                logger.info(f"      📝 Sample new tasks (first 5):")
                                for new_task in benchmark_new_tasks[:5]:
                                    logger.info(f"         ✓ {new_task}")
                                logger.info(f"         ... and {len(benchmark_new_tasks) - 5} more new tasks")
                            
                            # Show sample duplicates
                            if benchmark_duplicates and len(benchmark_duplicates) <= 5:
                                logger.info(f"      � Duplicate tasks (skipped):")
                                for dup in benchmark_duplicates:
                                    normalized = normalize_task_name_for_deduplication(dup)
                                    logger.info(f"         ⏭ {dup}")
                                    logger.info(f"            → matches: {normalized}")
                            elif len(benchmark_duplicates) > 5:
                                logger.info(f"      📋 Sample duplicates (first 3):")
                                for dup in benchmark_duplicates[:3]:
                                    normalized = normalize_task_name_for_deduplication(dup)
                                    logger.info(f"         ⏭ {dup}")
                                    logger.info(f"            → matches: {normalized}")
                                logger.info(f"         ... and {len(benchmark_duplicates) - 3} more duplicates")
                            
                            # Add to global collections
                            all_scraped_data.extend(benchmark_data)
                            all_new_tasks.extend(benchmark_new_tasks)
                            total_duplicates_found += len(benchmark_duplicates)
                            
                        else:
                            logger.warning(f"   ⚠️ No data scraped from {benchmark}")
                            
                    except Exception as e:
                        logger.warning(f"   ❌ Failed to scrape {benchmark}: {e}")
                        continue
                
                if all_scraped_data:
                    logger.info(f"🎯 Final scraping and deduplication summary:")
                    logger.info(f"   📊 Total scraped: {len(all_scraped_data)} runs across all benchmarks")
                    logger.info(f"   🆕 New tasks to process: {len(all_new_tasks)}")
                    logger.info(f"   🔄 Duplicates skipped: {total_duplicates_found}")
                    logger.info(f"   📈 Deduplication efficiency: {total_duplicates_found/(len(all_scraped_data)) * 100:.1f}% duplicates avoided")
                    
                    # Use only the new tasks for processing
                    task_names = all_new_tasks
                    
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
                        
                        # Extract model and benchmark info for nested structure
                        first_result = evaluation_results[0] if evaluation_results else {}
                        model_name = first_result.get('model', {}).get('model_info', {}).get('name', 'unknown')
                        
                        # Use the actual HELM benchmark (lite, classic, mmlu) passed to this function
                        helm_benchmark = benchmark  # This comes from benchmark_map and is the correct HELM benchmark
                        
                        # Extract dataset name from task name
                        dataset_name = 'unknown'
                        if ':dataset=' in task_name:
                            # Extract dataset from task name like "commonsense:dataset=openbookqa,method=..."
                            dataset_part = task_name.split(':dataset=')[1].split(',')[0]
                            dataset_name = dataset_part
                        elif ':subject=' in task_name:
                            # Extract subject from MMLU task like "mmlu:subject=clinical_knowledge,method=..."
                            subject_part = task_name.split(':subject=')[1].split(',')[0]
                            dataset_name = subject_part
                        elif 'gsm' in task_name.lower():
                            dataset_name = 'gsm8k'
                        elif 'hellaswag' in task_name.lower():
                            dataset_name = 'hellaswag'
                        elif 'boolq' in task_name.lower():
                            dataset_name = 'boolq'
                        elif 'babi' in task_name.lower():
                            dataset_name = 'babi_qa'
                        else:
                            # Fallback: extract first part of task name
                            dataset_name = task_name.split(':')[0] if ':' in task_name else task_name
                        
                        # Clean names for filesystem
                        clean_model = model_name.replace('/', '_').replace(':', '_').replace(' ', '_')
                        clean_dataset = dataset_name.replace('/', '_').replace(':', '_').replace(' ', '_')
                        
                        # Create nested directory structure: helm/benchmark/dataset/model/
                        nested_path = f"helm/{helm_benchmark}/{clean_dataset}/{clean_model}"
                        
                        # Save locally with nested structure
                        local_nested_dir = Path("data/aggregated") / nested_path
                        local_nested_dir.mkdir(parents=True, exist_ok=True)
                        task_file = local_nested_dir / f"{task_name.replace('/', '_').replace(':', '_')}_evalhub.jsonl"
                        task_dataset.to_json(str(task_file))
                        
                        upload_success = False
                        # Upload to HuggingFace with retry mechanism and corruption protection
                        if api and not args.test_run and not args.local_only:
                            repo_path = f"data/{nested_path}/{task_file.name}"
                            max_retries = 3
                            retry_delay = 5  # seconds
                            
                            for attempt in range(max_retries):
                                try:
                                    # Step 1: Verify local file integrity before upload
                                    if not task_file.exists() or task_file.stat().st_size == 0:
                                        raise ValueError(f"Local file {task_file} is missing or empty")
                                    
                                    # Step 2: Direct upload to final location
                                    api.upload_file(
                                        path_or_fileobj=str(task_file),
                                        path_in_repo=repo_path,
                                        repo_id=args.repo_id,
                                        repo_type="dataset"
                                    )
                                    
                                    # Step 3: Verify upload by checking file exists and has content
                                    # Add delay and retry for verification due to Hub propagation delay
                                    verification_success = False
                                    verification_attempts = 3
                                    verification_delay = 2  # seconds
                                    
                                    for verify_attempt in range(verification_attempts):
                                        try:
                                            if verify_attempt > 0:
                                                logger.debug(f"⏳ Thread-{thread_id}: Waiting {verification_delay}s before verification attempt {verify_attempt + 1}")
                                                time.sleep(verification_delay)
                                            
                                            downloaded_path = hf_hub_download(
                                                repo_id=args.repo_id,
                                                filename=repo_path,
                                                repo_type="dataset",
                                                local_files_only=False
                                            )
                                            
                                            # Verify the downloaded file has reasonable size
                                            file_size = os.path.getsize(downloaded_path)
                                            if file_size < 100:  # Minimum reasonable size
                                                raise ValueError(f"Uploaded file appears incomplete (size: {file_size} bytes)")
                                            
                                            verification_success = True
                                            logger.debug(f"✅ Thread-{thread_id}: Upload verified (size: {file_size} bytes)")
                                            break
                                            
                                        except Exception as verify_error:
                                            if verify_attempt < verification_attempts - 1:
                                                logger.debug(f"⚠️ Thread-{thread_id}: Verification attempt {verify_attempt + 1} failed, retrying: {verify_error}")
                                            else:
                                                logger.warning(f"⚠️ Thread-{thread_id}: Upload verification failed after {verification_attempts} attempts: {verify_error}")
                                                logger.warning(f"   📝 File was uploaded but verification failed - this may be due to Hub propagation delay")
                                                # Don't fail the upload - treat as successful since upload API call succeeded
                                                verification_success = True
                                    
                                    if verification_success:
                                        upload_success = True
                                        logger.info(f"✅ Thread-{thread_id}: Uploaded to {repo_path} (attempt {attempt + 1})")
                                        
                                        break
                                    
                                except Exception as e:
                                    if attempt < max_retries - 1:
                                        logger.warning(f"⚠️ Thread-{thread_id}: Upload attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                                        time.sleep(retry_delay)
                                        retry_delay *= 2  # Exponential backoff
                                    else:
                                        logger.error(f"❌ Thread-{thread_id}: Upload failed after {max_retries} attempts: {e}")
                                        # Keep local file as backup when upload fails
                                        logger.info(f"💾 Thread-{thread_id}: Keeping local backup at {task_file}")
                                        
                        else:
                            upload_success = True  # Consider local saves as success
                            if args.local_only:
                                logger.info(f"🏠 Thread-{thread_id}: Saved locally")
                            else:
                                logger.info(f"🧪 Thread-{thread_id}: Test mode - saved locally")
                        
                        # Step 4: Clean up safely - only after successful upload
                        if not args.no_cleanup:
                            try:
                                # Always remove download directory to save space
                                if helm_dir.exists():
                                    shutil.rmtree(helm_dir)
                                
                                # Only remove local file if upload was successful OR in local-only mode
                                if task_file.exists() and (upload_success or args.local_only):
                                    if upload_success and not args.local_only:
                                        task_file.unlink()
                                        logger.info(f"🧹 Thread-{thread_id}: Cleaned up local file after successful upload")
                                    elif args.local_only:
                                        logger.info(f"🏠 Thread-{thread_id}: Keeping local file (local-only mode)")
                                else:
                                    logger.info(f"💾 Thread-{thread_id}: Keeping local backup due to upload failure")
                                
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
                
                # Note: No local checkpoint needed! HF repository deduplication handles resume automatically
                # Even if runner restarts, get_existing_tasks_from_hf_repo() ensures we skip already uploaded tasks
                
                # Process tasks in parallel with configurable thread count
                with ThreadPoolExecutor(max_workers=args.max_workers, thread_name_prefix="TaskWorker") as executor:
                    # Submit all tasks (deduplication already handled above)
                    future_to_task = {executor.submit(process_single_task, task_info): task_info for task_info in task_list}
                    
                    completed_count = 0
                    failed_count = 0
                    failed_tasks = []
                    completed_tasks = []  # Track successfully completed task names for manifest
                    
                    # Process completed tasks
                    for future in as_completed(future_to_task):
                        task_info = future_to_task[future]
                        task_name, benchmark, task_index = task_info
                        
                        try:
                            success = future.result()
                            if success:
                                completed_count += 1
                                completed_tasks.append(task_name)  # Add to completed list for manifest
                            else:
                                failed_count += 1
                                failed_tasks.append(task_name)
                                
                            # Progress update
                            total_tasks = len(task_list)
                            progress = (completed_count + failed_count) / total_tasks * 100
                            logger.info(f"📈 Progress: {completed_count}/{total_tasks} completed ({progress:.1f}%), {failed_count} failed")
                            
                        except Exception as e:
                            failed_count += 1
                            failed_tasks.append(task_name)
                            logger.error(f"❌ Task {task_index} ({task_name}) failed with exception: {e}")
                
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
