# HELM Data Source

**HELM (Holistic Evaluation of Language Models)** provides comprehensive evaluation data across multiple models and datasets. This source is fully integrated with the EvalHub schema format.

## 🚀 Quick Start

```bash
# Test processing
python scripts/simple_helm_processor_evalhub.py --test-run

# Full processing 
python scripts/simple_helm_processor_evalhub.py --repo-id evaleval/every_eval_ever

# GitHub Actions (automated weekly)
# See .github/workflows/scrape_and_upload.yml
```

## 📊 HELM Output Format

All HELM evaluations are processed into EvalHub-compliant individual JSON files:

```json
{
  "evaluation_id": "mmlu:subject=clinical_knowledge,method=multiple_choice_joint,model=01-ai_yi-6b",
  "benchmark": "mmlu",
  "model_info": {
    "model_name": "01-ai/yi-6b",
    "model_family": "yi"
  },
  "instance_result": {
    "instance_id": 123,
    "input": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
    "output": "paralysis of the muscles of facial expression.",
    "references": ["paralysis of the muscles of facial expression."],
    "score": 1.0,
    "is_correct": true
  },
  "source": "helm"
}
```

## 📁 File Organization

Individual files are saved in nested structure:
```
data/evaluations/helm/
├── mmlu/
│   ├── clinical_knowledge/
│   │   ├── HELM_mmlu_clinical_knowledge_eval123_inst456.json
│   │   └── HELM_mmlu_clinical_knowledge_eval124_inst457.json
│   └── philosophy/
│       └── HELM_mmlu_philosophy_eval125_inst458.json
└── hellaswag/
    └── default/
        └── HELM_hellaswag_default_eval126_inst459.json
```

## ⚡ Performance Features

### Individual File Benefits
- **O(1) Deduplication**: File existence check, no database queries
- **Perfect Parallelization**: Each file processes independently  
- **Efficient Updates**: Only process new evaluations
- **HuggingFace Compatible**: Can aggregate to datasets format

### Processing Configuration
```bash
python scripts/simple_helm_processor_evalhub.py \
  --repo-id evaleval/every_eval_ever \
  --chunk-size 25 \
  --max-workers 2 \
  --source-name helm
```

## 🔧 Implementation Details

### Core Functions

#### `_create_evaluation_result()`
Converts HELM data to EvalHub format:
- Maps model names to families using `config/model_metadata.csv`
- Preserves original model names even if not in family enum
- Handles multiple evaluation metrics per instance
- Creates proper nested object structure

#### `save_individual_evaluations()`
Saves each evaluation as separate JSON file:
- Creates nested folder structure by source/benchmark/subject
- Uses unique filename: `{source}_{benchmark}_{subject}_{eval_id}_{instance_id}.json`
- Automatic deduplication via file existence checks

#### `process_chunk_with_individual_files()`
Parallel processing ready:
- ThreadPoolExecutor support for concurrent processing
- Chunk-based processing for memory efficiency
- Error handling and logging per file

### Model Name Handling
```python
# Model family mapping with fallback
model_family = model_mappings.get(model_name, "other")

# Original name always preserved
model_info = ModelInfo(
    model_name=model_name,  # Always the original name
    model_family=model_family  # Mapped or "other"
)
```

## 🧪 Testing

### Test Individual Processing
```bash
# Process synthetic test data
python scripts/simple_helm_processor_evalhub.py --test-run

# Test with specific benchmarks
python scripts/simple_helm_processor_evalhub.py --test-run \
  --test-benchmarks mmlu hellaswag boolq \
  --num-files-test 5
```

### Validate Output
```bash
# Check created files
ls data/evaluations/helm/

# Examine structure
cat data/evaluations/helm/test/default/HELM_test_*.json | jq .
```

### Run Test Suite
```bash
# Individual tests
python tests/test_evalhub_creation.py
python tests/test_individual_files.py
python tests/test_nested_structure_demo.py

# All tests
python tests/run_all_tests.py
```

## 🔄 GitHub Actions Integration

The HELM processor runs automatically via GitHub Actions:

### Workflow Features
- **Weekly Schedule**: Runs every Monday at 3 AM UTC
- **Schema Setup**: Automatically downloads EvalHub schema
- **Individual Files**: Creates separate JSON per evaluation
- **HuggingFace Upload**: Aggregates files for dataset hosting
- **Error Handling**: Comprehensive logging and artifact upload

### Workflow Configuration
```yaml
- name: Setup EvalHub Schema
  run: |
    chmod +x setup_schemas.sh
    ./setup_schemas.sh

- name: Run HELM Processing
  run: |
    python scripts/simple_helm_processor_evalhub.py \
      --repo-id "${HF_REPO_ID}" \
      --chunk-size 25 \
      --max-workers 2 \
      --source-name helm
```

## 📈 Data Flow

1. **Schema Setup**: Download EvalHub schema from GitHub
2. **Model Mapping**: Load model family mappings from config
3. **Data Discovery**: Find HELM evaluation files (future: web scraping)
4. **Individual Processing**: Create separate JSON file per evaluation
5. **Deduplication**: Skip existing files automatically
6. **Aggregation**: Combine individual files for HuggingFace upload
7. **Upload**: Push to evaleval/every_eval_ever dataset

## 🛠️ Configuration Files

### `config/model_metadata.csv`
Maps HELM model names to standardized families:
```csv
model_name,model_family
gpt-4,openai
claude-2,anthropic
llama-2-70b,meta
```

### `setup_schemas.sh`
Downloads and sets up EvalHub schema:
```bash
#!/bin/bash
git clone https://github.com/evaleval/evalHub.git external_schemas/evalHub
```

## 🔍 Troubleshooting

### Common Issues

#### Schema Import Error
```
⚠️ EvalHub schema not available: No module named 'eval_types'
```
**Solution**: Run `./setup_schemas.sh` to download schema

#### Missing HF Token
```
❌ HF_TOKEN environment variable required
```
**Solution**: Set `export HF_TOKEN="your_token"` or use `--test-run` for local testing

#### File Processing Errors
```
❌ Error processing file: [Errno 2] No such file or directory
```
**Solution**: Check file paths and ensure data exists in expected locations

### Debug Mode
```bash
# Enable detailed logging
PYTHONUNBUFFERED=1 python scripts/simple_helm_processor_evalhub.py --test-run
```

## 📚 Additional Resources

- **EvalHub Schema**: https://github.com/evaleval/evalHub
- **HELM Project**: https://crfm.stanford.edu/helm/
- **HuggingFace Dataset**: https://huggingface.co/datasets/evaleval/every_eval_ever
- **Test Examples**: `tests/test_evalhub_creation.py`

---

For general repository information, see the main [README.md](../README.md).

### Common Issues and Solutions

#### 1. Rate Limiting
If you encounter rate limiting:
```bash
python main.py helm --benchmark lite --max-workers 1
```

#### 2. Network Timeouts
For unreliable connections:
```bash
python main.py helm --benchmark lite --keep-temp
```

#### 3. Partial Downloads
To resume interrupted downloads:
```bash
python main.py helm --benchmark lite --overwrite
```

## Advanced Usage

### Custom Benchmark Processing
```bash
# Process specific models only
python main.py helm --benchmark mmlu --filter-models "gpt-3.5,gpt-4"

# Process specific datasets only  
python main.py helm --benchmark classic --filter-datasets "hellaswag,winogrande"

# Increase parallelism
python main.py helm --benchmark lite --max-workers 8
```

### Data Validation
```bash
# Validate processed data
python -m src.utils.evaluation_utils validate --benchmark lite

# Check for missing evaluations
python -m src.utils.dataset_utils check_completeness --benchmark mmlu
```

## File Organization

HELM data is organized as follows:

```
data/
├── benchmark_lines/           # Scraped evaluation catalogs
│   ├── helm_lite.csv         # Lite benchmark tasks
│   ├── helm_mmlu.csv         # MMLU benchmark tasks  
│   └── helm_classic.csv      # Classic benchmark tasks
├── downloads/                 # Raw HELM JSON files
│   └── {task_name}/          # One folder per evaluation task
│       ├── run_spec.json     # Task configuration
│       ├── instances.json    # Input data
│       ├── predictions.json  # Model outputs
│       ├── eval_cache.json   # Evaluation cache
│       ├── stats.json        # Statistical summaries
│       ├── scenario.json     # Dataset metadata
│       ├── per_instance_stats.json  # Instance-level metrics
│       └── scenario_state.json      # Processing state
└── processed/                 # Converted CSV files
    └── {benchmark}/          # One folder per benchmark
        └── {task_files}.csv  # Standardized evaluation data
```

## Integration with HuggingFace Datasets

The processed HELM data is automatically uploaded to HuggingFace Datasets via GitHub Actions:

- **Repository**: `evaleval/every_eval_ever`
- **Update Frequency**: Weekly (Mondays at 03:00 UTC)
- **Format**: Parquet shards optimized for streaming
- **Naming**: `data-XXXXX.parquet` with incremental numbering

Access the live dataset:
```python
from datasets import load_dataset

# Load HELM evaluation data
dataset = load_dataset("evaleval/every_eval_ever")

# Or stream for large datasets
dataset = load_dataset("evaleval/every_eval_ever", streaming=True)

# Filter for HELM data only
helm_data = dataset.filter(lambda x: x['source'] == 'helm')
```

## Troubleshooting

### Common Error Messages

**"Browser timeout while loading HELM page"**
- HELM website may be temporarily unavailable
- Try again later or reduce concurrent requests

**"Failed to download JSON file"**
- Network connectivity issues
- File may have been moved or deleted from HELM storage
- Check HELM's status page

**"Invalid JSON format in downloaded file"**
- Corrupted download or empty file
- Delete the file and re-run with `--overwrite`

**"Model name not found in run_spec.json"**
- HELM changed their data format
- May require updating the field mappings in converter.py

### Getting Help

For HELM-specific issues:
1. Check [HELM's official documentation](https://crfm.stanford.edu/helm/)
2. Review HELM's GitHub repository for known issues
3. Verify that HELM's website and storage are accessible

For processing issues:
1. Enable debug logging: `--verbose`
2. Keep temporary files: `--keep-temp`
3. Check the logs in `helm_processor.log`
