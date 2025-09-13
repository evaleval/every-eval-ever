# HELM Data Source

**HELM (Holistic Evaluation of Language Models)** provides comprehensive evaluation data across multiple models and datasets. This source is fully integrated with the EvalHub schema format.

## 🚀 Quick Start

```bash
# Test processing (works immediately)
python scripts/helm_processor.py --test-run

# Full processing (requires HELM data collection first)
python scripts/helm_processor.py --repo-id evaleval/every_eval_ever

# GitHub Actions (automated weekly)
# See .github/workflows/scrape_and_upload.yml
```

### ⚠️ Current Status
- ✅ **Test Mode**: Fully implemented and working
- ⚠️ **Production Mode**: Basic implementation (needs HELM data integration)
- 🔧 **Data Collection**: Requires HELM scraping/download implementation

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
python scripts/helm_processor.py \
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
python scripts/helm_processor.py --test-run

# Test with specific benchmarks
python scripts/helm_processor.py --test-run \
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

### ⚠️ Current Workflow Status
The workflow currently runs but requires HELM data collection implementation:
- ✅ Schema setup works correctly
- ⚠️ Production processing detects missing HELM data
- 🔧 Needs integration with HELM data collection pipeline

### Workflow Configuration
```yaml
- name: Setup EvalHub Schema
  run: |
    chmod +x setup_schemas.sh
    ./setup_schemas.sh

- name: Run HELM Processing
  run: |
    python scripts/helm_processor.py \
      --repo-id "${HF_REPO_ID}" \
      --chunk-size 25 \
      --max-workers 2 \
      --source-name helm
```

## 📈 Data Flow

### Current Implementation
1. **Schema Setup**: ✅ Download EvalHub schema from GitHub
2. **Model Mapping**: ✅ Load model family mappings from config
3. **Data Discovery**: ⚠️ Looks for HELM files in `data/downloads/` (manual setup required)
4. **Individual Processing**: ✅ Create separate JSON file per evaluation
5. **Deduplication**: ✅ Skip existing files automatically
6. **Aggregation**: ✅ Combine individual files for HuggingFace upload
7. **Upload**: ✅ Push to evaleval/every_eval_ever dataset

### Missing Components (TODO)
- **HELM Data Collection**: Web scraping or API integration to populate `data/downloads/`
- **Data Parsing**: Convert HELM JSON files to EvalHub format
- **Benchmark Discovery**: Automatic detection of available HELM benchmarks

### Expected Directory Structure
```
data/downloads/
├── {task_name_1}/
│   ├── instances.json      # Input data
│   ├── predictions.json    # Model outputs  
│   ├── run_spec.json      # Task configuration
│   ├── eval_cache.json    # Evaluation results
│   └── stats.json         # Statistical summaries
├── {task_name_2}/
│   └── ...
```

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
PYTHONUNBUFFERED=1 python scripts/helm_processor.py --test-run
```

### Note on Warnings
The processor automatically suppresses Pydantic warnings about protected namespaces (related to `model_info` fields) as these are expected with the EvalHub schema structure.

## 📚 Additional Resources

- **EvalHub Schema**: https://github.com/evaleval/evalHub
- **HELM Project**: https://crfm.stanford.edu/helm/
- **HuggingFace Dataset**: https://huggingface.co/datasets/evaleval/every_eval_ever
- **Test Examples**: `tests/test_evalhub_creation.py`

---

For general repository information, see the main [README.md](../README.md).
