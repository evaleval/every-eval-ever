# HELM Data Source

**HELM (Holistic Evaluation of Language Models)** provides comprehensive evaluation data across multiple models and datasets. This document covers the HELM-specific processing pipeline.

## Quick Start

To get started with HELM data immediately:

```bash
python main.py helm --benchmark lite
```

Or try other HELM benchmarks:

```bash
python main.py helm --benchmark mmlu
python main.py helm --benchmark classic
```

The system will automatically:
1. Scrape the HELM website to discover all available evaluations
2. Download the raw data files from HELM's storage
3. Convert everything to standardized CSV format

## Available HELM Benchmarks

- **`lite`**: A smaller subset for quick testing and development
- **`mmlu`**: Massive Multitask Language Understanding benchmark
- **`classic`**: Core evaluation tasks from classic NLP benchmarks

## HELM Pipeline Details

### Step 1: Data Discovery (`src/core/web_scraper.py`)
Scrapes the HELM website to build a comprehensive table of all available (model, dataset) evaluation runs.

**What it does:**
- Navigates to HELM's web interface
- Extracts all model-dataset combinations
- Creates CSV files listing available evaluations
- Handles pagination and dynamic content

**Output:** `data/benchmark_lines/helm_{benchmark}.csv`

### Step 2: Data Download (`src/core/downloader.py`) 
Downloads 8 different JSON files for each evaluation run from HELM's Google Cloud Storage.

**JSON Files Downloaded:**
- `run_spec.json`: Run configuration and metadata
- `instances.json`: Input instances and ground truth
- `predictions.json`: Model predictions
- `eval_cache.json`: Cached evaluation results
- `stats.json`: Statistical summaries
- `scenario.json`: Dataset and task information
- `per_instance_stats.json`: Instance-level statistics
- `scenario_state.json`: Processing state information

**Storage:** `data/downloads/{task_name}/`

### Step 3: Data Conversion (`src/core/converter.py`)
Transforms the raw HELM data into a standardized evaluation schema.

**Schema Fields:**
```
evaluation_id          # Unique identifier for each evaluation
dataset_name           # Name of the dataset being evaluated  
hf_split              # HuggingFace split (train/test/validation)
hf_index              # Index within the HuggingFace split
raw_input             # Raw input text or prompt
ground_truth          # Expected/correct answer
model_name            # Name of the model being evaluated
model_family          # Family/type of the model
output                # Model's actual output
evaluation_method_name # Method used for evaluation
evaluation_score      # Numerical score assigned
run                   # Evaluation run identifier
task                  # Specific task within the evaluation
adapter_method        # Method used for model adaptation
source                # Always 'helm' for this data source
source_version        # Version of HELM data
source_url            # URL to original HELM data
ingestion_timestamp   # When data was processed
license               # License information
category              # Evaluation category
```

## HELM-Specific Configuration

### Download Settings
- **Base URL**: HELM uses Google Cloud Storage for data hosting
- **Retry Logic**: Built-in retries for network failures
- **Parallel Downloads**: Configurable worker threads
- **Rate Limiting**: Respects HELM's rate limits

### Data Mapping
HELM data requires specific field mappings:
- Model names are extracted from run specifications
- Dataset names come from scenario configurations
- Evaluation scores are normalized across different metric types
- Input/output pairs are extracted from instances and predictions

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
