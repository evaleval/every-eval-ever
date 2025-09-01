# HELM Processing System

A comprehensive system for downloading, processing, and converting HELM (Holistic Evaluation of Language Models) evaluation data into standardized formats.

## Quick Start

To get started immediately, simply run:

```bash
python main_processor.py --benchmark lite
```

Or try other available benchmarks:

```bash
python main_processor.py --benchmark mmlu
python main_processor.py --benchmark classic
```

The system will automatically:
1. Scrape the HELM website to discover all available evaluations
2. Download the raw data files
3. Convert everything to standardized CSV format

That's it! Your processed data will be saved in the `data/processed/` directory.

## Overview

This system provides a complete pipeline for working with HELM evaluation data:

1. **Step 1**: Builds a table of all relevant (model, dataset) pairs by scraping the HELM website
2. **Step 2**: Downloads multiple JSON files for each (model, dataset) pair from HELM's storage
3. **Step 3**: Converts and merges the data into standardized CSV files with customizable schema

## How It Works

### Core Pipeline

The system operates in three main phases:

1. **Data Discovery** (`src/core/web_scraper.py`): Scrapes the HELM website to build a comprehensive table of all available (model, dataset) evaluation runs
2. **Data Download** (`src/core/downloader.py`): Downloads 8 different JSON files for each evaluation run from HELM's Google Cloud Storage
3. **Data Conversion** (`src/core/converter.py`): Transforms the raw HELM data into a standardized evaluation schema

### Project Structure

```
helm-data-scraper/
├── main_processor.py          # Main entry point
├── src/                       # Source code
│   ├── core/                  # Core functionality
│   │   ├── web_scraper.py     # Step 1: HELM website scraping
│   │   ├── downloader.py      # Step 2: Data downloading
│   │   ├── converter.py       # Step 3: Data conversion
│   │   └── aggregator.py      # Parquet aggregation
│   └── utils/                 # Utility modules
│       ├── data_loading.py
│       ├── dataset_utils.py
│       ├── evaluation_utils.py
│       ├── model_utils.py
│       └── advanced_mapping.py
├── data/                      # Data storage
│   ├── benchmark_lines/       # Scraped task tables
│   │   ├── helm_lite.csv
│   │   ├── helm_mmlu.csv
│   │   └── helm_classic.csv
│   ├── downloads/             # Raw HELM JSON files
│   │   └── {task_name}/
│   │       ├── run_spec.json
│   │       ├── instances.json
│   │       ├── display_requests.json
│   │       ├── display_predictions.json
│   │       ├── stats.json
│   │       ├── per_instance_stats.json
│   │       ├── scenario.json
│   │       └── scenario_state.json
│   └── processed/             # Final CSV outputs
│       ├── lite/              # Example benchmark directories
│       ├── mmlu/              # (additional benchmarks created
│       └── classic/           #  as needed based on HELM website)
├── config/                    # Configuration files
│   ├── settings.py           # Central configuration
│   ├── dataset_mappings.json # Dataset mappings
│   └── model_metadata.csv    # Model metadata
└── README.md
```

**Note**: The benchmark directories (`lite/`, `mmlu/`, `classic/`) are just examples. The system automatically creates directories for any benchmark available on the HELM website.

### Key Components

#### Core Modules (`src/core/`)
- **`web_scraper.py`**: Web scraping module for discovering available evaluations from HELM website
- **`downloader.py`**: Downloads raw HELM data from multiple versions with fallback support
- **`converter.py`**: Converts HELM data to standardized evaluation format
- **`aggregator.py`**: Aggregates data into Parquet format for efficient storage

#### Utility Modules (`src/utils/`)
- **`data_loading.py`**: JSON/CSV file operations and data loading utilities
- **`dataset_utils.py`**: Dataset name extraction and mapping utilities
- **`evaluation_utils.py`**: Evaluation metrics and scoring functions
- **`model_utils.py`**: Model metadata processing and identification
- **`advanced_mapping.py`**: Advanced question mapping for precise dataset alignment

#### Configuration (`config/`)
- **`settings.py`**: Centralized configuration for paths, constants, and system settings
- **`dataset_mappings.json`**: Dataset mapping configurations
- **`model_metadata.csv`**: Model metadata and family information

#### Entry Point
- **`main_processor.py`**: Main orchestrator that runs the entire pipeline with parallel processing

## Output Schema

The final CSV files contain standardized evaluation data with the following structure (customizable in `src/core/converter.py`):

```csv
evaluation_id,dataset_name,hf_split,hf_index,raw_input,ground_truth,model_name,model_family,output,evaluation_method_name,evaluation_score
```

**Key Fields:**
- `evaluation_id`: Unique identifier for each evaluation
- `dataset_name`: Normalized dataset name (e.g., "mmlu.anatomy")
- `hf_split`: Data split (train/test/validation)
- `hf_index`: Question index in the original dataset
- `raw_input`: The question text
- `ground_truth`: Correct answer (A, B, C, D for multiple choice)
- `model_name`: Model identifier
- `model_family`: Model family/architecture
- `output`: Model's predicted response
- `evaluation_method_name`: Evaluation metric used
- `evaluation_score`: Numerical score (0-1)

## Usage

### Basic Usage

```bash
# Process all tasks from a benchmark
python main_processor.py --benchmark lite

# Filter by adapter method
python main_processor.py --benchmark mmlu --adapter-method multiple_choice_joint

# Keep temporary files for debugging
python main_processor.py --benchmark classic --keep-temp
```

### Advanced Configuration

The system supports two modes for dataset processing:

1. **Generic Mode** (default): Works with any dataset using ID parsing
2. **Advanced Mapping Mode**: Uses JSON mapping files for precise question matching

### Customizing Output Schema

To modify the output format, edit the field ordering in `src/core/converter.py`:

```python
# In process_helm_data() function, modify this section:
ordered_fields = [
    'evaluation_id', 'dataset_name', 'hf_split', 'hf_index',
    'raw_input', 'ground_truth', 'model_name', 'model_family',
    'output', 'evaluation_method_name', 'evaluation_score'
    # Add or remove fields as needed
]
```

## Key Features

- **Parallel Processing**: Uses ProcessPoolExecutor for efficient multi-core processing
- **Version Fallback**: Automatically tries multiple HELM versions to find complete data
- **Flexible Schema**: Easily customizable output format
- **Robust Error Handling**: Continues processing even if individual tasks fail
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Dataset Agnostic**: Works with any HELM dataset without code changes

## Evaluation Metrics

The system supports a comprehensive set of evaluation metrics defined in `src/utils/evaluation_utils.py`. The current metrics were manually curated from observed HELM data and include:

- `exact_match` / `label_only_match`: Direct choice comparison
- `quasi_exact_match` / `quasi_label_only_match`: Choice comparison with tolerance
- `final_number_exact_match` / `final_number_match`: Numerical answer comparison
- `math_equiv_chain_of_thought`: Mathematical equivalence evaluation
- `f1_score`: F1 score calculation
- `exact_match_with_references`: Reference-based exact matching
- `quasi_exact_match_with_references`: Reference-based quasi matching
- `bleu_4`: BLEU-4 score
- `rouge_l`: ROUGE-L score
- `meteor`: METEOR score

**Note**: These metrics were manually identified from HELM data. Additional metrics may exist in the raw files that aren't currently supported. If you encounter evaluation errors, check the raw JSON files for other metric names and add them to the `get_evaluation_metrics()` function in `src/utils/evaluation_utils.py`.

## Configuration

Edit `config/settings.py` to customize:
- Download paths and directories
- HELM version ranges
- Concurrency settings
- Progress bar formatting

The system is designed to be completely generic - it works with any new HELM dataset without requiring code modifications.

## Project Organization

This project follows a clean, modular structure:

- **`src/`**: All source code is organized under this directory following Python best practices
- **`data/`**: All data files are separated from code, with clear distinction between raw and processed data
- **`config/`**: Configuration files are centralized for easy management
- **`main_processor.py`**: Single entry point at the project root for easy execution

This organization makes the project easy to understand, maintain, and extend.

