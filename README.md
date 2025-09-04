# Every Eval Ever

A comprehensive, extensible data pipeline for collecting, processing, and serving evaluation datasets from multiple sources. Currently supports **HELM** with a modular architecture designed for easy addition of new evaluation sources.

## 📑 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📋 How This Repository Works](#-how-this-repository-works)  
- [🧩 Supported Sources](#-supported-sources)
- [📊 Data Pipeline Overvi## 📈 Usage Examples

### Research Analysis
```python
from datasets import load_dataset
import pandas as pd

# Load the detailed evaluation dataset
dataset = load_dataset("evaleval/every_eval_ever", streaming=True)

# Convert to pandas for analysis
df = pd.DataFrame(dataset['train'])

# Analyze model performance on specific tasks
task_performance = df.groupby(['model_name', 'dataset_name'])['evaluation_score'].mean()
print(task_performance)
```

### Quick Statistics Access
```python
from datasets import load_dataset

# Load the comprehensive statistics dataset
stats = load_dataset("evaleval/every_eval_score_ever")
stats_df = pd.DataFrame(stats['train'])

# View top models across all benchmarks
top_models = stats_df.groupby('model_name')['evaluation_score'].mean().sort_values(ascending=False).head(10)
print(top_models)
```

## 🔧 Contributing

### Adding New Data Sources

1. **Create a new processor** in `src/core/your_source_name/`
2. **Add dataset mapping** in `config/dataset_mappings.json`
3. **Create processing script** following `scripts/optimized_helm_processor.py` pattern
4. **Update workflows** to include your source in automated processing
5. **Test with examples** in `tests/` directory

### Development Workflow

```bash
# Set up development environment
git clone https://github.com/yourusername/every-eval-ever.git
cd every-eval-ever
pip install -r requirements.txt

# Run code quality checks
black . && isort .
mypy src/

# Test processing pipeline
python scripts/optimized_helm_processor.py

# Generate test statistics
python scripts/generate_comprehensive_stats.py
```line-overview)
- [⚙️ Configuration](#️-configuration)
- [🤖 Automation](#-automation)
- [🔧 Adding New Data Sources](#-adding-new-data-sources)
- [💻 Development](#-development)
- [📈 Usage Examples](#-usage-examples)
- [🛠️ Output Format](#️-output-format)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙋 Support](#-support)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/evaleval/every-eval-ever.git
cd every-eval-ever

# Install dependencies
pip install -r requirements.txt

# Install Playwright for web scraping
playwright install
```

### Basic Usage
```bash
# The main processing is now automated via GitHub Actions
# But you can still run components manually for testing:

# Test HELM web scraping for a specific benchmark
python -m src.core.helm.web_scraper lite

# Run optimized HELM processing (chunked, with uploads)
python scripts/optimized_helm_processor.py \
  --benchmark lite \
  --chunk-size 200 \
  --max-workers 2 \
  --repo-id evaleval/every_eval_ever \
  --source-name helm

# Generate comprehensive statistics from uploaded data
python scripts/generate_comprehensive_stats.py \
  --main-repo-id evaleval/every_eval_ever \
  --stats-repo-id evaleval/every_eval_score_ever
```

### Accessing Processed Data
The processed data is available on HuggingFace Hub in two datasets:

#### Detailed Evaluation Data
```python
from datasets import load_dataset

# Load the complete detailed dataset (individual evaluation records)
dataset = load_dataset("evaleval/every_eval_ever")

# Stream for large datasets (recommended)
dataset = load_dataset("evaleval/every_eval_ever", streaming=True)

# Filter for specific sources
helm_data = dataset.filter(lambda x: x['source'] == 'helm')
```

#### Comprehensive Statistics (Leaderboard Data)
```python
# Load benchmark-model performance statistics with smart indexing
stats_dataset = load_dataset("evaleval/every_eval_score_ever")

# Files are organized by source with smart indexing:
# helm-00001.parquet, helm-00002.parquet, eval_harness-00001.parquet, etc.

# Get top performers across all benchmarks
top_performers = stats_dataset['train'].sort('accuracy', reverse=True)

# Filter for specific datasets
gsm8k_stats = stats_dataset['train'].filter(lambda x: x['dataset_name'] == 'gsm8kscenario')
```

#### Data Overview
- **`evaleval/every_eval_ever`**: Detailed individual evaluation records with smart chunking (`data-00001.parquet`, `data-00002.parquet`, etc.)
- **`evaleval/every_eval_score_ever`**: Comprehensive benchmark-model statistics with per-source indexing (`helm-00001.parquet`, etc.)
- **Update Frequency**: Data processed weekly with immediate comprehensive stats generation
- **Schema**: Standardized across all evaluation sources with rich metadata

## 📋 How This Repository Works

### 1. **Multi-Source Architecture**
Each evaluation source (like HELM) has its own processing pipeline while sharing common utilities:

```
src/
├── core/                      # Shared processing utilities
│   ├── aggregator.py         # Combines data into Parquet shards
│   ├── converter.py          # Base conversion utilities
│   ├── scrape_all_to_parquet.py  # Main processing orchestrator
│   └── helm/                 # HELM-specific processing
│       ├── downloader.py     # Downloads evaluation files
│       └── web_scraper.py    # Scrapes HELM website for tasks
├── utils/                     # Common utility functions
│   ├── data_loading.py       # Data loading and saving utilities
│   ├── evaluation_utils.py   # Evaluation processing helpers
│   └── model_utils.py        # Model metadata handling
└── scripts/                   # Automation and workflow scripts
    ├── optimized_helm_processor.py      # Main HELM processing pipeline
    └── generate_comprehensive_stats.py  # Statistics generation
```

### 2. **Optimized Processing Interface**
The system uses an optimized chunked processing approach:

```bash
# Optimized HELM processor with chunking and parallel uploads
python scripts/optimized_helm_processor.py \
  --benchmark {lite|mmlu|classic} \
  --chunk-size 200 \
  --max-workers 2 \
  --repo-id evaleval/every_eval_ever \
  --source-name helm

# Comprehensive statistics generation (processes all sources)
python scripts/generate_comprehensive_stats.py \
  --main-repo-id evaleval/every_eval_ever \
  --stats-repo-id evaleval/every_eval_score_ever
```

### 3. **Standardized Output Schema**
All sources produce data with the same schema for consistency:

```
evaluation_id          # Unique identifier
dataset_name           # Name of the dataset/task  
model_name            # Model being evaluated
model_family          # Model family/organization
raw_input             # Input text/prompt
ground_truth          # Expected answer
output                # Model's actual output
evaluation_score      # Numerical score (0.0-1.0)
evaluation_method_name # Method used for evaluation
source                # Source name (e.g., 'helm')
benchmark             # Benchmark category
hf_split              # HuggingFace dataset split
hf_index              # Index within the split
processing_date       # Date of processing (YYYY-MM-DD)
timestamp             # Processing timestamp
# ... additional metadata fields
```

### 4. **Automated Processing**
- **Weekly Scraping**: GitHub Actions workflow runs weekly to scrape latest data
- **Smart Indexing**: Automatically detects existing files and continues numbering (data-0001.parquet, data-0002.parquet, etc.)
- **Comprehensive Statistics**: After each scraping run, generates per-model-per-benchmark-per-dataset statistics
- **Error Handling**: Robust error handling and retry mechanisms for reliable processing

## 🧩 Supported Sources

### HELM (Holistic Evaluation of Language Models)

Stanford's comprehensive language model evaluation framework.

- **Documentation**: [docs/HELM.md](docs/HELM.md)
- **Benchmarks**: `lite`, `classic`, `mmlu`, and many more
- **Status**: ✅ **Production Ready**

**Example Usage:**
```bash
# Process HELM data with optimized chunked approach
python scripts/optimized_helm_processor.py

# Generate comprehensive statistics  
python scripts/generate_comprehensive_stats.py
```

## 📊 Data Pipeline Overview

### Primary Pipeline (Weekly)
1. **Discovery**: Scrape source websites to find available evaluations
2. **Download**: Fetch raw evaluation data and model outputs  
3. **Convert**: Transform into standardized schema
4. **Aggregate**: Combine into HuggingFace-optimized Parquet files
5. **Upload**: Deploy detailed data to `evaleval/every_eval_ever`

### Statistics Pipeline (Daily)
1. **Download**: Fetch all data shards incrementally (memory-efficient)
2. **Aggregate**: Calculate comprehensive benchmark-model statistics
3. **Upload**: Deploy leaderboard data to `evaleval/every_eval_score_ever`

Each step is modular and can be run independently for debugging. The dual pipeline approach ensures both detailed research data and quick leaderboard access.

## ⚙️ Configuration

The repository uses several configuration files:

- **`config/settings.py`**: Global settings and default values
- **`config/dataset_mappings.json`**: Maps source-specific fields to standard schema
- **`config/model_metadata.csv`**: Database of model information and families
- **`requirements.txt`**: Python dependencies

## 🤖 Automation

The repository uses **dual workflow architecture** for optimal modularity and resource management:

### 📋 Workflow 1: Data Processing & Upload
```yaml
# .github/workflows/scrape_and_upload.yml
- Name: "Weekly Evaluation Data Processing"
- Schedule: Mondays at 03:00 UTC
- Purpose: Process and upload detailed evaluation data
- Current: HELM (lite, mmlu, classic benchmarks)
- Future: Multi-source parallel processing
- Output: Detailed evaluation records + per-shard statistics
- Upload: evaleval/every_eval_ever
- Timeout: 6 hours (for heavy data processing)
```

### 📊 Workflow 2: Comprehensive Statistics Generation
```yaml
# .github/workflows/generate_comprehensive_stats.yml
- Name: "Generate Comprehensive Statistics"
- Schedule: Daily at 06:00 UTC (3 hours after data processing)
- Purpose: Create unified leaderboard across all benchmarks
- Method: Incremental processing (memory-efficient)
- Output: Comprehensive benchmark-model statistics
- Upload: evaleval/every_eval_score_ever
- Timeout: 2 hours (for lighter aggregation)
- Manual Trigger: With source selection and force regenerate options
```

### 🎯 Benefits of Dual Architecture
- **Failure Isolation**: Data and stats workflows are independent
- **Resource Optimization**: Each workflow uses appropriate timeouts
- **Flexible Scheduling**: Weekly data collection, daily stats updates
- **Manual Control**: Trigger stats generation anytime with custom parameters

### Manual Data Processing
For custom processing or testing:

```bash
# Run the optimized HELM processing pipeline locally
python scripts/optimized_helm_processor.py

# Generate comprehensive statistics for all sources
python scripts/generate_comprehensive_stats.py
```

### Manual Statistics Generation
For generating comprehensive statistics:

```bash
# Generate comprehensive statistics from all uploaded data
python scripts/generate_comprehensive_stats.py
```

## 🔧 Adding New Data Sources

This repository is designed to make adding new evaluation sources straightforward. Follow this step-by-step guide:

### Step 1: Create Source Directory Structure
```bash
# Create the new source directory
mkdir src/core/{your_source_name}
cd src/core/{your_source_name}

# Create required files
touch __init__.py
touch downloader.py    # Data downloading and processing
touch web_scraper.py   # Website scraping (if needed)
```

### Step 2: Implement Core Processor
Create `src/core/{your_source_name}/downloader.py`:

```python
#!/usr/bin/env python3
"""
{Your Source Name} evaluation data processor.
"""

import logging
from pathlib import Path

# Import shared utilities
from src.utils.data_loading import save_to_parquet
from src.utils.evaluation_utils import standardize_evaluation_data

def process_source_data():
    """Main processing function for {your_source_name}."""
    try:
        # Your processing logic here
        logger.info(f"Processing {your_source_name} data...")
        
        # 1. Scrape/discover available evaluations
        # 2. Download raw data
        # 3. Convert to standard schema
        # 4. Save as Parquet with smart indexing
        
        logger.info("✅ Processing completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise
```

logger = logging.getLogger(__name__)

def main():
    """Main entry point for {your_source_name} processing."""
    parser = argparse.ArgumentParser(description="Process {Your Source Name} evaluation data")
    
    # Add your source-specific arguments
    parser.add_argument("--benchmark", help="Benchmark name to process")
    parser.add_argument("--output", help="Output parquet file path")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Your processing logic here
        logger.info(f"Processing {args.benchmark} benchmark...")
        
        # 1. Scrape/discover available evaluations
        # evaluations = discover_evaluations(args.benchmark)
        
        # 2. Download raw data
        # raw_data = download_data(evaluations, args.max_workers)
        
        # 3. Convert to standard schema
        # standardized_data = convert_to_schema(raw_data)
        
        # 4. Save as Parquet
        # save_to_parquet(standardized_data, args.output)
        
        logger.info("✅ Processing completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise
```

### Step 3: Add Configuration
Add your source to `config/dataset_mappings.json`:

```json
{
  "your_source_name": {
    "field_mappings": {
      "source_field_name": "standard_field_name"
    },
    "evaluation_methods": ["method1", "method2"],
    "default_benchmark": "default_name"
  }
}
```

### Step 4: Create Processing Script
Create `scripts/your_source_processor.py` following the pattern of `scripts/optimized_helm_processor.py`.

### Step 5: Update Workflows
Add your source to `.github/workflows/scrape_and_upload.yml` to include it in automated processing.

### Step 6: Add Tests
Create test files in `tests/` directory to validate your source processing.

# Test automation script
python scripts/incremental_upload.py \
  --repo-id test/dataset \
  --benchmarks test \
  --source-name your_source_name \
  --max-workers 1
```

### Step 8: Schema Compliance
Ensure your data matches the standardized schema:

**Required Fields:**
- `evaluation_id` (string): Unique identifier
- `dataset_name` (string): Name of the task/dataset
- `model_name` (string): Model being evaluated
- `raw_input` (string): Input text/prompt
- `ground_truth` (string): Expected answer
- `output` (string): Model's actual output
- `evaluation_score` (float): Numerical score
- `source` (string): Your source name
- `ingestion_timestamp` (datetime): When processed

**Optional Fields:**
- `model_family`, `hf_split`, `hf_index`, `evaluation_method_name`, etc.

### Example: OpenAI Evals Integration
Here's a complete example for adding OpenAI Evals:

```bash
# 1. Create structure
mkdir src/sources/openai_evals
touch src/sources/openai_evals/{__init__.py,processor.py,converter.py}

# 2. Implement processor.py
# (See template above)

# 3. Add to main.py
# 'openai_evals': 'src.sources.openai_evals.processor'

# 4. Test
python main.py openai_evals --help
```

### Best Practices for New Sources

1. **Follow the Pattern**: Use existing HELM implementation as reference
2. **Error Handling**: Include comprehensive error handling and logging
3. **Rate Limiting**: Respect source websites' rate limits
4. **Resumability**: Support resuming interrupted downloads
5. **Documentation**: Provide clear usage examples and troubleshooting
6. **Testing**: Include unit tests for critical functions
7. **Schema Compliance**: Ensure data matches the standard schema exactly

### Getting Help

- **Reference Implementation**: Check `src/sources/helm/` for a complete example
- **Schema Details**: See `src/utils/evaluation_utils.py` for schema validation
- **Shared Utilities**: Use functions in `src/core/` and `src/utils/` when possible
- **Issues**: Open a GitHub issue for guidance on specific sources

## 💻 Development

### Testing and Debugging

```bash
# Run tests
python -m pytest tests/ -v

# Test processing components individually
python -c "from src.core.helm.web_scraper import main; main()"
python -c "from src.core.helm.downloader import main; main()"

# Debug with verbose logging
python scripts/optimized_helm_processor.py  # Already includes detailed logging
```

### Debugging Tips
1. **Check logs** in `helm_processor.log` for detailed processing information
2. **Use test data** in `tests/` directory for validation
3. **Monitor GitHub Actions** workflows for automated processing status
4. **Verify outputs** in HuggingFace datasets for data quality

## 📈 Usage Examples

### Research Analysis
```python
from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("evaleval/every_eval_ever", streaming=True)

# Convert to pandas for analysis
df = dataset.to_pandas()

# Compare model performance across datasets
performance = df.groupby(['model_name', 'dataset_name'])['evaluation_score'].mean()

# Find best performing models
top_models = df.groupby('model_name')['evaluation_score'].mean().sort_values(ascending=False)

# Analysis by evaluation method
method_analysis = df.groupby(['evaluation_method_name', 'model_name'])['evaluation_score'].mean()
```

### Model Evaluation Comparison
```python
# Filter for specific models
gpt_models = df[df['model_name'].str.contains('gpt', case=False)]

# Compare across datasets
comparison = gpt_models.pivot_table(
    values='evaluation_score',
    index='dataset_name', 
    columns='model_name',
    aggfunc='mean'
)

# Statistical analysis
from scipy import stats
model_a_scores = df[df['model_name'] == 'gpt-4']['evaluation_score']
model_b_scores = df[df['model_name'] == 'claude-2']['evaluation_score']
statistic, p_value = stats.ttest_ind(model_a_scores, model_b_scores)
```

### Time Series Analysis
```python
# Track evaluation trends over time
df['ingestion_date'] = pd.to_datetime(df['ingestion_timestamp']).dt.date

timeline = df.groupby(['ingestion_date', 'model_name'])['evaluation_score'].mean().reset_index()

# Plot trends
import matplotlib.pyplot as plt
for model in timeline['model_name'].unique():
    model_data = timeline[timeline['model_name'] == model]
    plt.plot(model_data['ingestion_date'], model_data['evaluation_score'], label=model)

plt.legend()
plt.title('Model Performance Over Time')
plt.xlabel('Date')
plt.ylabel('Average Evaluation Score')
## 🛠️ Output Format

The processed data is delivered through **two complementary datasets** optimized for different use cases:

### Dataset 1: Detailed Evaluation Records (`evaleval/every_eval_ever`)

**Purpose**: Individual evaluation instances for deep analysis and research

#### File Naming Convention with Smart Indexing
```
data-0001.parquet    # First chunk of processed data
data-0002.parquet    # Second chunk (automatically incremented)
data-0003.parquet    # Third chunk, etc.
```

#### Parquet Optimization
- **Compression**: SNAPPY for fast read/write
- **Row Groups**: Optimized for streaming access
- **Schema**: Consistent across all sources
- **Smart Chunking**: Automatic file size management with incremental indexing

#### Data Schema
Each record contains standardized evaluation information:

```python
{
  # Core evaluation data
  'evaluation_id': 'unique_hash_identifier',
  'dataset_name': 'openbookqa',
  'model_name': '01-ai/yi-34b',
  'model_family': 'yi-34b',
  'raw_input': 'Question text...',
  'ground_truth': 'Expected answer',
  'output': 'Model response',
  'evaluation_score': 1.0,
  'evaluation_method_name': 'exact_match',
  
  # Metadata and provenance
  'source': 'helm',
  'benchmark': 'lite',
  'hf_split': 'test',
  'hf_index': 4957,
  'processing_date': '2025-01-15',
  'timestamp': '2025-01-15T14:30:00Z'
}
```

### Dataset 2: Comprehensive Statistics (`evaleval/every_eval_score_ever`)

**Purpose**: Benchmark-model performance summaries for leaderboards and quick analysis

#### File Naming Convention with Source-Specific Indexing
```
source-helm-0001.parquet    # HELM statistics, chunk 1
source-helm-0002.parquet    # HELM statistics, chunk 2
source-other-0001.parquet   # Other source statistics, chunk 1
```

#### Statistics Schema
Each record represents a model-benchmark-dataset combination:
  'model_name': 'openai/gpt-4-0613',
  'model_family': 'gpt-4',
  'evaluation_method_name': 'label_only_match',
  
  # Performance metrics
  'total_samples': 1000,
  'accuracy': 0.932,           # 93.2%
  'mean_score': 0.932,
  'std_score': 0.251746,
  'min_score': 0.0,
```python
{
  # Identification
  'source': 'helm',
  'dataset_name': 'gsm8kscenario',
  'model_name': 'gpt-4',
  'benchmark': 'lite',
  
  # Performance metrics
  'evaluation_score': 0.847,
  'total_evaluations': 1000,
  'min_score': 0.0,
  'max_score': 1.0,
  
  # Metadata and timestamps
  'processing_date': '2025-01-15',
  'timestamp': '2025-01-15T14:30:00Z',
  'data_freshness_hours': 0
}
```

### Dataset Comparison

| Aspect | Detailed Records | Comprehensive Statistics |
|--------|------------------|-------------------------|
| **Use Case** | Research, debugging, deep analysis | Leaderboards, quick comparisons |
| **Record Type** | Individual evaluations | Model-benchmark-dataset summaries |
| **Indexing** | data-XXXXX.parquet | source-XXXXX.parquet |
| **Update Freq** | Weekly | Daily + after each scraping |
| **Memory Usage** | High (streaming recommended) | Low (fits in memory) |
| **Query Speed** | Slower (large data) | Faster (aggregated data) |

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Types of Contributions
1. **New Data Sources**: Add support for additional evaluation frameworks
2. **Bug Fixes**: Fix issues in existing processing pipelines  
3. **Performance Improvements**: Optimize data processing and storage
4. **Documentation**: Improve guides and API documentation
5. **Testing**: Add comprehensive test coverage

### Contribution Process
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/new-source`
3. **Implement** your changes with tests and documentation
4. **Test** thoroughly with sample data
5. **Submit** a pull request with detailed description

### Code Standards
- **Style**: Use `black` and `isort` for formatting
- **Types**: Add type hints for public functions
- **Tests**: Include unit tests for new functionality
- **Docs**: Update relevant documentation
- **Logging**: Use structured logging with appropriate levels

### Testing New Sources
Before submitting a new source integration:

```bash
# Test the integration
python main.py your_source --benchmark test --verbose

# Validate schema compliance
python -c "
from datasets import Dataset
import pandas as pd
df = pd.read_parquet('your_output.parquet')
dataset = Dataset.from_pandas(df)
### Testing New Sources
Before submitting a new source integration:

```bash
# Test the processing pipeline
python scripts/your_source_processor.py

# Validate schema compliance
python -c "
import pandas as pd
df = pd.read_parquet('your_output.parquet')
print('Required columns:', df.columns.tolist())
print('✅ Schema validation passed')
"

# Test with sample data in tests/ directory
python scripts/your_source_processor.py --test-mode
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/evaleval/every-eval-ever/issues)
- **Discussions**: [GitHub Discussions](https://github.com/evaleval/every-eval-ever/discussions)  
- **Dataset**: [HuggingFace Dataset Page](https://huggingface.co/datasets/evaleval/every_eval_ever)
- **Statistics**: [HuggingFace Statistics Page](https://huggingface.co/datasets/evaleval/every_eval_score_ever)

### FAQ

**Q: How do I access just HELM data?**
```python
dataset = load_dataset("evaleval/every_eval_ever")
helm_data = dataset.filter(lambda x: x['source'] == 'helm')
```

**Q: How do I get the latest statistics?**
```python
stats = load_dataset("evaleval/every_eval_score_ever")
latest_stats = stats['train']
```

**Q: How do I add a custom evaluation source?**
See the [Adding New Data Sources](#-adding-new-data-sources) section above.

**Q: What's the difference between benchmarks and datasets?**
- **Benchmark**: A collection of tasks (e.g., "lite", "mmlu")
- **Dataset**: Individual tasks within benchmarks (e.g., "hellaswag", "winogrande")

---

**🎯 Goal**: Create the most comprehensive, accessible, and standardized evaluation dataset for language model research.

**💡 Vision**: Enable researchers to easily compare models across all major evaluation frameworks through a single, unified interface.
